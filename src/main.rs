use std::{collections::HashMap, pin::Pin, sync::Arc};

use anyhow::Context;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use yaca_core::agent::config::AgentConfig;
use yaca_core::agent::orchestrator::{Initializer, OrchestratorAgent};
use yaca_core::agent::{Agent, AgentLifecycleHook};
use yaca_transport::convert;
use yaca_transport::pb::{self, agent_service_server::AgentService};
use yaca_transport::{AgentServiceServer, PAYLOAD_VERSION};

struct Registry<P: Initializer> {
    agents: HashMap<String, Arc<AgentState<P>>>,
    owners: HashMap<String, String>,
}

impl<P: Initializer> Default for Registry<P> {
    fn default() -> Self {
        Self {
            agents: HashMap::new(),
            owners: HashMap::new(),
        }
    }
}

struct AgentState<P: Initializer> {
    agent_id: Arc<str>,
    conversation_id: RwLock<String>,
    agent: Mutex<OrchestratorAgent<P>>,
    fanout: Arc<Fanout>,
    turn_id: Arc<Mutex<Option<String>>>,
    cancel: Mutex<CancellationToken>,
}

struct Subscriber {
    sender: mpsc::Sender<pb::AgentEvent>,
    terminate: mpsc::UnboundedSender<Status>,
}

struct Fanout {
    subscribers: Mutex<Vec<Subscriber>>,
}

impl Default for Fanout {
    fn default() -> Self {
        Self {
            subscribers: Mutex::new(Vec::new()),
        }
    }
}

impl Fanout {
    async fn attach(
        &self,
        sender: mpsc::Sender<pb::AgentEvent>,
    ) -> mpsc::UnboundedReceiver<Status> {
        let (terminate, terminate_rx) = mpsc::unbounded_channel();
        self.subscribers
            .lock()
            .await
            .push(Subscriber { sender, terminate });
        terminate_rx
    }

    async fn broadcast(&self, event: pb::AgentEvent) {
        let mut subscribers = self.subscribers.lock().await;
        let droppable = matches!(&event.kind, Some(pb::agent_event::Kind::UpdateMessage(_)));
        let mut dead = Vec::new();
        for (idx, sub) in subscribers.iter().enumerate() {
            match sub.sender.try_send(event.clone()) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) if droppable => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    let _ = sub.terminate.send(Status::resource_exhausted(
                        "subscription queue overflowed on a non-droppable event",
                    ));
                    dead.push(idx);
                }
                Err(mpsc::error::TrySendError::Closed(_)) => dead.push(idx),
            }
        }
        for idx in dead.into_iter().rev() {
            subscribers.swap_remove(idx);
        }
    }

    async fn close_all(&self) {
        self.subscribers.lock().await.clear();
    }
}

struct IpcLifecycleHook {
    agent_id: Arc<str>,
    fanout: Arc<Fanout>,
    turn_id: Arc<Mutex<Option<String>>>,
}

impl AgentLifecycleHook for IpcLifecycleHook {
    async fn on_switch_conversation(
        &self,
        id: &str,
        memory: Result<Vec<yaca_transport::Message>, yaca_transport::MemoryError>,
    ) -> anyhow::Result<()> {
        let event = match memory {
            Ok(messages) => convert::switch_ok(self.agent_id.to_string(), id, &messages)?,
            Err(err) => convert::switch_error(self.agent_id.to_string(), id, format!("{err:?}")),
        };
        self.fanout.broadcast(event).await;
        Ok(())
    }

    async fn on_new_message(
        &self,
        index: usize,
        message: &yaca_transport::Message,
    ) -> anyhow::Result<()> {
        let turn_id = self.turn_id.lock().await.clone().unwrap_or_default();
        self.fanout
            .broadcast(convert::new_message(
                self.agent_id.to_string(),
                index,
                message,
                turn_id,
            )?)
            .await;
        Ok(())
    }

    async fn on_update_message(
        &self,
        index: usize,
        message: &yaca_transport::MessageUpdate,
    ) -> anyhow::Result<()> {
        let turn_id = self.turn_id.lock().await.clone().unwrap_or_default();
        self.fanout
            .broadcast(convert::update_message(
                self.agent_id.to_string(),
                index,
                message,
                turn_id,
            )?)
            .await;
        Ok(())
    }
}

struct AgentServer<P: Initializer + Send + Sync + 'static> {
    registry: Arc<Mutex<Registry<P>>>,
    config: AgentConfig,
    build: Arc<dyn Fn(&AgentConfig, &str) -> anyhow::Result<P> + Send + Sync>,
}

impl<P: Initializer + Send + Sync + 'static> AgentServer<P> {
    fn new(
        config: AgentConfig,
        build: impl Fn(&AgentConfig, &str) -> anyhow::Result<P> + Send + Sync + 'static,
    ) -> Self {
        Self {
            registry: Arc::new(Mutex::new(Registry::default())),
            config,
            build: Arc::new(build),
        }
    }
}

#[tonic::async_trait]
impl<P: Initializer + Send + Sync + 'static> AgentService for AgentServer<P> {
    type SubscribeStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<pb::AgentEvent, Status>> + Send>>;

    async fn create_agent(
        &self,
        request: Request<pb::CreateAgentRequest>,
    ) -> Result<Response<pb::CreateAgentResponse>, Status> {
        let request = request.into_inner();
        if request.payload_version != PAYLOAD_VERSION {
            return Err(Status::failed_precondition(format!(
                "payload version mismatch: expected {PAYLOAD_VERSION}, got {}",
                request.payload_version
            )));
        }

        {
            let registry = self.registry.lock().await;
            if let Some(existing_id) = registry.owners.get(&request.conversation_id).cloned() {
                if registry.agents.contains_key(&existing_id) {
                    return Ok(Response::new(pb::CreateAgentResponse {
                        agent_id: existing_id,
                        payload_version: PAYLOAD_VERSION.to_string(),
                    }));
                }
            }
        }

        let params = (self.build)(&self.config, &request.model)
            .map_err(|e| Status::internal(format!("{e:#}")))?;
        let agent_id = Uuid::new_v4().to_string();
        let agent_id_arc = Arc::<str>::from(agent_id.as_str());
        let fanout = Arc::new(Fanout::default());
        let turn_id = Arc::new(Mutex::new(None));
        let hook = IpcLifecycleHook {
            agent_id: agent_id_arc.clone(),
            fanout: fanout.clone(),
            turn_id: turn_id.clone(),
        };
        let agent = OrchestratorAgent::new(params, request.conversation_id.clone())
            .with_lifecycle_hook(hook)
            .await;

        let state = Arc::new(AgentState {
            agent_id: agent_id_arc.clone(),
            conversation_id: RwLock::new(request.conversation_id.clone()),
            agent: Mutex::new(agent),
            fanout,
            turn_id,
            cancel: Mutex::new(CancellationToken::new()),
        });

        let mut registry = self.registry.lock().await;
        registry.agents.insert(agent_id.clone(), state);
        registry
            .owners
            .insert(request.conversation_id, agent_id.clone());

        Ok(Response::new(pb::CreateAgentResponse {
            agent_id,
            payload_version: PAYLOAD_VERSION.to_string(),
        }))
    }

    async fn destroy_agent(
        &self,
        request: Request<pb::DestroyAgentRequest>,
    ) -> Result<Response<pb::DestroyAgentResponse>, Status> {
        let request = request.into_inner();
        let state = {
            let mut registry = self.registry.lock().await;
            let state = registry.agents.remove(&request.agent_id);
            if let Some(state) = &state {
                let conversation = state.conversation_id.read().await.clone();
                registry.owners.remove(&conversation);
            }
            state
        };
        let Some(state) = state else {
            return Ok(Response::new(pb::DestroyAgentResponse {}));
        };
        state
            .fanout
            .broadcast(convert::agent_destroyed(
                state.agent_id.to_string(),
                "agent destroyed",
            ))
            .await;
        state.fanout.close_all().await;
        // Abort any in-flight turn so its unary resolves instead of hanging.
        state.cancel.lock().await.cancel();
        Ok(Response::new(pb::DestroyAgentResponse {}))
    }

    async fn send_turn(
        &self,
        request: Request<pb::SendTurnRequest>,
    ) -> Result<Response<pb::SendTurnResponse>, Status> {
        let request = request.into_inner();
        let state = self
            .registry
            .lock()
            .await
            .agents
            .get(&request.agent_id)
            .cloned()
            .ok_or_else(|| Status::not_found("agent not found"))?;
        let message = convert::decode_message(&request.message_json)
            .map_err(|e| Status::invalid_argument(format!("{e:#}")))?;
        let mut agent = state
            .agent
            .try_lock()
            .map_err(|_| Status::failed_precondition("agent busy"))?;
        *state.turn_id.lock().await = Some(request.turn_id.clone());
        let cancel = CancellationToken::new();
        *state.cancel.lock().await = cancel.clone();

        let result =
            Agent::send_turn(&mut *agent, message, request.max_tokens, cancel.clone()).await;

        let cancelled = cancel.is_cancelled();
        if cancelled {
            state
                .fanout
                .broadcast(convert::turn_completed(
                    state.agent_id.to_string(),
                    request.turn_id,
                    "cancelled",
                ))
                .await;
            return Err(Status::cancelled("turn cancelled"));
        }

        match result {
            Ok(()) => {
                state
                    .fanout
                    .broadcast(convert::turn_completed(
                        state.agent_id.to_string(),
                        request.turn_id,
                        "",
                    ))
                    .await;
                Ok(Response::new(pb::SendTurnResponse {}))
            }
            Err(err) => {
                state
                    .fanout
                    .broadcast(convert::turn_completed(
                        state.agent_id.to_string(),
                        request.turn_id,
                        format!("{err:#}"),
                    ))
                    .await;
                Err(Status::internal(format!("{err:#}")))
            }
        }
    }

    async fn cancel_turn(
        &self,
        request: Request<pb::CancelTurnRequest>,
    ) -> Result<Response<pb::CancelTurnResponse>, Status> {
        let request = request.into_inner();
        let state = self
            .registry
            .lock()
            .await
            .agents
            .get(&request.agent_id)
            .cloned()
            .ok_or_else(|| Status::not_found("agent not found"))?;
        state.cancel.lock().await.cancel();
        Ok(Response::new(pb::CancelTurnResponse {}))
    }

    async fn load_conversation(
        &self,
        request: Request<pb::LoadConversationRequest>,
    ) -> Result<Response<pb::LoadConversationResponse>, Status> {
        let request = request.into_inner();
        let state = self
            .registry
            .lock()
            .await
            .agents
            .get(&request.agent_id)
            .cloned()
            .ok_or_else(|| Status::not_found("agent not found"))?;

        {
            let registry = self.registry.lock().await;
            if let Some(owner) = registry.owners.get(&request.conversation_id) {
                if owner != &request.agent_id {
                    return Err(Status::failed_precondition(
                        "conversation is owned by another agent",
                    ));
                }
            }
        }

        {
            let mut agent = state.agent.lock().await;
            Agent::load_conversation(&mut *agent, request.conversation_id.clone())
                .await
                .map_err(|e| Status::internal(format!("{e:#}")))?;
        }

        let old = {
            let mut conversation = state.conversation_id.write().await;
            std::mem::replace(&mut *conversation, request.conversation_id.clone())
        };

        let mut registry = self.registry.lock().await;
        registry.owners.remove(&old);
        registry
            .owners
            .insert(request.conversation_id, request.agent_id);
        Ok(Response::new(pb::LoadConversationResponse {}))
    }

    async fn get_conversation(
        &self,
        request: Request<pb::GetConversationRequest>,
    ) -> Result<Response<pb::GetConversationResponse>, Status> {
        let request = request.into_inner();
        let state = self
            .registry
            .lock()
            .await
            .agents
            .get(&request.agent_id)
            .cloned()
            .ok_or_else(|| Status::not_found("agent not found"))?;
        Ok(Response::new(pb::GetConversationResponse {
            conversation_id: state.conversation_id.read().await.clone(),
        }))
    }

    async fn subscribe(
        &self,
        request: Request<pb::SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        let request = request.into_inner();
        let state = self
            .registry
            .lock()
            .await
            .agents
            .get(&request.agent_id)
            .cloned()
            .ok_or_else(|| Status::not_found("agent not found"))?;

        let (sender, mut receiver) = mpsc::channel(1024);
        let agent = state.agent.lock().await;
        let mut terminate = state.fanout.attach(sender.clone()).await;
        let conversation_id = state.conversation_id.read().await.clone();
        let snapshot = agent.snapshot().await;
        let event = match snapshot {
            Ok(messages) => convert::switch_ok(
                state.agent_id.to_string(),
                conversation_id.clone(),
                &messages,
            )
            .map_err(|e| Status::internal(format!("{e:#}")))?,
            Err(err) => convert::switch_error(
                state.agent_id.to_string(),
                conversation_id,
                format!("{err:?}"),
            ),
        };
        sender
            .send(event)
            .await
            .map_err(|_| Status::internal("subscriber closed before snapshot"))?;
        drop(agent);

        let stream = async_stream::stream! {
            let mut terminate_closed = false;
            loop {
                tokio::select! {
                    biased;
                    term = terminate.recv(), if !terminate_closed => {
                        match term {
                            Some(status) => {
                                yield Err(status);
                                break;
                            }
                            None => terminate_closed = true,
                        }
                    }
                    item = receiver.recv() => {
                        match item {
                            Some(event) => yield Ok(event),
                            None => break,
                        }
                    }
                }
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let config = AgentConfig::from_env().context("loading agent config")?;
    let socket_path = yaca_transport::default_unix_socket_path();
    let server = AgentServer::new(config, |config, model| {
        let mut config = config.clone();
        if !model.is_empty() {
            config.model = model.to_string();
        }
        config.build_openrouter_params()
    });
    let service = AgentServiceServer::new(server);

    yaca_transport::serve_uds(service, &socket_path, shutdown_signal()).await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

#[cfg(test)]
mod tests;
