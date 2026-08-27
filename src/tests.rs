//! End-to-end tests for `yaca-agent` over the in-process duplex loopback,
//! using a mock `CompletionModel` (no network).

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::StreamExt;

use yaca_core::agent::config::AgentConfig;
use yaca_core::agent::orchestrator::OrchestratorParams;
use yaca_core::tools::Environment;
use yaca_transport::convert::{self, Event};
use yaca_transport::pb::{self, agent_service_server::AgentServiceServer};
use yaca_transport::{Connection, Message, MessageUpdate};

use super::{AgentServer, Fanout};

type MockParams = OrchestratorParams<MockModel, MockClient, rig::memory::InMemoryConversationMemory>;

#[derive(Clone)]
struct MockModel {
    block: bool,
    chunks: usize,
    started: Arc<tokio::sync::Notify>,
}

impl rig::completion::CompletionModel for MockModel {
    async fn completion(
        &self,
        _request: rig::completion::CompletionRequest,
    ) -> Result<rig::completion::CompletionResponse, rig::completion::CompletionError> {
        Err(rig::completion::CompletionError::ResponseError(
            "unary not supported".into(),
        ))
    }

    async fn stream(
        &self,
        _request: rig::completion::CompletionRequest,
    ) -> Result<rig::streaming::StreamingCompletionResponse, rig::completion::CompletionError> {
        self.started.notify_one();
        let mut items: Vec<Result<rig::streaming::RawStreamingChoice, rig::completion::CompletionError>> =
            (0..self.chunks)
                .map(|_| Ok(rig::streaming::RawStreamingChoice::Message("x".into())))
                .collect();
        if !self.block {
            items.push(Ok(rig::streaming::RawStreamingChoice::FinalResponse(
                rig::streaming::StreamFinal::new("mock", rig::completion::Usage::new()),
            )));
        }
        let stream: rig::streaming::StreamingResult = if self.block {
            Box::pin(tokio_stream::iter(items).chain(tokio_stream::pending()))
        } else {
            Box::pin(tokio_stream::iter(items))
        };
        Ok(rig::streaming::StreamingCompletionResponse::stream(
            "mock", stream,
        ))
    }
}

#[derive(Clone)]
struct MockClient {
    block: bool,
    chunks: usize,
    started: Arc<tokio::sync::Notify>,
}

impl rig::client::CompletionClient for MockClient {
    type CompletionModel = MockModel;

    fn completion_model(&self, _model: impl Into<String>) -> Self::CompletionModel {
        MockModel {
            block: self.block,
            chunks: self.chunks,
            started: self.started.clone(),
        }
    }
}

async fn spawn_server_with(block: bool, chunks: usize) -> (Connection, Arc<tokio::sync::Notify>) {
    let started = Arc::new(tokio::sync::Notify::new());
    let (server_io, client_io) = yaca_transport::duplex::pair();
    let started_clone = started.clone();

    let server: AgentServer<MockParams> = AgentServer::new(AgentConfig::default(), move |_c, _m| {
        Ok(OrchestratorParams::new(
            Environment::default(),
            MockClient {
                block,
                chunks,
                started: started_clone.clone(),
            },
            "mock-model",
            rig::memory::InMemoryConversationMemory::new(),
        ))
    });
    let svc = AgentServiceServer::new(server);
    tokio::spawn(async move {
        yaca_transport::duplex::serve(svc, server_io).await.unwrap();
    });

    let channel = yaca_transport::duplex::connect(client_io).await.unwrap();
    (Connection::new(channel), started)
}

async fn spawn_server(block: bool) -> (Connection, Arc<tokio::sync::Notify>) {
    spawn_server_with(block, 1).await
}

/// Consume and discard the synthesized snapshot; returns the live event stream.
async fn subscribe_and_drain_snapshot(conn: &Connection, agent_id: &str) -> EventStream {
    let mut events = conn.subscribe(agent_id).await.unwrap();
    let first = events.next_event().await.unwrap().unwrap();
    assert!(
        matches!(first, Event::SwitchConversation { .. }),
        "first event must be the synthesized snapshot, got {first:?}"
    );
    events
}

type EventStream = yaca_transport::EventStream;

#[tokio::test]
async fn payload_version_mismatch_is_rejected() {
    let (server_io, client_io) = yaca_transport::duplex::pair();
    let server: AgentServer<MockParams> = AgentServer::new(AgentConfig::default(), |_c, _m| {
        Ok(OrchestratorParams::new(
            Environment::default(),
            MockClient {
                block: false,
                chunks: 1,
                started: Arc::new(tokio::sync::Notify::new()),
            },
            "mock-model",
            rig::memory::InMemoryConversationMemory::new(),
        ))
    });
    let svc = AgentServiceServer::new(server);
    tokio::spawn(async move {
        yaca_transport::duplex::serve(svc, server_io).await.unwrap();
    });
    let channel = yaca_transport::duplex::connect(client_io).await.unwrap();

    let mut client = pb::agent_service_client::AgentServiceClient::new(channel);
    let err = client
        .create_agent(pb::CreateAgentRequest {
            conversation_id: "c".to_string(),
            model: String::new(),
            payload_version: "wrong/version".to_string(),
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::FailedPrecondition);
}

#[tokio::test]
async fn create_agent_is_idempotent_per_conversation() {
    let (conn, _started) = spawn_server(false).await;

    let a1 = conn.create_agent("shared", None::<String>).await.unwrap();
    let a2 = conn.create_agent("shared", None::<String>).await.unwrap();
    assert_eq!(a1.id(), a2.id(), "same conversation must attach to one agent");

    let a3 = conn.create_agent("other", None::<String>).await.unwrap();
    assert_ne!(a1.id(), a3.id(), "different conversations get distinct agents");
}

#[tokio::test]
async fn full_turn_emits_snapshot_new_update_completed() {
    let (conn, _started) = spawn_server(false).await;
    let agent = conn.create_agent("main", None::<String>).await.unwrap();
    let mut events = subscribe_and_drain_snapshot(&conn, agent.id()).await;

    agent
        .send_turn(Message::user("hi"), 1024, "turn-1")
        .await
        .unwrap();

    let mut new_seen = false;
    let mut update_seen = false;
    loop {
        let event = events.next_event().await.unwrap().unwrap();
        match event {
            Event::NewMessage { index, .. } => {
                assert_eq!(index, 0, "fresh conversation opens at index 0");
                new_seen = true;
            }
            Event::UpdateMessage { .. } => update_seen = true,
            Event::TurnCompleted { turn_id, error } => {
                assert_eq!(turn_id, "turn-1");
                assert!(error.is_empty(), "successful turn has empty error: {error}");
                break;
            }
            other => panic!("unexpected event {other:?}"),
        }
    }
    assert!(new_seen, "expected a NewMessage event");
    assert!(update_seen, "expected an UpdateMessage event");
}

#[tokio::test]
async fn cancel_turn_mid_turn_yields_cancelled() {
    let (conn, started) = spawn_server(true).await;
    let agent = Arc::new(conn.create_agent("main", None::<String>).await.unwrap());
    let mut events = subscribe_and_drain_snapshot(&conn, agent.id()).await;

    let agent_clone = agent.clone();
    let turn = tokio::spawn(async move {
        agent_clone
            .send_turn(Message::user("hi"), 1024, "turn-1")
            .await
    });

    // Wait until the model's stream is open, i.e. the turn is in flight.
    started.notified().await;
    agent.cancel_turn().await.unwrap();

    let err = turn.await.unwrap().expect_err("cancelled turn must fail the unary");
    assert!(format!("{err:#}").contains("cancelled"), "{err:#}");

    // The event stream must end the turn with TurnCompleted{error: "cancelled"}.
    loop {
        let event = events.next_event().await.unwrap().unwrap();
        match event {
            Event::TurnCompleted { turn_id, error } => {
                assert_eq!(turn_id, "turn-1");
                assert_eq!(error, "cancelled");
                break;
            }
            Event::NewMessage { .. } | Event::UpdateMessage { .. } => continue,
            other => panic!("unexpected event {other:?}"),
        }
    }
}

#[tokio::test]
async fn concurrent_send_turn_returns_agent_busy() {
    let (conn, started) = spawn_server(true).await;
    let agent = Arc::new(conn.create_agent("main", None::<String>).await.unwrap());
    let mut events = subscribe_and_drain_snapshot(&conn, agent.id()).await;

    let agent_clone = agent.clone();
    let first = tokio::spawn(async move {
        agent_clone
            .send_turn(Message::user("hi"), 1024, "turn-1")
            .await
    });

    started.notified().await;

    let second = agent
        .send_turn(Message::user("hi"), 1024, "turn-2")
        .await;
    let err = second.expect_err("second concurrent turn must be rejected");
    assert!(format!("{err:#}").contains("agent busy"), "{err:#}");

    // Tear down the in-flight turn and drain to the terminal event.
    agent.cancel_turn().await.unwrap();
    let _ = first.await.unwrap();
    while let Some(event) = events.next_event().await {
        if let Event::TurnCompleted { .. } = event.unwrap() {
            break;
        }
    }
}

#[tokio::test]
async fn destroy_agent_mid_turn_aborts_and_closes_stream() {
    let (conn, started) = spawn_server(true).await;
    let agent = Arc::new(conn.create_agent("main", None::<String>).await.unwrap());
    let mut events = subscribe_and_drain_snapshot(&conn, agent.id()).await;

    let agent_clone = agent.clone();
    let turn = tokio::spawn(async move {
        agent_clone
            .send_turn(Message::user("hi"), 1024, "turn-1")
            .await
    });

    // Wait until the turn is in flight, then destroy the agent.
    started.notified().await;
    agent.destroy().await.unwrap();

    // The in-flight turn's unary must resolve (aborted), not hang.
    let result = turn.await.unwrap();
    assert!(result.is_err(), "destroyed turn must fail its unary");

    // The subscriber must observe AgentDestroyed and then a clean stream end.
    let mut saw_destroyed = false;
    while let Some(event) = events.next_event().await {
        match event {
            Ok(Event::AgentDestroyed { .. }) => saw_destroyed = true,
            Ok(_) => {}
            Err(err) => panic!("unexpected stream error after destroy: {err:#}"),
        }
    }
    assert!(saw_destroyed, "subscriber must observe AgentDestroyed");
}

#[tokio::test]
async fn fanout_drops_updates_and_terminates_on_non_droppable_overflow() {
    let fanout = Fanout::default();
    let (sender, _receiver) = mpsc::channel(2);
    let mut terminate = fanout.attach(sender).await;

    let update = |n: usize| {
        convert::update_message(
            "a",
            n,
            &MessageUpdate::AssistantReasoningAppend(format!("x{n}")),
            "t",
        )
        .unwrap()
    };

    // Fill the subscriber queue with droppable updates.
    for n in 0..2 {
        fanout.broadcast(update(n)).await;
    }
    // A droppable overflow is silently dropped; the subscription survives.
    fanout.broadcast(update(2)).await;
    assert!(
        terminate.try_recv().is_err(),
        "droppable overflow must not terminate the subscription"
    );

    // A non-droppable overflow terminates the subscription with RESOURCE_EXHAUSTED.
    let new_message = convert::new_message("a", 0, &Message::user("hi"), "t").unwrap();
    fanout.broadcast(new_message).await;
    let status = terminate
        .recv()
        .await
        .expect("non-droppable overflow must terminate the subscription");
    assert_eq!(status.code(), tonic::Code::ResourceExhausted);
}

#[tokio::test]
async fn subscriber_disconnect_mid_turn_does_not_fail_turn() {
    let (conn, _started) = spawn_server(false).await;
    let agent = conn.create_agent("main", None::<String>).await.unwrap();
    // Subscribe, drain the snapshot, then drop the stream: the subscriber
    // disconnects before the turn starts.
    {
        let mut events = conn.subscribe(agent.id()).await.unwrap();
        let _ = events.next_event().await.unwrap().unwrap();
    }

    // The turn must still complete successfully with no live subscriber.
    agent
        .send_turn(Message::user("hi"), 1024, "turn-1")
        .await
        .unwrap();
}
