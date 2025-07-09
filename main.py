from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
import json
import sys
import os
from typing import AsyncGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), "rao"))
from rao.start import run_agents

app = FastAPI(
    title="AI Agent Orchestrator API",
    version="1.0.0",
    description="""
    AI Agent Orchestrator with Multi-Agent Streaming

    This API provides real-time streaming of multi-agent orchestration processes.
    The /chat/stream endpoint returns Server-Sent Events (SSE) for real-time updates.
    
    ## Streaming Events
    
    The streaming response includes these essential event types:
    - `agent_created`: Individual agent created with its specific query/task
    - `agent_response_chunk`: Real-time chunks of agent responses
    - `agent_complete`: Agent finishes processing
    - `agent_retry`: Agent failed and is retrying (with error info)
    - `agent_failed`: Agent failed after all retries (continues with fallback)
    - `final_agent_start`: Final synthesis agent initializes
    - `final_response_chunk`: Real-time chunks of final response
    - `complete`: Orchestration finished
    - `error`: Critical error occurred
    
    ## Usage
    
    For web applications, use EventSource API:
    ```javascript
    const eventSource = new EventSource('/chat/stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data);
    };
    ```
    """,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(
        description="The query or task for the AI agents to process",
        examples=[
            "Analyze the current state of AI market and provide investment recommendations"
        ],
    )
    model: str = Field(
        default="google:gemini-2.5-flash",
        description="The AI model to use for orchestration",
        examples=["google:gemini-2.5-flash"],
    )
    fallback_models: list[str] = Field(
        default=[
            "google:gemini-2.5-flash",
            "google:gemini-2.0-flash",
            "google:gemini-2.0-flash-lite",
        ],
        description="Fallback models to use if the primary model fails",
    )


class SSEEvent:
    def __init__(self, event_type: str, data: dict):
        self.event_type = event_type
        self.data = data

    def format(self) -> str:
        return f"event: {self.event_type}\ndata: {json.dumps(self.data)}\n\n"


async def stream_agent_response(
    query: str, model: str, fallback_models: list[str]
) -> AsyncGenerator[str, None]:
    models_to_try = [model] + fallback_models
    current_model_index = 0

    while current_model_index < len(models_to_try):
        current_model = models_to_try[current_model_index]

        try:
            yield SSEEvent(
                "model_info",
                {"current_model": current_model, "attempt": current_model_index + 1},
            ).format()

            events_queue = asyncio.Queue()

            async def callback_wrapper(event_data):
                await events_queue.put(event_data)

            orchestration_task = asyncio.create_task(
                run_agents(query, callback_wrapper)
            )

            orchestration_complete = False
            model_failed = False

            while True:
                try:
                    done, pending = await asyncio.wait(
                        [asyncio.create_task(events_queue.get()), orchestration_task],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=0.1,
                    )

                    if orchestration_task in done and not orchestration_complete:
                        try:
                            await orchestration_task
                            orchestration_complete = True
                        except Exception as e:
                            error_message = str(e).lower()
                            if any(
                                keyword in error_message
                                for keyword in [
                                    "overloaded",
                                    "rate limit",
                                    "quota",
                                    "capacity",
                                ]
                            ):
                                yield SSEEvent(
                                    "model_overloaded",
                                    {
                                        "model": current_model,
                                        "error": str(e),
                                        "trying_fallback": current_model_index
                                        < len(models_to_try) - 1,
                                    },
                                ).format()
                                model_failed = True
                                break
                            else:
                                yield SSEEvent(
                                    "error",
                                    {
                                        "message": f"Error with model {current_model}: {str(e)}"
                                    },
                                ).format()
                                return

                    for task in done:
                        if task != orchestration_task:
                            event_data = await task
                            if isinstance(event_data, dict):
                                event_type = event_data.get("event")
                                data = event_data.get("data", {})

                                if event_type in [
                                    "agent_created",
                                    "agent_response_chunk",
                                    "agent_complete",
                                    "agent_retry",
                                    "agent_failed",
                                    "final_agent_start",
                                    "final_response_chunk",
                                    "complete",
                                    "error",
                                ]:
                                    yield SSEEvent(event_type, data).format()

                    if orchestration_complete and events_queue.empty():
                        break

                    if model_failed:
                        break

                    for task in pending:
                        if task != orchestration_task:
                            task.cancel()

                except asyncio.TimeoutError:
                    if orchestration_complete and events_queue.empty():
                        break
                    continue
                except asyncio.CancelledError:
                    orchestration_task.cancel()
                    break

            if model_failed:
                current_model_index += 1
                if current_model_index < len(models_to_try):
                    yield SSEEvent(
                        "fallback_attempt",
                        {
                            "previous_model": current_model,
                            "next_model": models_to_try[current_model_index],
                            "attempt": current_model_index + 1,
                            "total_models": len(models_to_try),
                        },
                    ).format()
                    continue
                else:
                    yield SSEEvent(
                        "error",
                        {
                            "message": "All models failed or are overloaded. Please try again later."
                        },
                    ).format()
                    return

            # Process any remaining events
            while not events_queue.empty():
                event_data = await events_queue.get()
                if isinstance(event_data, dict):
                    event_type = event_data.get("event")
                    data = event_data.get("data", {})

                    if event_type in [
                        "agent_created",
                        "agent_response_chunk",
                        "agent_complete",
                        "agent_retry",
                        "agent_failed",
                        "final_agent_start",
                        "final_response_chunk",
                        "complete",
                        "error",
                    ]:
                        yield SSEEvent(event_type, data).format()

            if orchestration_complete:
                break

        except Exception as e:
            error_message = str(e).lower()
            if any(
                keyword in error_message
                for keyword in ["overloaded", "rate limit", "quota", "capacity"]
            ):
                yield SSEEvent(
                    "model_overloaded",
                    {
                        "model": current_model,
                        "error": str(e),
                        "trying_fallback": current_model_index < len(models_to_try) - 1,
                    },
                ).format()
                current_model_index += 1
                if current_model_index < len(models_to_try):
                    continue

            yield SSEEvent(
                "error",
                {
                    "message": f"Error during agent orchestration with model {current_model}: {str(e)}"
                },
            ).format()
            return


@app.post(
    "/chat/stream",
    summary="Stream AI Agent Orchestration",
    description="""
    Initiates a multi-agent orchestration process and streams real-time updates.
    
    **Response Format**: Server-Sent Events (SSE) with Content-Type: text/event-stream
    
    **Event Structure**:
    ```
    event: agent_created
    data: {"agent_id": 0, "type": "Research Agent", "usecase": "Market analysis", "query": "Analyze current market trends", "depends_on": []}
    
    event: agent_response_chunk
    data: {"agent_id": 0, "type": "Research Agent", "content": "Partial response...", "is_final": false}
    
    event: agent_retry
    data: {"agent_id": 1, "type": "Analysis Agent", "attempt": 1, "max_retries": 2, "error": "Connection timeout"}
    
    event: final_response_chunk
    data: {"content": "Final synthesized response chunk", "is_final": true}
    ```
    
    **Note**: This endpoint is best tested with curl or custom clients that support SSE.
    Swagger UI cannot display streaming responses properly.
    
    **Example cURL**:
    ```bash
    curl -X POST "http://localhost:8000/chat/stream" \\
         -H "Content-Type: application/json" \\
         -d '{"message": "Your query here"}' \\
         --no-buffer
    ```
    """,
    responses={
        200: {
            "description": "Streaming response with real-time agent updates",
            "content": {
                "text/event-stream": {
                    "example": """event: agent_created
data: {"agent_id": 0, "type": "Research Agent", "usecase": "Market analysis", "query": "Analyze current market trends", "depends_on": []}

event: agent_response_chunk
data: {"agent_id": 0, "type": "Research Agent", "content": "Partial response...", "is_final": false}

event: agent_retry
data: {"agent_id": 1, "type": "Analysis Agent", "attempt": 1, "max_retries": 2, "error": "Connection timeout"}

event: final_response_chunk
data: {"content": "Final synthesized response chunk", "is_final": true}
"""
                }
            },
        },
        400: {"description": "Invalid request - empty message"},
        500: {"description": "Internal server error during orchestration"},
    },
    tags=["AI Orchestration"],
)
async def chat_stream(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return StreamingResponse(
        stream_agent_response(request.message, request.model, request.fallback_models),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "AI Agent Orchestrator"}


@app.get("/", tags=["Info"])
async def root():
    return {
        "message": "AI Agent Orchestrator API",
        "endpoints": {
            "chat_stream": "/chat/stream",
            "health": "/health",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
        "streaming_info": {
            "format": "Server-Sent Events (SSE)",
            "content_type": "text/event-stream",
            "note": "Use EventSource API in browsers or curl with --no-buffer for testing",
        },
    }


@app.get("/stream-test", tags=["Testing"], include_in_schema=False)
async def stream_test():
    with open("test.html", "r") as file:
        html_content = file.read()

    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
