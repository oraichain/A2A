from typing import AsyncIterable, override
from common.types import (
    GetTaskRequest,
    GetTaskResponse,
    SendTaskRequest,
    TaskNotFoundError,
    TaskQueryParams,
    TaskSendParams,
    Message,
    TaskStatus,
    Artifact,
    TextPart,
    TaskState,
    SendTaskResponse,
    InternalError,
    JSONRPCResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    Task,
    TaskIdParams,
    InvalidParamsError,
)
from common.server.task_manager import InMemoryTaskManager
from agents.autogen.agent import Agent
import common.server.utils as utils
from typing import Union
import asyncio
import logging
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class AgentTaskManager(InMemoryTaskManager):
    def __init__(self, agent: Agent):
        logger.debug("Initializing AgentTaskManager")
        super().__init__()
        self.agent = agent
        logger.debug(f"AgentTaskManager initialized with agent: {agent}")
        
    @override
    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        logger.debug(f"on_get_task called with request {request.id}")
        task_query_params: TaskQueryParams = request.params

        async with self.lock:
            task = self.tasks.get(task_query_params.id)
            if task is None:
                return GetTaskResponse(id=request.id, error=TaskNotFoundError())
        return GetTaskResponse(id=request.id, result=task)

    async def _run_streaming_agent(self, request: SendTaskStreamingRequest):
        logger.debug(f"Starting _run_streaming_agent with request: {request.id}")
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        logger.debug(f"User query extracted: '{query[:50]}...' for task {task_send_params.id}")

        try:
            logger.debug(f"Streaming agent response for task {task_send_params.id}")
            async for item in self.agent.stream(query, task_send_params.sessionId):
                logger.debug(f"Received agent stream item for task {task_send_params.id}")
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]
                artifact = None
                message = None
                parts = [
                    {
                        "type": "text",
                        "text": item["content"],
                        "images": item["images"],
                        "model_usage": item.get("model_usage", None),
                    }
                ]
                end_stream = False
                
                logger.debug(f"Stream item state - is_complete: {is_task_complete}, require_input: {require_user_input}")

                if is_task_complete:
                    logger.debug(f"Task {task_send_params.id} is complete")
                    task_state = TaskState.COMPLETED
                    artifact = Artifact(parts=parts, index=0, append=False)
                    end_stream = True
                elif not is_task_complete and not require_user_input:
                    logger.debug(f"Task {task_send_params.id} is still working")
                    task_state = TaskState.WORKING
                    message = Message(role="agent", parts=parts)
                elif require_user_input:
                    logger.debug(f"Task {task_send_params.id} requires user input")
                    task_state = TaskState.INPUT_REQUIRED
                    message = Message(role="agent", parts=parts)
                    end_stream = True

                logger.debug(f"Setting task status to {task_state} for {task_send_params.id}")
                task_status = TaskStatus(state=task_state, message=message)
                
                logger.debug(f"Updating store for task {task_send_params.id}")
                latest_task = await self.update_store(
                    task_send_params.id,
                    task_status,
                    None if artifact is None else [artifact],
                )
                
                logger.debug(f"Sending task notification for {task_send_params.id}")
                await self.send_task_notification(latest_task)

                if artifact:
                    logger.debug(f"Creating artifact update event for task {task_send_params.id}")
                    task_artifact_update_event = TaskArtifactUpdateEvent(
                        id=task_send_params.id, artifact=artifact
                    )
                    logger.debug(f"Enqueuing artifact update event for {task_send_params.id}")
                    await self.enqueue_events_for_sse(
                        task_send_params.id, task_artifact_update_event
                    )

                logger.debug(f"Creating status update event for task {task_send_params.id}, final={end_stream}")
                task_update_event = TaskStatusUpdateEvent(
                    id=task_send_params.id, status=task_status, final=end_stream
                )
                logger.debug(f"Enqueuing status update event for {task_send_params.id}")
                await self.enqueue_events_for_sse(
                    task_send_params.id, task_update_event
                )
                logger.debug(f"Finished processing stream item for {task_send_params.id}")

        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}")
            logger.debug(f"Exception details for task {task_send_params.id}: {traceback.format_exc()}")
            await self.enqueue_events_for_sse(
                task_send_params.id,
                InternalError(
                    message=f"An error occurred while streaming the response: {e}"
                ),
            )
        logger.debug(f"Exiting _run_streaming_agent for task {task_send_params.id}")

    def _validate_request(
        self, request: Union[SendTaskRequest, SendTaskStreamingRequest]
    ) -> JSONRPCResponse | None:
        logger.debug(f"Validating request {request.id}")
        task_send_params: TaskSendParams = request.params
        
        logger.debug(f"Checking modality compatibility: requested={task_send_params.acceptedOutputModes}, supported={self.agent.SUPPORTED_CONTENT_TYPES}")
        if not utils.are_modalities_compatible(
            task_send_params.acceptedOutputModes,
            self.agent.SUPPORTED_CONTENT_TYPES,
        ):
            logger.warning(
                "Unsupported output mode. Received %s, Support %s",
                task_send_params.acceptedOutputModes,
                self.agent.SUPPORTED_CONTENT_TYPES,
            )
            logger.debug(f"Request {request.id} rejected due to incompatible types")
            return utils.new_incompatible_types_error(request.id)

        logger.debug(f"Checking push notification configuration for request {request.id}")
        if (
            task_send_params.pushNotification
            and not task_send_params.pushNotification.url
        ):
            logger.warning("Push notification URL is missing")
            logger.debug(f"Request {request.id} rejected due to missing push notification URL")
            return JSONRPCResponse(
                id=request.id,
                error=InvalidParamsError(message="Push notification URL is missing"),
            )

        logger.debug(f"Request {request.id} validation successful")
        return None

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        logger.debug(f"on_send_task called with request {request.id}")
        raise NotImplementedError("Not implemented")

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        logger.debug(f"on_send_task_subscribe started for request {request.id}")
        try:
            logger.debug(f"Validating request {request.id}")
            error = self._validate_request(request)
            if error:
                logger.debug(f"Request {request.id} validation failed: {error}")
                return error

            logger.debug(f"Upserting task for request {request.id}")
            await self.upsert_task(request.params)

            task_send_params: TaskSendParams = request.params
            logger.debug(f"Setting up SSE consumer for task {task_send_params.id}")
            sse_event_queue = await self.setup_sse_consumer(task_send_params.id, False)

            logger.debug(f"Creating agent streaming task for {task_send_params.id}")
            asyncio.create_task(self._run_streaming_agent(request))

            logger.debug(f"Returning SSE event queue for task {task_send_params.id}")
            return self.dequeue_events_for_sse(
                request.id, task_send_params.id, sse_event_queue
            )
        except Exception as e:
            logger.error(f"Error in SSE stream: {e}")
            logger.debug(f"Exception details for request {request.id}: {traceback.format_exc()}")
            print(traceback.format_exc())
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message="An error occurred while streaming the response"
                ),
            )

    # TODO: this function is used for on_send_task. We have not implemented the on_send_task yet, so it is currently not used.
    async def _process_agent_response(
        self, request: SendTaskRequest, agent_response: dict
    ) -> SendTaskResponse:
        """Processes the agent's response and updates the task store."""
        logger.debug(f"Processing agent response for request {request.id}")
        task_send_params: TaskSendParams = request.params
        task_id = task_send_params.id
        history_length = task_send_params.historyLength
        task_status = None

        logger.debug(f"Creating response parts for task {task_id}")
        parts = [{"type": "text", "text": agent_response["content"]}]
        artifact = None
        
        logger.debug(f"Checking if user input is required for task {task_id}: {agent_response['require_user_input']}")
        if agent_response["require_user_input"]:
            logger.debug(f"Setting task {task_id} status to INPUT_REQUIRED")
            task_status = TaskStatus(
                state=TaskState.INPUT_REQUIRED,
                message=Message(role="agent", parts=parts),
            )
        else:
            logger.debug(f"Setting task {task_id} status to COMPLETED")
            task_status = TaskStatus(state=TaskState.COMPLETED)
            artifact = Artifact(parts=parts)
            
        logger.debug(f"Updating store for task {task_id}")
        task = await self.update_store(
            task_id, task_status, None if artifact is None else [artifact]
        )
        
        logger.debug(f"Appending task history for task {task_id} with length {history_length}")
        task_result = self.append_task_history(task, history_length)
        
        logger.debug(f"Sending task notification for task {task_id}")
        await self.send_task_notification(task)
        
        logger.debug(f"Returning response for request {request.id}")
        return SendTaskResponse(id=request.id, result=task_result)

    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        logger.debug(f"Getting user query for task {task_send_params.id}")
        part = task_send_params.message.parts[0]
        if not isinstance(part, TextPart):
            logger.debug(f"Invalid part type for task {task_send_params.id}: {type(part)}")
            raise ValueError("Only text parts are supported")
        logger.debug(f"User query extracted for task {task_send_params.id}: '{part.text[:50]}...'")
        return part.text

    async def send_task_notification(self, task: Task):
        return
        logger.debug(f"Checking push notification for task {task.id}")
        if not await self.has_push_notification_info(task.id):
            logger.info(f"No push notification info found for task {task.id}")
            return
            
        logger.debug(f"Getting push notification info for task {task.id}")
        push_info = await self.get_push_notification_info(task.id)

        logger.info(f"Notifying for task {task.id} => {task.status.state}")
        logger.debug(f"Sending push notification to {push_info.url} for task {task.id}")
        await self.notification_sender_auth.send_push_notification(
            push_info.url, data=task.model_dump(exclude_none=True)
        )
        logger.debug(f"Push notification sent for task {task.id}")

    async def on_resubscribe_to_task(
        self, request
    ) -> AsyncIterable[SendTaskStreamingResponse] | JSONRPCResponse:
        logger.debug(f"on_resubscribe_to_task called with request {request.id}")
        task_id_params: TaskIdParams = request.params
        try:
            logger.debug(f"Setting up SSE consumer for task {task_id_params.id} with resubscribe=True")
            sse_event_queue = await self.setup_sse_consumer(task_id_params.id, True)
            
            logger.debug(f"Returning SSE event queue for task {task_id_params.id}")
            return self.dequeue_events_for_sse(
                request.id, task_id_params.id, sse_event_queue
            )
        except Exception as e:
            logger.error(f"Error while reconnecting to SSE stream: {e}")
            logger.debug(f"Exception details for task resubscription {task_id_params.id}: {traceback.format_exc()}")
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message=f"An error occurred while reconnecting to stream: {e}"
                ),
            )
