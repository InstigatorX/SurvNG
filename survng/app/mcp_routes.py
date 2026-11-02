"""Stateless MCP endpoint for Grok Bot and other token-authenticated clients.

The handler speaks the Streamable HTTP subset used by a single JSON-RPC POST:
initialize, ping, tools/list, and tools/call. It does not keep a session or
open a server-sent stream. Browser session cookies are rejected. The route
answers 401 unless API authentication is enabled and the bearer token matches.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from survng import __version__ as SURVNG_VERSION

from .assistant import (
    AssistantPageContext,
    AssistantToolCall,
    _is_safe_app_href,
    sanitize_assistant_data,
)
from .config import ApiScope
from .incident_utils import event_snapshot_path
from .intelligence_routes import IntelligenceService
from .manager import AppManager
from .security import ApiPrincipal, authenticate_api_token

LOGGER = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2025-03-26"
_MAX_BODY_BYTES = 256 * 1024
_MAX_IMAGE_EDGE = 1280
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603

_INSTRUCTIONS = (
    "SurvNG investigation tools. Read tools report health, incidents, and one "
    "snapshot. Camera control and clip export appear only when the token "
    "allows them. Visual similarity is not confirmed identity."
)

_READ: ApiScope = "read"
_CONTROL: ApiScope = "camera:control"
_ADMIN: ApiScope = "admin"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _CameraIdArgs(_StrictModel):
    camera_id: str = Field(min_length=1, max_length=128)


class _CameraFeatureArgs(_StrictModel):
    camera_id: str = Field(min_length=1, max_length=128)
    enabled: bool


class _IncidentImageArgs(_StrictModel):
    event_id: int = Field(gt=0)


class _RecentIncidentsArgs(_StrictModel):
    limit: int = Field(default=20, ge=1, le=50)


class McpProtocolError(Exception):
    def __init__(self, code: int, message: str, *, status_code: int = 200) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class McpDependencies:
    get_config: Callable[[], Any]
    get_manager: Callable[[], AppManager]
    manager_lock: threading.RLock
    intelligence: IntelligenceService
    application_stopping: threading.Event


@dataclass(frozen=True, slots=True)
class _ToolOutcome:
    content: list[dict[str, Any]]
    is_error: bool = False


def _object_schema(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required or [],
    }


def _string(max_length: int) -> dict[str, Any]:
    return {"type": "string", "maxLength": max_length}


_TIME_ZONE = _string(128)
_CAMERA_ID = _string(128)
_WINDOW = {
    "camera_id": _CAMERA_ID,
    "start_at": _string(64),
    "end_at": _string(64),
    "time_zone": _TIME_ZONE,
}
_SEARCH_PROPERTIES = {
    **_WINDOW,
    "event_type": {"type": "string", "enum": ["all", "object", "motion"]},
    "object_label": _string(128),
    "zone": _string(128),
    "minimum_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "face_name": _string(128),
    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
}


def _text_result(payload: Any, *, is_error: bool = False) -> _ToolOutcome:
    return _ToolOutcome(
        [{"type": "text", "text": json.dumps(payload, separators=(",", ":"))}],
        is_error=is_error,
    )


def _bounded_jpeg(path: Any) -> bytes:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError("event snapshot is unavailable")
    height, width = frame.shape[:2]
    long_edge = max(height, width)
    if long_edge <= 0:
        raise FileNotFoundError("event snapshot is unavailable")
    if long_edge > _MAX_IMAGE_EDGE:
        scale = _MAX_IMAGE_EDGE / long_edge
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )
    if not ok:
        raise FileNotFoundError("event snapshot is unavailable")
    return encoded.tobytes()


def _validation_message(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid arguments"
    location = ".".join(str(part) for part in errors[0].get("loc", ()))
    return f"invalid arguments: {location}" if location else "invalid arguments"


@dataclass(frozen=True, slots=True)
class _Tool:
    name: str
    description: str
    scope: ApiScope
    input_schema: dict[str, Any]
    kind: str


def _assistant_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
    *,
    scope: ApiScope = _READ,
) -> _Tool:
    return _Tool(name, description, scope, _object_schema(properties, required), "assistant")


_TOOLS: tuple[_Tool, ...] = (
    _assistant_tool(
        "get_system_health",
        "Overall recorder, detector, storage, and camera status.",
        {},
    ),
    _assistant_tool(
        "get_camera_health",
        "Runtime status for one camera, or every camera when camera_id is omitted.",
        {"camera_id": _CAMERA_ID},
    ),
    _assistant_tool(
        "explain_configuration",
        "Credential-free summary of the active SurvNG configuration.",
        {},
    ),
    _assistant_tool(
        "inspect_incident",
        "Details for the incident that contains event_id.",
        {"event_id": {"type": "integer", "minimum": 1}},
        ["event_id"],
    ),
    _assistant_tool(
        "search_incidents",
        "Structured incident search. Use ISO-8601 start_at and end_at with offsets.",
        _SEARCH_PROPERTIES,
    ),
    _assistant_tool(
        "semantic_search_recordings",
        "Search indexed incident images by a short visual description.",
        {
            "query": _string(500),
            "camera_id": _CAMERA_ID,
            "object_label": _string(128),
            "start_at": _string(64),
            "end_at": _string(64),
            "time_zone": _TIME_ZONE,
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        ["query"],
    ),
    _assistant_tool(
        "summarize_recent_activity",
        "Counts of object incidents in a time window.",
        {
            **_WINDOW,
            "object_label": _string(128),
            "zone": _string(128),
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
    ),
    _assistant_tool(
        "trace_across_cameras",
        "Bounded cross-camera timeline for an incident, face name, or object label.",
        {
            "event_id": {"type": "integer", "minimum": 1},
            "face_name": _string(128),
            "object_label": _string(128),
            "start_at": _string(64),
            "end_at": _string(64),
            "time_zone": _TIME_ZONE,
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
    ),
    _Tool(
        "get_incident_image",
        "One JPEG of a stored incident snapshot, with the long edge capped at 1280 pixels.",
        _READ,
        _object_schema(
            {"event_id": {"type": "integer", "minimum": 1}},
            ["event_id"],
        ),
        "image",
    ),
    _Tool(
        "recent_incidents",
        "Latest incident lifecycle records from the notification journal.",
        _READ,
        _object_schema(
            {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
        ),
        "recent",
    ),
    _Tool(
        "start_camera",
        "Start one camera. Requires the camera:control scope.",
        _CONTROL,
        _object_schema({"camera_id": _CAMERA_ID}, ["camera_id"]),
        "start_camera",
    ),
    _Tool(
        "stop_camera",
        "Stop one camera. Requires the camera:control scope.",
        _CONTROL,
        _object_schema({"camera_id": _CAMERA_ID}, ["camera_id"]),
        "stop_camera",
    ),
    _Tool(
        "set_recording",
        "Enable or disable recording for one camera. Requires the camera:control scope.",
        _CONTROL,
        _object_schema(
            {"camera_id": _CAMERA_ID, "enabled": {"type": "boolean"}},
            ["camera_id", "enabled"],
        ),
        "set_recording",
    ),
    _Tool(
        "set_detection",
        "Enable or disable object detection for one camera. Requires the camera:control scope.",
        _CONTROL,
        _object_schema(
            {"camera_id": _CAMERA_ID, "enabled": {"type": "boolean"}},
            ["camera_id", "enabled"],
        ),
        "set_detection",
    ),
    _assistant_tool(
        "create_media_export",
        "Queue one recording clip or timelapse. Requires the admin scope.",
        {
            "camera_id": _CAMERA_ID,
            "start_at": _string(64),
            "end_at": _string(64),
            "time_zone": _TIME_ZONE,
            "export_kind": {"type": "string", "enum": ["recording", "timelapse"]},
            "source": {"type": "string", "enum": ["main", "live"]},
            "sample_interval_seconds": {"type": "number", "minimum": 1, "maximum": 3600},
            "output_fps": {"type": "integer", "minimum": 1, "maximum": 60},
            "width": {"type": "integer", "minimum": 320, "maximum": 3840},
            "height": {"type": "integer", "minimum": 240, "maximum": 2160},
        },
        ["camera_id", "start_at", "end_at", "export_kind"],
        scope=_ADMIN,
    ),
)
_TOOLS_BY_NAME = {tool.name: tool for tool in _TOOLS}


class McpEndpoint:
    def __init__(self, deps: McpDependencies) -> None:
        self.deps = deps

    async def handle(self, request: Request) -> Response:
        principal = self._principal(request)
        if principal is None:
            return JSONResponse(
                {"detail": "valid bearer token required"},
                status_code=401,
                headers={
                    "Cache-Control": "no-store",
                    "WWW-Authenticate": 'Bearer realm="SurvNG"',
                },
            )
        try:
            message, rpc_id, notification = self._parse(await request.body())
        except McpProtocolError as exc:
            return self._error_response(None, exc)
        if notification:
            return Response(status_code=202)
        try:
            result = self._dispatch(message, principal)
        except McpProtocolError as exc:
            return self._error_response(rpc_id, exc)
        return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": result})

    def _principal(self, request: Request) -> ApiPrincipal | None:
        config = self.deps.get_config()
        if not config.api_auth.enabled:
            return None
        principal = authenticate_api_token(
            request.headers.get("authorization", ""),
            config.api_auth,
        )
        if principal is None or principal.kind != "token":
            return None
        return principal

    def _parse(self, raw: bytes) -> tuple[dict[str, Any], Any, bool]:
        if len(raw) > _MAX_BODY_BYTES:
            raise McpProtocolError(_INVALID_REQUEST, "request is too large", status_code=400)
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise McpProtocolError(_PARSE_ERROR, "parse error", status_code=400) from exc
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise McpProtocolError(_INVALID_REQUEST, "invalid request", status_code=400)
        method = message.get("method")
        if not isinstance(method, str) or not method:
            raise McpProtocolError(_INVALID_REQUEST, "invalid request", status_code=400)
        if "id" not in message:
            return message, None, True
        rpc_id = message.get("id")
        if isinstance(rpc_id, bool) or (
            rpc_id is not None and not isinstance(rpc_id, (str, int))
        ):
            raise McpProtocolError(_INVALID_REQUEST, "invalid request", status_code=400)
        return message, rpc_id, False

    def _dispatch(self, message: dict[str, Any], principal: ApiPrincipal) -> Any:
        method = message["method"]
        params = message.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise McpProtocolError(_INVALID_PARAMS, "invalid params")
        if method == "initialize":
            return {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "survng", "version": SURVNG_VERSION},
                "instructions": _INSTRUCTIONS,
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [self._tool_descriptor(tool) for tool in self._visible(principal)]}
        if method == "tools/call":
            return self._call_tool(params, principal)
        raise McpProtocolError(_METHOD_NOT_FOUND, "method not found")

    def _visible(self, principal: ApiPrincipal) -> list[_Tool]:
        return [tool for tool in _TOOLS if principal.permits(tool.scope)]

    @staticmethod
    def _tool_descriptor(tool: _Tool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
        }

    def _call_tool(self, params: dict[str, Any], principal: ApiPrincipal) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise McpProtocolError(_INVALID_PARAMS, "invalid params")
        tool = _TOOLS_BY_NAME.get(name)
        if tool is None or not principal.permits(tool.scope):
            raise McpProtocolError(_INVALID_PARAMS, "unknown tool")
        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise McpProtocolError(_INVALID_PARAMS, "invalid arguments")
        self._check_arguments(tool, arguments)
        try:
            outcome = self._invoke(tool, arguments)
        except McpProtocolError:
            raise
        except ValidationError as exc:
            raise McpProtocolError(_INVALID_PARAMS, _validation_message(exc)) from exc
        except Exception:
            LOGGER.exception("MCP tool %s failed", name)
            outcome = _text_result("SurvNG could not complete the tool call", is_error=True)
        return {
            "content": outcome.content,
            "isError": outcome.is_error,
        }

    @staticmethod
    def _check_arguments(tool: _Tool, arguments: dict[str, Any]) -> None:
        properties = tool.input_schema.get("properties") or {}
        required = tool.input_schema.get("required") or []
        if any(key not in properties for key in arguments):
            raise McpProtocolError(_INVALID_PARAMS, "invalid arguments")
        if any(key not in arguments for key in required):
            raise McpProtocolError(_INVALID_PARAMS, "invalid arguments")

    def _invoke(self, tool: _Tool, arguments: dict[str, Any]) -> _ToolOutcome:
        if tool.kind == "assistant":
            return self._invoke_assistant(tool.name, arguments)
        if tool.kind == "image":
            args = _IncidentImageArgs.model_validate(arguments)
            return self._incident_image(args.event_id)
        if tool.kind == "recent":
            args = _RecentIncidentsArgs.model_validate(arguments)
            return self._recent_incidents(args.limit)
        if tool.kind == "start_camera":
            args = _CameraIdArgs.model_validate(arguments)
            return self._camera_command("start_camera", args.camera_id)
        if tool.kind == "stop_camera":
            args = _CameraIdArgs.model_validate(arguments)
            return self._camera_command("stop_camera", args.camera_id)
        if tool.kind == "set_recording":
            args = _CameraFeatureArgs.model_validate(arguments)
            return self._camera_command("set_recording", args.camera_id, args.enabled)
        if tool.kind == "set_detection":
            args = _CameraFeatureArgs.model_validate(arguments)
            return self._camera_command("set_detection", args.camera_id, args.enabled)
        raise McpProtocolError(_INVALID_PARAMS, "unknown tool")

    def _invoke_assistant(self, name: str, arguments: dict[str, Any]) -> _ToolOutcome:
        payload = dict(arguments)
        time_zone = str(payload.pop("time_zone", "") or "America/New_York")
        try:
            AssistantPageContext(time_zone=time_zone)
            call = AssistantToolCall.model_validate({**payload, "name": name})
        except ValidationError as exc:
            raise McpProtocolError(_INVALID_PARAMS, _validation_message(exc)) from exc
        try:
            evidence = self.deps.intelligence.run_assistant_tool(call, time_zone)
        except HTTPException as exc:
            if exc.status_code == 503:
                raise McpProtocolError(
                    _INTERNAL_ERROR,
                    "SurvNG is shutting down",
                    status_code=503,
                ) from exc
            detail = exc.detail if isinstance(exc.detail, str) else "request failed"
            return _text_result(detail[:500], is_error=True)
        return _text_result({"evidence": evidence})

    def _with_manager(self, operation: Callable[[AppManager], _ToolOutcome]) -> _ToolOutcome:
        with self.deps.manager_lock:
            if self.deps.application_stopping.is_set():
                raise McpProtocolError(
                    _INTERNAL_ERROR,
                    "SurvNG is shutting down",
                    status_code=503,
                )
            return operation(self.deps.get_manager())

    def _incident_image(self, event_id: int) -> _ToolOutcome:
        def load(manager: AppManager) -> _ToolOutcome:
            event = manager.events.get(event_id)
            if event is None:
                return _text_result("event not found", is_error=True)
            try:
                snapshot = event_snapshot_path(
                    manager.storage_dir,
                    event,
                    getattr(manager, "media_storage", None),
                )
                encoded = _bounded_jpeg(snapshot)
            except (FileNotFoundError, PermissionError):
                return _text_result("event snapshot is unavailable", is_error=True)
            return _ToolOutcome([
                {
                    "type": "text",
                    "text": json.dumps({"event_id": event_id}, separators=(",", ":")),
                },
                {
                    "type": "image",
                    "data": base64.b64encode(encoded).decode("ascii"),
                    "mimeType": "image/jpeg",
                },
            ])

        return self._with_manager(load)

    def _recent_incidents(self, limit: int) -> _ToolOutcome:
        def load(manager: AppManager) -> _ToolOutcome:
            rows: list[Any] = []
            for item in manager.incidents.snapshot():
                if not manager.incident_notification_allowed(item):
                    continue
                raw = manager.incident_notification_payload(item)
                if not isinstance(raw, dict):
                    continue
                href = str(raw.get("incident_path") or "")
                safe = sanitize_assistant_data(raw)
                if not isinstance(safe, dict):
                    continue
                if _is_safe_app_href(href):
                    safe["href"] = href
                rows.append(safe)
            # The journal snapshot is newest-first. Keep that order and cap it.
            return _text_result({"incidents": rows[:limit]})

        return self._with_manager(load)

    def _camera_command(
        self,
        operation: str,
        camera_id: str,
        enabled: bool | None = None,
    ) -> _ToolOutcome:
        def run(manager: AppManager) -> _ToolOutcome:
            if operation == "start_camera":
                ok = manager.start_camera(camera_id)
                body: dict[str, Any] = {"ok": bool(ok), "camera_id": camera_id}
            elif operation == "stop_camera":
                ok = manager.stop_camera(camera_id)
                body = {"ok": bool(ok), "camera_id": camera_id}
            elif operation == "set_recording":
                ok = manager.set_recording(camera_id, bool(enabled))
                body = {
                    "ok": bool(ok),
                    "camera_id": camera_id,
                    "recording_enabled": bool(enabled),
                }
            else:
                ok = manager.set_detection(camera_id, bool(enabled))
                body = {
                    "ok": bool(ok),
                    "camera_id": camera_id,
                    "detection_enabled": bool(enabled),
                }
            if not ok:
                return _text_result("camera not found", is_error=True)
            return _text_result(body)

        return self._with_manager(run)

    @staticmethod
    def _error_response(rpc_id: Any, exc: McpProtocolError) -> JSONResponse:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": exc.code, "message": exc.message},
            },
            status_code=exc.status_code,
        )


def create_mcp_router(deps: McpDependencies) -> APIRouter:
    router = APIRouter()
    endpoint = McpEndpoint(deps)

    async def mcp(request: Request) -> Response:
        return await endpoint.handle(request)

    router.add_api_route("/api/mcp", mcp, methods=["POST"])
    return router
