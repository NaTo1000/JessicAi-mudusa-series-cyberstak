"""
HiveNode – a single participant in the distributed CPU/RAM hive.

Each node:
* Monitors its own resources and enforces the 5 % cap.
* Advertises available headroom to the mesh.
* Accepts task packets from the scheduler when headroom allows.
* Sends heartbeats so the cluster manager can detect failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .resource_monitor import MAX_CPU_PERCENT, MAX_RAM_PERCENT, ResourceMonitor, ResourceSnapshot

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECS: float = 5.0
TASK_TIMEOUT_SECS: float = 60.0


@dataclass
class NodeInfo:
    """Serialisable description of a node broadcast over the mesh."""

    node_id: str
    host: str
    port: int
    cpu_headroom: float = 0.0
    ram_headroom: float = 0.0
    active_tasks: int = 0
    last_seen: float = field(default_factory=time.monotonic)
    public_key_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "cpu_headroom": round(self.cpu_headroom, 3),
            "ram_headroom": round(self.ram_headroom, 3),
            "active_tasks": self.active_tasks,
            "last_seen": self.last_seen,
            "public_key_fingerprint": self.public_key_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NodeInfo":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class TaskPacket:
    """A unit of work dispatched to a node."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: bytes = b""
    priority: int = 5          # 1 (highest) … 10 (lowest)
    submitted_at: float = field(default_factory=time.monotonic)
    checksum: str = ""         # SHA-256 of payload

    def __post_init__(self) -> None:
        if not self.checksum and self.payload:
            self.checksum = hashlib.sha256(self.payload).hexdigest()

    def verify(self) -> bool:
        """Return True if the payload matches its checksum."""
        return hashlib.sha256(self.payload).hexdigest() == self.checksum


@dataclass
class TaskResult:
    task_id: str
    result: Any
    elapsed_secs: float
    node_id: str
    error: Optional[str] = None
    checksum: str = ""

    def __post_init__(self) -> None:
        if not self.checksum and self.result is not None:
            payload = json.dumps(self.result, default=str).encode()
            self.checksum = hashlib.sha256(payload).hexdigest()


class HiveNode:
    """
    Autonomous hive participant.

    Parameters
    ----------
    host : str
        Bind address for the node's task-reception socket.
    port : int
        Listen port.
    node_id : str | None
        Stable identifier.  Auto-generated if not supplied.
    max_cpu : float
        Hard CPU cap in percent (default 5 %).
    max_ram : float
        Hard RAM cap in percent (default 5 %).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 7331,
        node_id: Optional[str] = None,
        max_cpu: float = MAX_CPU_PERCENT,
        max_ram: float = MAX_RAM_PERCENT,
    ) -> None:
        self.node_id: str = node_id or str(uuid.uuid4())
        self.host = host
        self.port = port
        self._monitor = ResourceMonitor(max_cpu=max_cpu, max_ram=max_ram)
        self._monitor.register_callback(self._on_resource_update)
        self._active_tasks: Dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self._result_callbacks: List[Callable[[TaskResult], None]] = []
        self._running = False
        self._server: Optional[asyncio.AbstractServer] = None
        self._cap_exceeded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._monitor.start()
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        asyncio.create_task(self._heartbeat_loop())
        logger.info(
            "HiveNode %s started on %s:%d", self.node_id[:8], self.host, self.port
        )

    async def stop(self) -> None:
        self._running = False
        await self._monitor.stop()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        for t in list(self._active_tasks.values()):
            t.cancel()
        logger.info("HiveNode %s stopped", self.node_id[:8])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def info(self) -> NodeInfo:
        snap = self._monitor.snapshot
        return NodeInfo(
            node_id=self.node_id,
            host=self.host,
            port=self.port,
            cpu_headroom=snap.headroom_cpu,
            ram_headroom=snap.headroom_ram,
            active_tasks=len(self._active_tasks),
        )

    def register_result_callback(self, fn: Callable[[TaskResult], None]) -> None:
        self._result_callbacks.append(fn)

    async def submit_task(self, packet: TaskPacket) -> Optional[TaskResult]:
        """
        Execute a task locally if resources allow, otherwise return None.
        The caller (scheduler) will redirect to another node when None is returned.
        """
        if self._cap_exceeded:
            logger.debug(
                "Node %s at cap – rejecting task %s", self.node_id[:8], packet.task_id[:8]
            )
            return None
        if not packet.verify():
            logger.warning("Task %s failed checksum verification", packet.task_id[:8])
            return None

        task = asyncio.create_task(
            self._execute(packet), name=f"task-{packet.task_id[:8]}"
        )
        self._active_tasks[packet.task_id] = task
        try:
            result = await asyncio.wait_for(task, timeout=TASK_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            result = TaskResult(
                task_id=packet.task_id,
                result=None,
                elapsed_secs=TASK_TIMEOUT_SECS,
                node_id=self.node_id,
                error="timeout",
            )
        except asyncio.CancelledError:
            result = TaskResult(
                task_id=packet.task_id,
                result=None,
                elapsed_secs=0.0,
                node_id=self.node_id,
                error="cancelled",
            )
        finally:
            self._active_tasks.pop(packet.task_id, None)

        for cb in self._result_callbacks:
            try:
                cb(result)
            except Exception:
                logger.exception("Result callback error")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute(self, packet: TaskPacket) -> TaskResult:
        t0 = time.monotonic()
        try:
            # Deserialise the payload (expects JSON-encoded callable descriptor)
            descriptor = json.loads(packet.payload)
            fn_name = descriptor.get("fn", "noop")
            args = descriptor.get("args", [])
            result = await asyncio.get_event_loop().run_in_executor(
                None, _dispatch_fn, fn_name, args
            )
            return TaskResult(
                task_id=packet.task_id,
                result=result,
                elapsed_secs=time.monotonic() - t0,
                node_id=self.node_id,
            )
        except Exception as exc:
            return TaskResult(
                task_id=packet.task_id,
                result=None,
                elapsed_secs=time.monotonic() - t0,
                node_id=self.node_id,
                error=str(exc),
            )

    def _on_resource_update(self, snap: ResourceSnapshot) -> None:
        self._cap_exceeded = not snap.is_within_cap

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECS)
            logger.debug("Heartbeat – node=%s %s", self.node_id[:8], self.info.to_dict())

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Accept a raw task packet sent over the wire."""
        try:
            raw = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            packet_data = json.loads(raw)
            packet = TaskPacket(
                task_id=packet_data["task_id"],
                payload=packet_data["payload"].encode(),
                priority=packet_data.get("priority", 5),
                checksum=packet_data.get("checksum", ""),
            )
            result = await self.submit_task(packet)
            if result:
                writer.write(json.dumps(result.__dict__).encode())
            else:
                writer.write(b'{"error":"cap_exceeded"}')
            await writer.drain()
        except Exception:
            logger.exception("Connection handler error")
        finally:
            writer.close()


# ---------------------------------------------------------------------------
# Simple built-in function registry (extended at integration time)
# ---------------------------------------------------------------------------

_FN_REGISTRY: Dict[str, Callable] = {}


def register_fn(name: str) -> Callable:
    """Decorator to register a function by name for remote dispatch."""
    def decorator(fn: Callable) -> Callable:
        _FN_REGISTRY[name] = fn
        return fn
    return decorator


def _dispatch_fn(fn_name: str, args: list) -> Any:
    fn = _FN_REGISTRY.get(fn_name)
    if fn is None:
        raise ValueError(f"Unknown function: {fn_name!r}")
    return fn(*args)


@register_fn("noop")
def _noop(*_: Any) -> None:
    return None


@register_fn("ping")
def _ping() -> str:
    return "pong"
