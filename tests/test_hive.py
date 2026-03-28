"""
Tests for the HiveNode task dispatch and lifecycle.
"""
from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from hive.node import HiveNode, TaskPacket, TaskResult, register_fn, _dispatch_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_packet(fn: str = "ping", args=None, priority: int = 5) -> TaskPacket:
    payload = json.dumps({"fn": fn, "args": args or []}).encode()
    return TaskPacket(payload=payload, priority=priority)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestTaskPacket:
    def test_checksum_auto_set(self):
        p = make_packet()
        assert len(p.checksum) == 64  # SHA-256 hex

    def test_verify_valid(self):
        p = make_packet()
        assert p.verify()

    def test_verify_tampered(self):
        p = make_packet()
        p.payload = b"tampered"
        assert not p.verify()


class TestFunctionRegistry:
    def test_ping(self):
        assert _dispatch_fn("ping", []) == "pong"

    def test_noop(self):
        assert _dispatch_fn("noop", []) is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown function"):
            _dispatch_fn("__nonexistent__", [])

    def test_custom_registration(self):
        @register_fn("test_add")
        def _add(a, b):
            return a + b

        assert _dispatch_fn("test_add", [3, 4]) == 7


class TestHiveNode:
    @pytest.mark.asyncio
    async def test_lifecycle(self):
        node = HiveNode(host="127.0.0.1", port=17001)
        await node.start()
        info = node.info
        assert info.node_id == node.node_id
        assert info.port == 17001
        await node.stop()

    @pytest.mark.asyncio
    async def test_submit_ping(self):
        node = HiveNode(host="127.0.0.1", port=17002)
        await node.start()
        packet = make_packet("ping")
        result = await node.submit_task(packet)
        assert result is not None
        assert result.result == "pong"
        assert not result.error
        await node.stop()

    @pytest.mark.asyncio
    async def test_submit_invalid_checksum_rejected(self):
        node = HiveNode(host="127.0.0.1", port=17003)
        await node.start()
        packet = make_packet("ping")
        packet.payload = b"corrupted"
        # checksum mismatch → should return None
        result = await node.submit_task(packet)
        assert result is None
        await node.stop()

    @pytest.mark.asyncio
    async def test_cap_exceeded_rejects_tasks(self):
        node = HiveNode(host="127.0.0.1", port=17004)
        await node.start()
        # Simulate cap exceeded
        node._cap_exceeded = True
        packet = make_packet("ping")
        result = await node.submit_task(packet)
        assert result is None
        await node.stop()

    @pytest.mark.asyncio
    async def test_result_callback_called(self):
        received = []
        node = HiveNode(host="127.0.0.1", port=17005)
        node.register_result_callback(received.append)
        await node.start()
        packet = make_packet("ping")
        await node.submit_task(packet)
        assert len(received) == 1
        assert received[0].result == "pong"
        await node.stop()
