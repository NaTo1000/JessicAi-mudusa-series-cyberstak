"""
Tests for the PEX mesh protocol (packet encoding/verification).
"""
from __future__ import annotations

import asyncio
import os

import pytest

from mesh.pex_mesh import PexMesh
from hive.node import NodeInfo


def make_mesh(node_id: str = "test-node-001", port_offset: int = 0) -> PexMesh:
    key = os.urandom(32)
    info = NodeInfo(
        node_id=node_id,
        host="127.0.0.1",
        port=5400 + port_offset,
    )
    return PexMesh(
        local_info=info,
        session_key=key,
        bind_host="127.0.0.1",
        udp_port=15353 + port_offset,
        tcp_port=15354 + port_offset,
    )


class TestPexMeshCrypto:
    def test_encode_decode_roundtrip(self):
        mesh = make_mesh()
        msg = {"type": "HELLO", "info": {"node_id": "abc", "host": "127.0.0.1", "port": 5000}}
        encoded = mesh._encode_message(msg)
        decoded = mesh._decode_message(encoded)
        assert decoded["type"] == "HELLO"

    def test_wrong_key_rejected(self):
        mesh = make_mesh()
        other = make_mesh(node_id="other-node")  # different session key
        msg = {"type": "TEST"}
        encoded = mesh._encode_message(msg)
        with pytest.raises(ValueError, match="HMAC"):
            other._decode_message(encoded)

    def test_tampered_body_rejected(self):
        mesh = make_mesh()
        msg = {"type": "TEST"}
        encoded = bytearray(mesh._encode_message(msg))
        # Flip a byte in the body
        encoded[-1] ^= 0xFF
        with pytest.raises(ValueError):
            mesh._decode_message(bytes(encoded))

    def test_short_packet_rejected(self):
        mesh = make_mesh()
        with pytest.raises(ValueError):
            mesh._decode_message(b"\x00\x01\x02")

    def test_wrong_magic_rejected(self):
        mesh = make_mesh()
        msg = {"type": "TEST"}
        encoded = bytearray(mesh._encode_message(msg))
        encoded[0] = 0xFF  # corrupt magic
        with pytest.raises(ValueError, match="magic"):
            mesh._decode_message(bytes(encoded))


class TestNodeInfo:
    def test_to_dict_from_dict_roundtrip(self):
        info = NodeInfo(
            node_id="abc123",
            host="192.168.1.1",
            port=7331,
            cpu_headroom=3.5,
            ram_headroom=2.1,
            active_tasks=2,
        )
        d = info.to_dict()
        restored = NodeInfo.from_dict(d)
        assert restored.node_id == info.node_id
        assert restored.host == info.host
        assert restored.port == info.port
        assert abs(restored.cpu_headroom - info.cpu_headroom) < 0.01
