"""Tests for PexChain."""

import pytest
from nonganon_tesseract.pex_chain import PexChain, PexNode


class TestPexNode:
    def test_invoke_calls_handler(self):
        called = []

        def handler(ctx):
            called.append(True)
            return {"done": True}

        node = PexNode("test", handler)
        result = node.invoke({})
        assert called
        assert result["done"] is True

    def test_invoke_cascades_to_children(self):
        order = []

        def parent_handler(ctx):
            order.append("parent")
            return {}

        def child_handler(ctx):
            order.append("child")
            return {}

        parent = PexNode("parent", parent_handler)
        child = PexNode("child", child_handler)
        parent.add_child(child)
        parent.invoke({})
        assert order == ["parent", "child"]

    def test_add_child_returns_self(self):
        node = PexNode("n", lambda ctx: None)
        child = PexNode("c", lambda ctx: None)
        returned = node.add_child(child)
        assert returned is node

    def test_children_is_copy(self):
        node = PexNode("n", lambda ctx: None)
        child = PexNode("c", lambda ctx: None)
        node.add_child(child)
        kids = node.children
        kids.clear()
        assert len(node.children) == 1

    def test_handler_returning_none_no_crash(self):
        node = PexNode("n", lambda ctx: None)
        ctx = {"existing": 1}
        node.invoke(ctx)
        assert ctx == {"existing": 1}

    def test_fingerprint_is_hex(self):
        node = PexNode("n", lambda ctx: {"x": 1})
        node.invoke({})
        fp = node.fingerprint()
        assert len(fp) == 64
        int(fp, 16)


class TestPexChain:
    def _make_simple_chain(self):
        chain = PexChain("test-chain")
        chain.register("a", lambda ctx: {"a_done": True})
        chain.register("b", lambda ctx: {"b_done": True})
        chain.link("a", "b")
        return chain

    def test_register_and_execute(self):
        chain = self._make_simple_chain()
        ctx = chain.execute("a")
        assert ctx.get("a_done") is True
        assert ctx.get("b_done") is True

    def test_duplicate_register_raises(self):
        chain = PexChain("c")
        chain.register("x", lambda ctx: None)
        with pytest.raises(ValueError):
            chain.register("x", lambda ctx: None)

    def test_unknown_root_raises(self):
        chain = PexChain("c")
        with pytest.raises(KeyError):
            chain.execute("nonexistent")

    def test_unknown_link_raises(self):
        chain = PexChain("c")
        chain.register("a", lambda ctx: None)
        with pytest.raises(KeyError):
            chain.link("a", "missing")

    def test_initial_context_passed(self):
        chain = PexChain("c")
        chain.register("reader", lambda ctx: {"saw": ctx.get("seed")})
        result = chain.execute("reader", {"seed": 99})
        assert result["saw"] == 99

    def test_context_accumulates_across_nodes(self):
        chain = PexChain("c")
        chain.register("first", lambda ctx: {"first_val": 1})
        chain.register("second", lambda ctx: {"second_val": ctx["first_val"] + 1})
        chain.link("first", "second")
        result = chain.execute("first")
        assert result["second_val"] == 2

    def test_topology_returns_adjacency(self):
        chain = self._make_simple_chain()
        topo = chain.topology()
        assert topo["a"] == ["b"]
        assert topo["b"] == []

    def test_repr_contains_name(self):
        chain = PexChain("my-chain")
        assert "my-chain" in repr(chain)
