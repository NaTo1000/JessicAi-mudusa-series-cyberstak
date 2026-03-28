"""
Pex-Chain — Distributed Decision Logic
========================================

Implements the Pex-Chained system that distributes decision-making
logic across a graph of :class:`PexNode` instances connected in
directed chains and sub-chains.

Core concepts
-------------
PexNode
    A processing node that holds a callable *handler* and an ordered
    list of downstream *children*.  When a node is invoked it runs its
    handler, then cascades the result to every child in sequence.

PexChain
    A container that registers named nodes, sets up directed edges
    between them, and provides a single :meth:`execute` entry-point
    that triggers the cascade from a chosen root node.

Cascading semantics
--------------------
Each node receives the *accumulated context* dict produced by all
previously executed nodes.  A node's handler is called with this
context and may return a dict of additional key/value pairs to merge
into the context.  Returning ``None`` is equivalent to returning an
empty dict (no-op contribution).

This allows seamless synchronisation: every node downstream of a
given node automatically sees all data that was produced by its
ancestors.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Handler = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


# ---------------------------------------------------------------------------
# PexNode
# ---------------------------------------------------------------------------


class PexNode:
    """A single processing node in a Pex-Chain graph.

    Parameters
    ----------
    name:
        Unique human-readable identifier for this node.
    handler:
        Callable that receives the accumulated context dict and
        optionally returns a dict of new key/value pairs to merge.
    """

    def __init__(self, name: str, handler: Handler) -> None:
        self.name = name
        self._handler = handler
        self._children: List["PexNode"] = []
        self._last_contribution: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------

    def add_child(self, child: "PexNode") -> "PexNode":
        """Append *child* as a downstream node and return *self* for chaining."""
        self._children.append(child)
        return self

    # ------------------------------------------------------------------

    def invoke(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this node's handler and cascade to all children.

        Parameters
        ----------
        context:
            Mutable accumulated context dict.  Updated in-place.

        Returns
        -------
        Dict[str, Any]
            The final state of *context* after this node and all its
            descendants have been invoked.
        """
        logger.debug("PexNode %r invoking handler.", self.name)
        result = self._handler(context)
        if result:
            self._last_contribution = dict(result)
            context.update(result)
        else:
            self._last_contribution = {}
        for child in self._children:
            child.invoke(context)
        return context

    # ------------------------------------------------------------------

    @property
    def children(self) -> List["PexNode"]:
        """Immutable view of downstream children."""
        return list(self._children)

    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """SHA-256 of the last contribution dict."""
        payload = str(sorted((self._last_contribution or {}).items())).encode()
        return hashlib.sha256(payload).hexdigest()

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"PexNode(name={self.name!r}, children={[c.name for c in self._children]})"


# ---------------------------------------------------------------------------
# PexChain
# ---------------------------------------------------------------------------


class PexChain:
    """Registry and executor for a Pex-Chained node graph.

    Parameters
    ----------
    name:
        Human-readable label for this chain (for logging/diagnostics).

    Examples
    --------
    >>> def ingest(ctx):
    ...     return {"raw": ctx.get("input", [])}
    >>> def preprocess(ctx):
    ...     return {"features": [v * 0.1 for v in ctx.get("raw", [])]}
    >>> chain = PexChain("demo")
    >>> chain.register("ingest", ingest)
    >>> chain.register("preprocess", preprocess)
    >>> chain.link("ingest", "preprocess")
    >>> result = chain.execute("ingest", {"input": [1, 2, 3]})
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._nodes: Dict[str, PexNode] = {}

    # ------------------------------------------------------------------

    def register(self, node_name: str, handler: Handler) -> PexNode:
        """Create and register a new :class:`PexNode`.

        Parameters
        ----------
        node_name:
            Unique name for the node.
        handler:
            Processing function for the node.

        Returns
        -------
        PexNode
            The newly created node.

        Raises
        ------
        ValueError
            If a node with *node_name* already exists in this chain.
        """
        if node_name in self._nodes:
            raise ValueError(
                f"PexChain {self.name!r}: node {node_name!r} already registered."
            )
        node = PexNode(node_name, handler)
        self._nodes[node_name] = node
        logger.debug("PexChain %r: registered node %r.", self.name, node_name)
        return node

    # ------------------------------------------------------------------

    def link(self, parent_name: str, child_name: str) -> None:
        """Add a directed edge from *parent_name* to *child_name*.

        Raises
        ------
        KeyError
            If either node name is not registered.
        """
        parent = self._get_node(parent_name)
        child = self._get_node(child_name)
        parent.add_child(child)
        logger.debug(
            "PexChain %r: linked %r → %r.", self.name, parent_name, child_name
        )

    # ------------------------------------------------------------------

    def execute(
        self, root_name: str, initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Trigger the cascade starting from the node named *root_name*.

        Parameters
        ----------
        root_name:
            Name of the root node to start from.
        initial_context:
            Seed data for the context dict.  Defaults to ``{}``.

        Returns
        -------
        Dict[str, Any]
            The accumulated context after the full cascade.
        """
        ctx: Dict[str, Any] = dict(initial_context or {})
        root = self._get_node(root_name)
        root.invoke(ctx)
        logger.info(
            "PexChain %r: execution from %r complete, context keys=%s.",
            self.name,
            root_name,
            list(ctx.keys()),
        )
        return ctx

    # ------------------------------------------------------------------

    def topology(self) -> Dict[str, List[str]]:
        """Return the chain's adjacency map as ``{node: [children]}``."""
        return {
            name: [c.name for c in node.children]
            for name, node in self._nodes.items()
        }

    # ------------------------------------------------------------------

    def _get_node(self, name: str) -> PexNode:
        try:
            return self._nodes[name]
        except KeyError:
            raise KeyError(
                f"PexChain {self.name!r}: unknown node {name!r}. "
                f"Registered nodes: {list(self._nodes.keys())}"
            )

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"PexChain(name={self.name!r}, nodes={list(self._nodes.keys())})"
