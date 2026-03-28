"""
quantum_circuits.py – Simulated quantum processing blocks.

Implements quantum-inspired circuit simulation without requiring specialised
hardware.  Each ``QuantumCircuitBlock`` represents a parameterised quantum
circuit with:

  * Hadamard-style superposition initialisation
  * Parameterised rotation gates (Rx, Ry, Rz)
  * CNOT-style entanglement layers
  * Interference-based readout → classical feature vector

The ``QuantumInterferenceLayer`` wires *n* circuit blocks together and
produces an output tensor that the neural mesh can consume.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumCircuitBlock(nn.Module):
    """A single trainable quantum circuit simulation block.

    Models a parameterised quantum circuit operating on ``num_qubits`` qubits
    with ``depth`` alternating layers of rotation gates and entanglement.  The
    final state amplitudes are projected to a ``hidden_size``-dimensional
    output via learned interference weights.

    Args:
        num_qubits (int): Number of simulated qubits.
        depth (int): Number of rotation + entanglement layers.
        hidden_size (int): Output feature dimension.
    """

    def __init__(self, num_qubits: int, depth: int, hidden_size: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.hidden_size = hidden_size

        # Learnable rotation angles – three axes per qubit per layer
        self.rotation_params = nn.ParameterList(
            [
                nn.Parameter(torch.randn(num_qubits, 3) * 0.1)
                for _ in range(depth)
            ]
        )

        # Interference readout: map 2*num_qubits real/imag amplitudes → hidden
        self.readout = nn.Linear(2 * num_qubits, hidden_size, bias=True)

        # Layer norm for stabilised interference output
        self.norm = nn.LayerNorm(hidden_size)

    # ------------------------------------------------------------------
    # Gate primitives (operating on complex amplitude tensors)
    # ------------------------------------------------------------------

    def _apply_rotation(
        self, state: torch.Tensor, qubit: int, angles: torch.Tensor
    ) -> torch.Tensor:
        """Apply single-qubit rotation (Rx·Ry·Rz) in-place on *state*.

        ``state`` shape: ``(batch, 2^num_qubits, 2)`` where dim=-1 holds
        (real, imag).  For efficiency we use a 2×2 unitary approximation via
        first-order Euler decomposition rather than building the full
        2^n × 2^n matrix.
        """
        theta_x, theta_y, theta_z = angles[qubit].unbind(-1)

        # Build 2×2 rotation matrix for the target qubit (complex)
        cx, sx = torch.cos(theta_x / 2), torch.sin(theta_x / 2)
        cy, sy = torch.cos(theta_y / 2), torch.sin(theta_y / 2)
        cz, sz = torch.cos(theta_z / 2), torch.sin(theta_z / 2)

        # Combined rotation: Rz·Ry·Rx  (real + imag stored as float pairs)
        # Single-qubit amplitude update on a subset of amplitudes
        n = 1 << self.num_qubits
        stride = 1 << qubit  # spacing between pairs for this qubit

        state_new = state.clone()
        for k in range(0, n, 2 * stride):
            for j in range(stride):
                idx0 = k + j
                idx1 = k + j + stride
                a0 = state[:, idx0, :]  # (batch, 2)
                a1 = state[:, idx1, :]

                # Rx rotation
                a0_new_r = cx * a0[:, 0] + sx * a1[:, 1]
                a0_new_i = cx * a0[:, 1] - sx * a1[:, 0]
                a1_new_r = cx * a1[:, 0] + sx * a0[:, 1]
                a1_new_i = cx * a1[:, 1] - sx * a0[:, 0]

                # Ry rotation (folded in)
                a0r = cy * a0_new_r - sy * a1_new_r
                a0i = cy * a0_new_i - sy * a1_new_i
                a1r = sy * a0_new_r + cy * a1_new_r
                a1i = sy * a0_new_i + cy * a1_new_i

                # Rz rotation (phase)
                a0_final_r = cz * a0r + sz * a0i
                a0_final_i = cz * a0i - sz * a0r
                a1_final_r = cz * a1r - sz * a1i
                a1_final_i = cz * a1i + sz * a1r

                state_new[:, idx0, 0] = a0_final_r
                state_new[:, idx0, 1] = a0_final_i
                state_new[:, idx1, 0] = a1_final_r
                state_new[:, idx1, 1] = a1_final_i

        return state_new

    def _apply_entanglement(self, state: torch.Tensor) -> torch.Tensor:
        """Apply a ring of CNOT gates (qubit i → qubit (i+1) % n)."""
        n = 1 << self.num_qubits
        state_new = state.clone()
        for ctrl in range(self.num_qubits):
            tgt = (ctrl + 1) % self.num_qubits
            ctrl_stride = 1 << ctrl
            tgt_stride = 1 << tgt
            for k in range(0, n, 2 * ctrl_stride):
                for j in range(ctrl_stride):
                    # Only flip target when control qubit is |1⟩
                    idx_ctrl1 = k + j + ctrl_stride
                    idx_ctrl1_tgt0 = idx_ctrl1 & ~tgt_stride
                    idx_ctrl1_tgt1 = idx_ctrl1 | tgt_stride
                    # Swap target amplitudes
                    tmp = state_new[:, idx_ctrl1_tgt0, :].clone()
                    state_new[:, idx_ctrl1_tgt0, :] = state_new[
                        :, idx_ctrl1_tgt1, :
                    ]
                    state_new[:, idx_ctrl1_tgt1, :] = tmp
        return state_new

    def _interference_readout(self, state: torch.Tensor) -> torch.Tensor:
        """Collapse the quantum state to a classical feature vector.

        Computes the marginal probabilities (sum of squared amplitudes per
        qubit) for both computational-basis projections, yielding a
        ``2 * num_qubits`` real-valued interference pattern.
        """
        n = 1 << self.num_qubits
        features = []
        for q in range(self.num_qubits):
            stride = 1 << q
            amp_0 = []
            amp_1 = []
            for k in range(0, n, 2 * stride):
                for j in range(stride):
                    amp_0.append(state[:, k + j, :])
                    amp_1.append(state[:, k + j + stride, :])
            amp_0_t = torch.stack(amp_0, dim=1)  # (B, count, 2)
            amp_1_t = torch.stack(amp_1, dim=1)
            prob_0 = (amp_0_t ** 2).sum(dim=(1, 2))  # (B,)
            prob_1 = (amp_1_t ** 2).sum(dim=(1, 2))
            features.extend([prob_0, prob_1])
        return torch.stack(features, dim=-1)  # (B, 2*num_qubits)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a batch of inputs through the quantum circuit block.

        Args:
            x: ``(batch, hidden_size)`` input tensor used to seed the
               initial quantum state via amplitude encoding.

        Returns:
            Tensor of shape ``(batch, hidden_size)`` after readout.
        """
        batch = x.shape[0]
        n_amplitudes = 1 << self.num_qubits  # 2^num_qubits

        # Amplitude encoding: project input → initial state vector
        amp_real = x[:, :n_amplitudes] if x.shape[-1] >= n_amplitudes else \
            F.pad(x, (0, n_amplitudes - x.shape[-1]))[:, :n_amplitudes]
        amp_imag = torch.zeros_like(amp_real)

        # Normalise so |ψ⟩ has unit norm
        norm_factor = (amp_real ** 2 + amp_imag ** 2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        amp_real = amp_real / norm_factor

        # State tensor: (batch, 2^n_qubits, 2) — last dim = (real, imag)
        state = torch.stack([amp_real, amp_imag], dim=-1)

        # Alternate rotation + entanglement layers
        for layer_idx in range(self.depth):
            angles = self.rotation_params[layer_idx]
            for q in range(self.num_qubits):
                state = self._apply_rotation(state, q, angles)
            state = self._apply_entanglement(state)

        # Interference readout → (batch, 2*num_qubits)
        features = self._interference_readout(state)

        # Project to hidden_size
        out = self.readout(features)
        return self.norm(out)


class QuantumInterferenceLayer(nn.Module):
    """Wires multiple :class:`QuantumCircuitBlock` together.

    Each block receives the same hidden representation and its output is
    combined via learned weights to produce the final interference signal.
    This models the constructive/destructive interference between parallel
    quantum paths.

    Args:
        num_blocks (int): Number of parallel circuit blocks.
        num_qubits (int): Qubits per block.
        depth (int): Circuit depth per block.
        hidden_size (int): Shared hidden dimension.
    """

    def __init__(
        self,
        num_blocks: int,
        num_qubits: int,
        depth: int,
        hidden_size: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                QuantumCircuitBlock(num_qubits, depth, hidden_size)
                for _ in range(num_blocks)
            ]
        )
        # Learned combination weights (softmax-normalised for interference)
        self.combine_weights = nn.Parameter(torch.ones(num_blocks) / num_blocks)
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the interference combination of all circuit blocks.

        Args:
            x: ``(batch, seq_len, hidden_size)`` or ``(batch, hidden_size)``.

        Returns:
            Tensor of the same shape as *x*.
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)  # (B, 1, H)

        B, T, H = x.shape
        x_flat = x.reshape(B * T, H)

        block_outputs = torch.stack(
            [block(x_flat) for block in self.blocks], dim=-1
        )  # (B*T, H, num_blocks)

        weights = F.softmax(self.combine_weights, dim=0)  # (num_blocks,)
        out_flat = (block_outputs * weights).sum(dim=-1)  # (B*T, H)
        out = self.output_norm(out_flat).reshape(B, T, H)

        return out.squeeze(1) if squeeze else out
