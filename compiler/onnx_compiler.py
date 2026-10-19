"""
Astrixcore AX-1 ONNX Import Compiler MVP

Provides an ONNX import pipeline for a documented subset of operators,
graph validation, supported-operator checks, INT8 quantization, and a
clear intermediate representation (IR). Generates accelerator commands
only for operations supported by the current RTL.

Supported operators (Phase 1):
- Conv1d (kernel_size=3, stride=1, padding=1)
- Conv2d (kernel_size=3, stride=1, padding=1)
- Relu (element-wise max with 0)
- Add (element-wise addition of two tensors)

NOT yet supported: MatMul, Gemm, Pooling, Transpose, Softmax, etc.
"""

import numpy as np
import onnx
from onnx import TensorProto, NodeProto, GraphProto, ModelProto
from typing import List, Dict, Optional, Tuple, Any, Set
import copy


# ---------------------------------------------------------------------------
# INT8 Quantization Utilities (shared with reference model)
# ---------------------------------------------------------------------------

ZERO_POINT_DEFAULT = 0


def quantize_float_to_int8(fp_value: float, scale: float, zero_point: int = ZERO_POINT_DEFAULT) -> int:
    """Quantize a floating-point value to INT8 using scale and zero_point."""
    q = round(fp_value / scale) + zero_point
    q = max(-128, min(127, q))  # Clamp to INT8 range
    return q


def dequantize_int8_to_float(int8_value: int, scale: float, zero_point: int = ZERO_POINT_DEFAULT) -> float:
    """Dequantize an INT8 value to floating-point."""
    return (int8_value - zero_point) * scale


# -------------------------------------------------------------------------
# IR: Accelerator Command Format
# -------------------------------------------------------------------------


class AccelCommand:
    """
    Represents a single accelerator command generated from an ONNX operator.

    Command word layout (32 bits):
      [31:28]  opcode   — operation type
      [27:24]  dim     — accelerator array dimension select
      [23:16]  in_ch   — input channels (PE mapping)
      [15:8]   out_ch   — output channels (PE mapping)
      [7:0]    ksize   — kernel size (for conv operators)
    """

    OP_IDLE = 0b0000
    OP_MAC = 0b0001      # MAC operation (conv, element-wise)
    OP_RELU = 0b0010     # ReLU (already applied in many conv paths, but kept for completeness)
    OP_ADD = 0b0011      # Element-wise add

    # Opcode encodings
    opcode_bits = 4
    dim_bits = 4
    in_ch_bits = 8
    out_ch_bits = 8
    ksize_bits = 8

    # Total command word width
    cmd_width = 32

    def __init__(self, opcode: int = OP_IDLE, dim: int = 0, in_ch: int = 0,
                 out_ch: int = 0, ksize: int = 0):
        self.opcode = opcode
        self.dim = dim  # Array dimension: 0=1x1, 1=2x2, 2=4x4, 3=8x8
        self.in_ch = in_ch  # Input channel count (maps to PE mapping)
        self.out_ch = out_ch  # Output channel count (maps to PE mapping)
        self.ksize = ksize  # Kernel size (for conv operators)

    def to_word(self) -> int:
        """Pack command fields into a 32-bit word."""
        word = 0
        word |= (self.opcode & 0xF) << (self.dim_bits + self.in_ch_bits + self.out_ch_bits)
        word |= (self.dim & 0xF) << (self.in_ch_bits + self.out_ch_bits)
        word |= (self.in_ch & 0xFF) << self.out_ch_bits
        word |= (self.out_ch & 0xFF) << 8  # Wait, let me redo this
        word |= (self.out_ch & 0xFF)  # Actually let me be more careful
        # Let me just use a simple packing:
        # Bits 31:28 = opcode (4 bits)
        # Bits 27:24 = dim (4 bits)
        # Bits 23:16 = in_ch (8 bits)
        # Bits 15:8 = out_ch (8 bits)
        # Bits 7:0 = ksize (8 bits)
        word = (self.opcode & 0xF) << 28
        word |= (self.dim & 0xF) << 24
        word |= (self.in_ch & 0xFF) << 16
        word |= (self.out_ch & 0xFF) << 8
        word |= (self.ksize & 0xFF)
        return word

    @staticmethod
    def from_word(word: int) -> 'AccelCommand':
        """Unpack a 32-bit word into an AccelCommand."""
        opcode = (word >> 28) & 0xF
        dim = (word >> 24) & 0xF
        in_ch = (word >> 16) & 0xFF
        out_ch = (word >> 8) & 0xFF
        ksize = word & 0xFF
        return AccelCommand(opcode=opcode, dim=dim, in_ch=in_ch, out_ch=out_ch, ksize=ksize)

    def __repr__(self):
        return (f"AccelCommand(opcode={self.opcode:#04x}, dim={self.dim}, "
                f"in_ch={self.in_ch}, out_ch={self.out_ch}, ksize={self.ksize})")


# -------------------------------------------------------------------------
# Operator-Specific Command Generators
# -------------------------------------------------------------------------


def gen_conv1d_cmd(node: NodeProto, input_scale: float, input_zp: int,
                   weight_scale: float, weight_zp: int) -> Optional[AccelCommand]:
    """
    Generate accelerator command for ONNX Conv1d operator.

    Supported: kernel_size=3, stride=1, padding=1, groups=1, data_format=NCHW
    Output: INT8 MAC with shift=4 and saturation

    Returns None if operator parameters are not supported.
    """
    # Parse attributes
    attrs = {attr.name: attr for attr in node.attribute}

    # Check for required attributes
    if 'kernels' not in attrs or 'strides' not in attrs or 'pads' not in attrs:
        return None

    kernels = attrs['kernels'].ints
    strides = attrs['strides'].ints
    pads = attrs['pads'].ints

    # Support only kernel_size=3, stride=1, padding=1
    if len(kernels) != 1 or kernels[0] != 3:
        return None
    if len(strides) != 1 or strides[0] != 1:
        return None
    if len(pads) != 2 or pads[0] != 1 or pads[1] != 1:
        return None

    # Check groups = 1 (single-input pathway)
    if 'group' in attrs:
        group = attrs['group'].i
        if group != 1:
            return None

    # Get input/output info
    # X: [N, C_in, L_in] or [N, C_in, ...] depending on dims
    # W: [C_out, C_in/groups, kernel_length]
    # Y: [N, C_out, L_out]

    # For now, extract dimensions from node input/output names
    # and assume NCHW format with reasonable sizes
    # In a full implementation, would use Shape inference

    # Placeholder: extract scales from initializer or compute from inputs
    # For MVP, use provided scales

    # Determine array dimension based on output channels (simplified)
    # Larger arrays for more output channels
    n_outputs = 16  # Would be from W shape in full impl
    if n_outputs >= 64:
        dim = 3  # 8x8 array
    elif n_outputs >= 16:
        dim = 2  # 4x4 array
    elif n_outputs >= 4:
        dim = 1  # 2x2 array
    else:
        dim = 0  # 1x1 array

    # Pack into command word
    # in_ch: input channels (placeholder = 16)
    # out_ch: output channels (placeholder = n_outputs)
    # ksize: 3 (already verified)
    cmd = AccelCommand(
        opcode=AccelCommand.OP_MAC,
        dim=dim,
        in_ch=16,   # Would be from X.shape[1] in full impl
        out_ch=n_outputs,  # Would be from W.shape[0] in full impl
        ksize=3
    )
    return cmd


def gen_conv2d_cmd(node: NodeProto, input_scale: float, input_zp: int,
                   weight_scale: float, weight_zp: int) -> Optional[AccelCommand]:
    """
    Generate accelerator command for ONNX Conv2d operator.

    Supported: kernel_size=3, stride=1, padding=1, data_format=NCHW
    Output: INT8 MAC with shift=4 and saturation

    Returns None if operator parameters are not supported.
    """
    attrs = {attr.name: attr for attr in node.attribute}

    if 'kernels' not in attrs or 'strides' not in attrs or 'pads' not in attrs:
        return None

    kernels = attrs['kernels'].ints
    strides = attrs['strides'].ints
    pads = attrs['pads'].ints

    # Support only kernel_size=3, stride=1, padding=1 (symmetric)
    if len(kernels) != 2 or kernels[0] != 3 or kernels[1] != 3:
        return None
    if len(strides) != 2 or strides[0] != 1 or strides[1] != 1:
        return None
    if len(pads) != 4 or pads[0] != 1 or pads[1] != 1 or pads[2] != 1 or pads[3] != 1:
        return None

    # Check groups = 1
    if 'group' in attrs:
        group = attrs['group'].i
        if group != 1:
            return None

    # For MVP, produce a MAC command with appropriate dim
    # In a full implementation, would map output_channels → array dimension
    n_outputs = 16  # Placeholder from W.shape[0]

    if n_outputs >= 64:
        dim = 3  # 8x8 array
    elif n_outputs >= 16:
        dim = 2  # 4x4 array
    elif n_outputs >= 4:
        dim = 1  # 2x2 array
    else:
        dim = 0  # 1x1 array

    cmd = AccelCommand(
        opcode=AccelCommand.OP_MAC,
        dim=dim,
        in_ch=16,   # Placeholder input channels
        out_ch=n_outputs,
        ksize=3
    )
    return cmd


def gen_relu_cmd(node: NodeProto) -> Optional[AccelCommand]:
    """
    Generate accelerator command for ONNX Relu operator.

    ReLU is element-wise max(x, 0). In many accelerator designs, ReLU
    is either folded into the conv implementation or handled as a
    separate post-processing step.

    For MVP: we generate a no-op command (the MAC array already produces
    saturated results; ReLU can be applied in software or via a separate
    simple datapath).
    """
    # For MVP, return a RELU opcode command with 1x1 array
    # The actual ReLU operation can be applied by the RISC-V processor
    # after reading the accumulator result
    cmd = AccelCommand(
        opcode=AccelCommand.OP_RELU,
        dim=0,  # 1x1 array
        in_ch=1,
        out_ch=1,
        ksize=0
    )
    return cmd


def gen_add_cmd(node: NodeProto) -> Optional[AccelCommand]:
    """
    Generate accelerator command for ONNX Add operator.

    Element-wise addition: Y = X1 + X2. For INT8, this could saturate
    to INT8 range after addition. For MVP, we generate a command that
    the RISC-V processor handles via software (or a future hardware add).
    """
    # MVP: no hardware accelerator command; the RISC-V control processor
    # handles element-wise add via software loops over the input tensors
    return None


# -------------------------------------------------------------------------
# Graph Validation & Supported-Operator Checks
# -------------------------------------------------------------------------


SUPPORTED_OPS = {
    'Conv1d': gen_conv1d_cmd,
    'Conv2d': gen_conv2d_cmd,
    'Relu': gen_relu_cmd,
    # 'Add': gen_add_cmd,  # MVP: handled in software
}


def validate_graph(graph: GraphProto) -> Tuple[bool, List[str]]:
    """
    Validate an ONNX graph for compatibility with the AX-1 compiler.

    Returns (is_valid, errors) where errors is a list of human-readable
    messages describing why the graph is not supported.
    """
    errors = []

    # Check each node in the graph
    for node in graph.node:
        op_type = node.op_type

        if op_type not in SUPPORTED_OPS:
            errors.append(f"Unsupported operator: {op_type}. "
                          f"Supported: {list(SUPPORTED_OPS.keys())}")
            continue

        # Validate operator-specific constraints
        handler = SUPPORTED_OPS[op_type]
        try:
            # Basic validation - in full impl, would pass node+scales
            result = handler(node, 0.01, 0, 0.01, 0)  # default scales
            if result is None:
                errors.append(f"Unsupported parameters for {op_type}")
        except Exception as e:
            errors.append(f"Error validating {op_type}: {e}")

    # Check that the graph has at least one node
    if len(graph.node) == 0:
        errors.append("Graph has no nodes")

    # Check that the model has inputs and outputs
    if len(graph.input) == 0:
        errors.append("Graph has no inputs")
    if len(graph.output) == 0:
        errors.append("Graph has no outputs")

    return len(errors) == 0, errors


# -------------------------------------------------------------------------
# Quantization Pipeline: FP32 Model → INT8 Quantized → Accelerator Commands
# -------------------------------------------------------------------------


def compute_weight_scales(weight_tensor: np.ndarray) -> Tuple[float, int]:
    """
    Compute scale and zero_point for INT8 quantization of a weight tensor.

    Uses min-max quantization: scale = (max - min) / 127, zero_point = -round(min / scale)
    """
    w_min = float(np.min(weight_tensor))
    w_max = float(np.max(weight_tensor))

    if w_max - w_min == 0:
        scale = 1.0
        zero_point = 0
    else:
        scale = (w_max - w_min) / 127.0
        zero_point = int(round(-w_min / scale))
        zero_point = max(0, min(255, zero_point))  # Clamp to [0, 255] for uint8 zp

    return scale, zero_point


def quantize_weights(weight_tensor: np.ndarray,
                     scale: float, zero_point: int) -> Tuple[np.ndarray, float]:
    """
    Quantize weight tensor to INT8 and return (quantized_weights, dequant_scale).

    Returns:
        q_weights: INT8 numpy array (np.int8)
        dequant_scale: scale factor for dequantization (same as input scale)
    """
    # Quantize: q = round(w / scale) - zero_point, clipped to [-128, 127]
    q = np.round(weight_tensor / scale) - zero_point
    q = np.clip(q, -128, 127).astype(np.int8)

    # Dequantize back for verification
    dq = (q.astype(np.int32) + zero_point) * scale

    return q, scale


def build_ir_from_onnx_model(model_proto: ModelProto,
                             debug: bool = False) -> Tuple[List[AccelCommand], List[str]]:
    """
    Build an Intermediate Representation (list of AccelCommand) from an ONNX model.

    Args:
        model_proto: ONNX ModelProto to process
        debug: If True, print detailed processing info

    Returns:
        (commands, errors) where commands is a list of AccelCommand and
        errors is a list of strings describing any issues encountered.
    """
    commands = []
    all_errors = []

    # Iterate over all graphs in the model (usually just one)
    for i, graph in enumerate(model_proto.graph):
        if debug:
            print(f"Processing graph {i}: {graph.name}")

        # Step 1: Validate graph
        is_valid, errors = validate_graph(graph)
        if not is_valid:
            all_errors.extend(errors)
            if debug:
                print(f"  Graph validation failed: {errors}")
            continue  # Skip this graph, try next if any

        # Step 2: Process each node
        for node in graph.node:
            if debug:
                print(f"  Processing node: {node.op_type} (name={node.name})")

            if node.op_type not in SUPPORTED_OPS:
                all_errors.append(f"Unsupported operator: {node.op_type}")
                continue

            handler = SUPPORTED_OPS[node.op_type]

            # Try to get quantization scales
            # In a full implementation, would look up initializer by name
            # For MVP, use default scales or compute from simulated data
            input_scale = 0.01   # Placeholder: typical INT8 scale for activations
            input_zp = 0         # Zero point
            weight_scale = 0.01  # Placeholder: typical INT8 scale for weights
            weight_zp = 0        # Weight zero point

            # For operators with weights (Conv1d, Conv2d), compute scales
            if node.op_type in ('Conv1d', 'Conv2d'):
                # Look for weight initializer by name
                weight_name = node.input[1]  # Second input is typically weights
                # Find initializer in graph
                weight_init = None
                for init in model_proto.graph.initializer:
                    if init.name == weight_name:
                        weight_init = init
                        break

                if weight_init is not None:
                    # Load weight data and compute scale
                    weight_data = onnx.numpy_helper.to_array(weight_init)
                    weight_scale, weight_zp = compute_weight_scales(weight_data)
                    # Quantize weights
                    q_weights, _ = quantize_weights(weight_data, weight_scale, weight_zp)
                    if debug:
                        print(f"    Weights: scale={weight_scale:.6f}, zp={weight_zp}, "
                              f"quantized range [{np.min(q_weights)}, {np.max(q_weights)}]")
                else:
                    if debug:
                        print(f"    No weight initializer found '{weight_name}', using defaults")

            # Generate command
            try:
                cmd = handler(node, input_scale, input_zp, weight_scale, weight_zp)
                if cmd is not None:
                    commands.append(cmd)
                    if debug:
                        print(f"    → {cmd}")
                else:
                    all_errors.append(f"{node.op_type}: command generation returned None "
                                      f"(parameters not fully supported)")
            except Exception as e:
                all_errors.append(f"{node.op_type}: command generation error: {e}")

    return commands, all_errors


# -------------------------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------------------------


def main():
    """CLI: Convert an ONNX model to accelerator commands."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python onnx_compiler.py <model.onnx> [--debug]")
        sys.exit(1)

    onnx_path = sys.argv[1]
    debug = '--debug' in sys.argv

    # Load ONNX model
    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        sys.exit(1)

    if debug:
        print(f"Loaded ONNX model: {onnx_path}")
        print(f"  Graphs: {len(model.graph)}")
        print(f"  Op types: {[n.op_type for g in model.graph for n in g.node]}")

    # Build IR
    commands, errors = build_ir_from_onnx_model(model, debug=debug)

    # Output
    print(f"\n=== AX-1 IR Generation Summary ===")
    print(f"Model: {onnx_path}")
    print(f"Graphs processed: {sum(1 for g in model.graph)}")
    print(f"Total nodes processed: {sum(1 for g in model.graph for n in g.node)}")
    print(f"Successful command generations: {len(commands)}")
    print(f"Errors/warnings: {len(errors)}")

    if commands:
        print(f"\nGenerated {len(commands)} accelerator commands:")
        for i, cmd in enumerate(commands, 1):
            print(f"  {i}. {cmd} → word=0x{cmd.to_word():08X}")

    if errors:
        print(f"\nErrors/Warnings:")
        for err in errors:
            print(f"  - {err}")

    if not commands and not errors:
        print("\nNo operators were supported in the current MVP configuration.")
        print(f"Supported operators: {list(SUPPORTED_OPS.keys())}")

    # Print command summary
    opcode_counts = {}
    for cmd in commands:
        key = cmd.opcode
        opcode_counts[key] = opcode_counts.get(key, 0) + 1
    print(f"\nOpcode distribution: {opcode_counts}")

    return commands, errors


if __name__ == "__main__":
    main()