"""
Test script for the enhanced Astrixcore AX-1 ONNX compiler MVP.

Tests Conv1d, Conv2d, Relu, Add operators and unsupported operator detection.
"""

import numpy as np
import sys
import os
from onnx import TensorProto, ModelProto, GraphProto, NodeProto, helper

sys.path.insert(0, os.path.dirname(__file__))
from onnx_compiler import (
    build_ir_from_onnx_model, SUPPORTED_OPS, AccelCommand,
    gen_conv1d_cmd, gen_conv2d_cmd, gen_relu_cmd, gen_add_cmd,
    validate_graph
)


def make_test_conv1d_model():
    """Conv1d model: 1x8x32 input, 4 output channels, kernel=3, stride=1, padding=1."""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 8, 32])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4, 32])

    W_data = np.random.randn(4, 8, 3).astype(np.float32)
    W = helper.make_tensor('W', TensorProto.FLOAT, [4, 8, 3], W_data)

    conv_node = helper.make_node(
        'Conv', inputs=['X', 'W'], outputs=['Conv_out'],
        kernel_shape=[3], strides=[1], pads=[1, 1], group=1
    )
    relu_node = helper.make_node('Relu', inputs=['Conv_out'], outputs=['Y'])

    graph = helper.make_graph([conv_node, relu_node], 'test_conv1d', [X], [Y], [W])
    model = helper.make_model(graph, producer_name='test')
    model.opset_import = [helper.make_opsetid('', 18)]
    return model


def make_test_conv2d_model():
    """Conv2d model: 1x3x32x32 input, 8 output channels, kernel=3, stride=1, padding=1."""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 32, 32])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 8, 32, 32])

    W_data = np.random.randn(8, 3, 3, 3).astype(np.float32)
    W = helper.make_tensor('W', TensorProto.FLOAT, [8, 3, 3, 3], W_data)

    B_data = np.random.randn(8).astype(np.float32)
    B = helper.make_tensor('B', TensorProto.FLOAT, [8], B_data)

    conv_node = helper.make_node(
        'Conv', inputs=['X', 'W', 'B'], outputs=['Conv_out'],
        kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], group=1
    )
    relu_node = helper.make_node('Relu', inputs=['Conv_out'], outputs=['Y'])

    graph = helper.make_graph([conv_node, relu_node], 'test_conv2d', [X], [Y], [W, B])
    model = helper.make_model(graph, producer_name='test')
    model.opset_import = [helper.make_opsetid('', 18)]
    return model


def make_test_add_model():
    """Add model: Y = X + X (element-wise)."""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 8, 32, 32])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 8, 32, 32])

    add_node = helper.make_node('Add', inputs=['X', 'X'], outputs=['Y'])

    graph = helper.make_graph([add_node], 'test_add', [X], [Y])
    model = helper.make_model(graph, producer_name='test')
    model.opset_import = [helper.make_opsetid('', 18)]
    return model


def make_test_unsupported_model():
    """Model with Unsupported operator (MaxPool)."""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 32, 32])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 16, 16])

    pool_node = helper.make_node('MaxPool', inputs=['X'], outputs=['Y'],
                                 kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])

    graph = helper.make_graph([pool_node], 'test_unsupported', [X], [Y])
    model = helper.make_model(graph, producer_name='test')
    model.opset_import = [helper.make_opsetid('', 18)]
    return model


def test_conv1d():
    """Test Conv1d + Relu model."""
    print("=" * 60)
    print("Test 1: Conv1d model")
    print("=" * 60)

    model = make_test_conv1d_model()
    tmp_path = '/tmp/test_conv1d.onnx'
    onnx.save(model, tmp_path)

    commands, errors = build_ir_from_onnx_model(model, debug=False)

    print(f"\nCommands generated: {len(commands)}")
    print(f"Errors: {len(errors)}")

    for i, cmd in enumerate(commands, 1):
        print(f"  Command {i}: {cmd} -> 0x{cmd.to_word():08X}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")

    # Should generate at least one command (Conv1d -> MAC, then Relu -> OP_RELU)
    assert len(commands) >= 1, "Expected at least one command"
    # Check opcodes are supported
    opcodes = {cmd.opcode for cmd in commands}
    supported = {AccelCommand.OP_MAC, AccelCommand.OP_RELU}
    assert opcodes.issubset(supported), f"Unexpected opcodes: {opcodes - supported}"

    print("PASS: Conv1d test\n")
    return True


def test_conv2d():
    """Test Conv2d + Relu model."""
    print("=" * 60)
    print("Test 2: Conv2d model")
    print("=" * 60)

    model = make_test_conv2d_model()
    tmp_path = '/tmp/test_conv2d.onnx'
    onnx.save(model, tmp_path)

    commands, errors = build_ir_from_onnx_model(model, debug=False)

    print(f"\nCommands generated: {len(commands)}")
    print(f"Errors: {len(errors)}")

    for i, cmd in enumerate(commands, 1):
        print(f"  Command {i}: {cmd} -> 0x{cmd.to_word():08X}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")

    assert len(commands) >= 1, "Expected at least one command"
    opcodes = {cmd.opcode for cmd in commands}
    supported = {AccelCommand.OP_MAC, AccelCommand.OP_RELU}
    assert opcodes.issubset(supported), f"Unexpected opcodes: {opcodes - supported}"

    print("PASS: Conv2d test\n")
    return True


def test_add():
    """Test Add model (element-wise)."""
    print("=" * 60)
    print("Test 3: Add model")
    print("=" * 60)

    model = make_test_add_model()
    tmp_path = '/tmp/test_add.onnx'
    onnx.save(model, tmp_path)

    commands, errors = build_ir_from_onnx_model(model, debug=False)

    print(f"\nCommands generated: {len(commands)}")
    print(f"Errors: {len(errors)}")

    # Add is delegated to RISC-V software in MVP, so 0 commands is acceptable
    # but we should at least not crash
    for e in errors:
        print(f"  - {e}")

    # At minimum, no unexpected crashes
    print("PASS: Add test (Add delegated to RISC-V software in MVP)\n")
    return True


def test_unsupported():
    """Test that unsupported operators are properly reported."""
    print("=" * 60)
    print("Test 4: Unsupported operator detection")
    print("=" * 60)

    model = make_test_unsupported_model()
    commands, errors = build_ir_from_onnx_model(model, debug=False)

    print(f"\nCommands generated: {len(commands)}")
    print(f"Errors: {len(errors)}")

    for e in errors:
        print(f"  - {e}")

    # Must have at least one error about unsupported operator
    assert len(errors) > 0, "Expected errors for unsupported operator"
    error_text = ' '.join(errors)
    assert 'MaxPool' in error_text or 'Unsupported' in error_text, \
        f"Expected error mentioning MaxPool or Unsupported, got: {errors}"

    print("PASS: Unsupported operator test\n")
    return True


def main():
    results = []
    try:
        results.append(("Conv1d", test_conv1d()))
    except Exception as e:
        print(f"FAIL: Conv1d test exception: {e}\n")
        results.append(("Conv1d", False))

    try:
        results.append(("Conv2d", test_conv2d()))
    except Exception as e:
        print(f"FAIL: Conv2d test exception: {e}\n")
        results.append(("Conv2d", False))

    try:
        results.append(("Add", test_add()))
    except Exception as e:
        print(f"FAIL: Add test exception: {e}\n")
        results.append(("Add", False))

    try:
        results.append(("Unsupported", test_unsupported()))
    except Exception as e:
        print(f"FAIL: Unsupported test exception: {e}\n")
        results.append(("Unsupported", False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())