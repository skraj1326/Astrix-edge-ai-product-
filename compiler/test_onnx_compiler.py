"""
Test script for the Astrixcore AX-1 ONNX compiler MVP.

Creates simple ONNX models (Conv2d + Relu) and verifies the compiler
can generate accelerator commands.
"""

import numpy as np
import onnx
from onnx import TensorProto, ModelProto, GraphProto, NodeProto, helper
import sys
import os

# Add compiler to path
sys.path.insert(0, os.path.dirname(__file__))
from onnx_compiler import build_ir_from_onnx_model, SUPPORTED_OPS, AccelCommand


def create_simple_conv2d_model() -> ModelProto:
    """
    Create a simple ONNX Conv2d + Relu model for testing.

    Graph structure:
    Input (X: 1x3x32x32) -> Conv2d (3x3 kernel, stride 1, padding 1) -> Relu -> Output (Y)
    Weights (W: 8x3x3x3) bias (optional)
    """
    # Input tensor type
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 32, 32])

    # Output tensor type
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 8, 32, 32])

    # Weight tensor: 8 output channels, 3 input channels, 3x3 kernel
    W_data = np.random.randn(8, 3, 3, 3).astype(np.float32)
    W = helper.make_tensor('W', TensorProto.FLOAT, [8, 3, 3, 3], W_data)

    # Bias (optional)
    B_data = np.random.randn(8).astype(np.float32)
    B = helper.make_tensor('B', TensorProto.FLOAT, [8], B_data)

    # Conv2d node
    conv_node = helper.make_node(
        'Conv',
        inputs=['X', 'W', 'B'],
        outputs=['Conv_out'],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],  # [left, right, top, bottom] for NCHW
        dilations=[1, 1],
        group=1
    )

    # ReLU node
    relu_node = helper.make_node(
        'Relu',
        inputs=['Conv_out'],
        outputs=['Y']
    )

    # Graph
    graph = helper.make_graph(
        [conv_node, relu_node],
        'simple_conv2d_graph',
        [X],
        [Y],
        [W, B]
    )

    # Model
    model = helper.make_model(graph, producer_name='astrixcore-test')
    model.opset_import = [helper.make_opsetid('', 18)]

    return model


def create_conv1d_model() -> ModelProto:
    """
    Create a simple ONNX Conv1d model for testing.
    """
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 8, 32])

    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4, 32])

    W_data = np.random.randn(4, 8, 3).astype(np.float32)
    W = helper.make_tensor('W', TensorProto.FLOAT, [4, 8, 3], W_data)

    conv_node = helper.make_node(
        'Conv',
        inputs=['X', 'W'],
        outputs=['Conv_out'],
        kernel_shape=[3],
        strides=[1],
        pads=[1, 1],
        group=1
    )

    relu_node = helper.make_node(
        'Relu',
        inputs=['Conv_out'],
        outputs=['Y']
    )

    graph = helper.make_graph(
        [conv_node, relu_node],
        'simple_conv1d_graph',
        [X],
        [Y],
        [W]
    )

    model = helper.make_model(graph, producer_name='astrixcore-test')
    model.opset_import = [helper.make_opsetid('', 18)]

    return model


def test_conv2d_model():
    """Test Conv2d + Relu model through the compiler."""
    print("=" * 60)
    print("Test 1: Conv2d + Relu model")
    print("=" * 60)

    model = create_simple_conv2d_model()

    # Save to temp file
    tmp_path = '/tmp/test_conv2d.onnx'
    onnx.save(model, tmp_path)

    # Build IR
    commands, errors = build_ir_from_onnx_model(model, debug=True)

    print(f"\nCommands generated: {len(commands)}")
    print(f"Errors: {len(errors)}")

    for i, cmd in enumerate(commands, 1):
        print(f"  Command {i}: {cmd} -> 0x{cmd.to_word():08X}")

    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(f"  - {err}")

    # Verify we got at least one command
    assert len(commands) > 0, "Expected at least one accelerator command"
    # Verify command has valid opcode
    assert commands[0].opcode != AccelCommand.OP_IDLE, "Expected a non-idle opcode"
    # Verify it's a MAC or RELU command
    assert commands[0].opcode in (AccelCommand.OP_MAC, AccelCommand.OP_RELU), \
        f"Expected OP_MAC or OP_RELU, got {commands[0].opcode}"

    print("PASS: Conv2d model test\n")
    return True


def test_conv1d_model():
    """Test Conv1d + Relu model through the compiler."""
    print("=" * 60)
    print("Test 2: Conv1d + Relu model")
    print("=" * 60)

    model = create_conv1d_model()

    tmp_path = '/tmp/test_conv1d.onnx'
    onnx.save(model, tmp_path)

    commands, errors = build_ir_from_onnx_model(model, debug=True)

    print(f"\nCommands generated: {len(commands)}")
    print(f"Errors: {len(errors)}")

    for i, cmd in enumerate(commands, 1):
        print(f"  Command {i}: {cmd} -> 0x{cmd.to_word():08X}")

    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(f"  - {err}")

    assert len(commands) > 0, "Expected at least one accelerator command"
    assert commands[0].opcode != AccelCommand.OP_IDLE, "Expected a non-idle opcode"
    assert commands[0].opcode in (AccelCommand.OP_MAC, AccelCommand.OP_RELU), \
        f"Expected OP_MAC or OP_RELU, got {commands[0].opcode}"

    print("PASS: Conv1d model test\n")
    return True


def test_unsupported_operator():
    """Test that unsupported operators are properly reported."""
    print("=" * 60)
    print("Test 3: Unsupported operator detection")
    print("=" * 60)

    # Create a model with a Pooling operator (not supported in MVP)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 32, 32])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 32, 32])

    pool_node = helper.make_node(
        'MaxPool',
        inputs=['X'],
        outputs=['Y'],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0]
    )

    graph = helper.make_graph(
        [pool_node],
        'unsupported_graph',
        [X],
        [Y]
    )

    model = helper.make_model(graph, producer_name='test')
    model.opset_import = [helper.make_opsetid('', 18)]

    commands, errors = build_ir_from_onnx_model(model, debug=True)

    print(f"\nCommands generated: {len(commands)}")
    print(f"Errors: {len(errors)}")

    for err in errors:
        print(f"  - {err}")

    # Pooling should be flagged as unsupported
    assert len(errors) > 0, "Expected errors for unsupported operator"
    # Check that unsupported operator is mentioned
    error_text = ' '.join(errors)
    assert 'MaxPool' in error_text or 'Unsupported' in error_text, \
        f"Expected error mentioning MaxPool or Unsupported, got: {errors}"

    print("PASS: Unsupported operator test\n")
    return True


def main():
    """Run all tests."""
    results = []

    try:
        results.append(("Conv2d model", test_conv2d_model()))
    except Exception as e:
        print(f"FAIL: Conv2d model test exception: {e}\n")
        results.append(("Conv2d model", False))

    try:
        results.append(("Conv1d model", test_conv1d_model()))
    except Exception as e:
        print(f"FAIL: Conv1d model test exception: {e}\n")
        results.append(("Conv1d model", False))

    try:
        results.append(("Unsupported operator", test_unsupported_operator()))
    except Exception as e:
        print(f"FAIL: Unsupported operator test exception: {e}\n")
        results.append(("Unsupported operator", False))

    # Summary
    print("=" * 60)
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