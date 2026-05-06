#!/usr/bin/env python3
"""
Verify that PaddlePaddle is correctly using the GPU.
"""

import sys

import paddle


def check_gpu():
    print(f"PaddlePaddle version: {paddle.__version__}")

    # Check if compiled with CUDA
    is_compiled_with_cuda = paddle.device.is_compiled_with_cuda()
    print(f"Compiled with CUDA: {is_compiled_with_cuda}")

    if not is_compiled_with_cuda:
        print("ERROR: PaddlePaddle is NOT compiled with CUDA support.")
        print("Please ensure you installed the correct paddlepaddle-gpu version.")
        return False

    # Check available devices
    # Note: In some environments (Paddle 3.2.0 + CUDA 12.6), get_available_device()
    # might return empty list even if GPU is usable. We rely on functional test below.
    devices = paddle.device.get_available_device()
    print(f"Available devices: {devices}")

    # Try to set device explicitly
    try:
        paddle.device.set_device("gpu:0")
        print("Successfully set device to gpu:0")
    except Exception as e:
        print(f"ERROR: Failed to set device to gpu:0: {e}")
        if not devices:
            print("No devices listed and set_device failed. GPU likely not working.")
        return False

    # Try to allocate a tensor on GPU
    try:
        t = paddle.to_tensor([1, 2, 3])
        place = t.place
        print(f"Successfully created tensor on: {place}")
        if not place.is_gpu_place():
            print("ERROR: Tensor is not on GPU despite request.")
            return False

        # Try a small computation
        t2 = t * 2
        print(f"Computation successful: {t2.numpy()}")

    except Exception as e:
        print(f"ERROR: Failed to use GPU for computation: {e}")
        return False

    if not devices:
        print(
            "\nWARNING: GPU is working but not listed in get_available_device(). "
            "This is a known Paddle 3.2.0 issue."
        )

    print("\nSUCCESS: PaddlePaddle is correctly configured for GPU usage!")
    return True


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
