#!/usr/bin/env python3
"""Compute a valid 2-D filter operation using explicit sliding windows."""

import argparse

import numpy as np


INPUT = np.array(
    [
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
    ],
    dtype=int,
)

FILTER = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=int,
)


def convolve_2d(input_array: np.ndarray, filter_array: np.ndarray, stride: int = 1) -> np.ndarray:
    """Slide a filter over a 2-D array and compute each window's dot product."""
    # Basic checks so the function is used correctly.
    if input_array.ndim != 2 or filter_array.ndim != 2:
        raise ValueError("Input and filter must both be 2-D arrays.")
    if stride < 1:
        raise ValueError("Stride must be a positive integer.")

    # Get the size of the input image and the filter.
    input_height, input_width = input_array.shape
    filter_height, filter_width = filter_array.shape
    if filter_height > input_height or filter_width > input_width:
        raise ValueError("Filter dimensions cannot exceed input dimensions.")

    # The output is smaller than the input because the filter slides across it.
    # We compute how many positions fit in each direction.
    output_height = (input_height - filter_height) // stride + 1
    output_width = (input_width - filter_width) // stride + 1

    # Create an empty array to store the convolution result.
    output = np.zeros((output_height, output_width), dtype=np.result_type(input_array, filter_array))

    # Move the filter across the input one output position at a time.
    for output_row in range(output_height):
        for output_column in range(output_width):
            # Top-left corner of the current filter window.
            row_start = output_row * stride
            column_start = output_column * stride

            # Extract the slice of the input that the filter is currently covering.
            # Example: if the filter is 3x3 and we are at row_start=1, col_start=2,
            # this grabs rows 1:4 and columns 2:5.
            window = input_array[
                row_start : row_start + filter_height,
                column_start : column_start + filter_width,
            ]

            # Multiply the window and filter element-wise, then add everything up.
            # This is the dot product (sum of element-wise products) for this location.
            output[output_row, output_column] = np.sum(window * filter_array)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2-D convolution example.")
    parser.add_argument("--stride", type=int, default=1, help="Distance between filter positions (default: 1).")
    arguments = parser.parse_args()

    output = convolve_2d(INPUT, FILTER, arguments.stride)
    print(f"Stride: {arguments.stride}")
    print("Output feature map:")
    print(output)
    print(f"Output shape: {output.shape}")
    print(
        "With stride 2, the filter moves two positions at a time, so it evaluates "
        "fewer locations; for this input the output shape is (2, 2) instead of (3, 3)."
    )


if __name__ == "__main__":
    main()