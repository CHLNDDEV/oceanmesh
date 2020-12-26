import os
import sys

import pytest
import numpy as np

import oceanmesh


@pytest.mark.serial
def test_verbose():
    def square(p, x1=0.0, x2=10, y1=0.0, y2=1.0):
        min = np.minimum
        return -min(min(min(-y1 + p[:, 1], y2 - p[:, 1]), -x1 + p[:, 0]), x2 - p[:, 0])

    def constant_size(p):
        return np.array([0.1] * len(p))

    for verbosity, correct_size in zip([0, 1, 2], [0, 125, 5947]):
        sys.stdout = open("output.txt", "w")
        points, cells = oceanmesh.generate_mesh(
            domain=square,
            edge_length=constant_size,
            min_edge_length=0.10,
            bbox=(0.0, 1.0, 0.0, 1.0),
            verbose=verbosity,
        )
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        output_size = os.path.getsize("output.txt")

        assert output_size == correct_size


if __name__ == "__main__":
    test_verbose()
