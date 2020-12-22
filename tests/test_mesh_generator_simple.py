import numpy as np

from oceanmesh import generate_mesh, simp_vol


def test_mesh_generator_rectangle():
    h0 = 0.1
    bbox = (0.0, 1.0, 0.0, 1.0)

    def drectangle(p, x1, x2, y1, y2):
        min = np.minimum
        return -min(min(min(-y1 + p[:, 1], y2 - p[:, 1]), -x1 + p[:, 0]), x2 - p[:, 0])

    def domain(x):
        return drectangle(x, *bbox)

    def edge_length(p):
        return np.array([0.1] * len(p))

    points, cells = generate_mesh(
        domain=domain, edge_length=edge_length, h0=h0, bbox=bbox
    )

    assert np.isclose(np.sum(simp_vol(points, cells)), 1.0, 0.01)
