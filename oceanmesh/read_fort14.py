import numpy as np
import os

def read_fort14(filename, read_bou=False):
    """
    Reads an ADCIRC fort.14 mesh file.

    Args:
        filename (str): Name of the input file. Default is 'fort.14'.
        read_bou (bool): If True, also reads the boundary data.

    Returns:
        tuple: EToV, VX, B, opedat, boudat, title
    """
    print("Reading fort.14 file")

    # Assert to check if the file exists
    assert os.path.exists(filename), f"Error: File '{filename}' does not exist at the specified path."

    with open(filename, 'r') as fid:
        # Leer la primera línea con el título
        mesh_title = fid.readline().strip()
        print(mesh_title)

        # Leer el número de nodos y elementos
        msgline = fid.readline().strip()
        N = list(map(int, msgline.split()))

        # Leer nodos (número, posición, profundidad)
        Val = np.loadtxt(fid, max_rows=N[1], dtype=float)

        # Leer triangulación
        idx = np.loadtxt(fid, max_rows=N[0], dtype=int)

        # Ordenar los datos leídos
        elements = idx[:, 2:5]
        num_nodes = np.max(elements)
        nodes = np.full((num_nodes, 2), np.nan)
        bathymetry = np.full(num_nodes, np.nan)

        nodes[Val[:, 0].astype(int) - 1, :] = Val[:, 1:3]
        bathymetry[Val[:, 0].astype(int) - 1] = Val[:, 3]

        opedat = None
        boudat = None

        if read_bou:
            # Leer frontera abierta
            line_nope = fid.readline().strip()
            num_open_boundaries = int(line_nope.split('=')[0].strip())  # Extraer el número antes del signo '='
            line_neta = fid.readline().strip()
            num_open_bound_nodes = int(line_neta.split('=')[0].strip())  # Extraer el número antes del signo '='

            num_nodes_each_open_bound = np.zeros(num_open_boundaries, dtype=int)
            bound_type = np.zeros(num_open_boundaries, dtype=int)
            nbdv = np.zeros((num_open_bound_nodes, num_open_boundaries), dtype=int)

            for i in range(num_open_boundaries):
                line = fid.readline().strip()
                varg = int(line.split()[0])
                num_nodes_each_open_bound[i] = varg
                nbdv[:num_nodes_each_open_bound[i], i] = np.loadtxt(fid, max_rows=num_nodes_each_open_bound[i], dtype=int)

            opedat = {
                "num_open_boundaries": num_open_boundaries,
                "num_open_bound_nodes": num_open_bound_nodes,
                "num_nodes_each_open_bound": num_nodes_each_open_bound,
                "bound_type": bound_type,
                "open_boundary_nodes": nbdv[:np.max(num_nodes_each_open_bound), :],
            }

            # Leer frontera terrestre
            line_nbou = fid.readline().strip()
            num_land_boundaries = int(line_nbou.split('=')[0].strip())  # Extraer el número antes del signo '='
            line_nvel = fid.readline().strip()
            total_land_bound_nodes = int(line_nvel.split('=')[0].strip())  # Extraer el número antes del signo '='

            num_nodes_each_land_bound = np.zeros(num_land_boundaries, dtype=int)
            land_boundary_types = np.zeros(num_land_boundaries, dtype=int)
            nbvv = np.zeros((total_land_bound_nodes, num_land_boundaries), dtype=int)

            for i in range(num_land_boundaries):
                line = fid.readline().strip()
                varg = list(map(int, line.split()[:2]))
                num_nodes_each_land_bound[i], land_boundary_types[i] = varg

                if land_boundary_types[i] in [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 60, 61, 101, 52]:
                    nbvv[:num_nodes_each_land_bound[i], i] = np.loadtxt(fid, max_rows=num_nodes_each_land_bound[i], dtype=int)

            boudat = {
                "num_land_boundaries": num_land_boundaries,
                "total_land_bound_nodes": total_land_bound_nodes,
                "num_nodes_each_land_bound": num_nodes_each_land_bound,
                "land_boundary_types": land_boundary_types,
                "nbvv": nbvv[:np.max(num_nodes_each_land_bound), :],
            }

    return elements, nodes, bathymetry, opedat, boudat, mesh_title
