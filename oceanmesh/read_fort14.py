import numpy as np

def read_fort14(finame="fort.14", read_bou=False):
    """
    Reads an ADCIRC fort.14 mesh file.

    Args:
        finame (str): Name of the input file. Default is 'fort.14'.
        read_bou (bool): If True, also reads the boundary data.

    Returns:
        tuple: EToV, VX, B, opedat, boudat, title
    """
    print("Reading fort.14 file")

    with open(finame, 'r') as fid:
        # Leer la primera línea con el título
        title = fid.readline().strip()
        print(title)

        # Leer el número de nodos y elementos
        msgline = fid.readline().strip()
        N = list(map(int, msgline.split()))

        # Leer nodos (número, posición, profundidad)
        Val = np.loadtxt(fid, max_rows=N[1], dtype=float)

        # Leer triangulación
        idx = np.loadtxt(fid, max_rows=N[0], dtype=int)

        # Ordenar los datos leídos
        EToV = idx[:, 2:5]
        num_nodes = np.max(EToV)
        VX = np.full((num_nodes, 2), np.nan)
        B = np.full(num_nodes, np.nan)

        VX[Val[:, 0].astype(int) - 1, :] = Val[:, 1:3]
        B[Val[:, 0].astype(int) - 1] = Val[:, 3]

        opedat = None
        boudat = None

        if read_bou:
            # Leer frontera abierta
            line_nope = fid.readline().strip()
            nope = int(line_nope.split('=')[0].strip())  # Extraer el número antes del signo '='
            print(nope)
            line_neta = fid.readline().strip()
            neta = int(line_neta.split('=')[0].strip())  # Extraer el número antes del signo '='
            print(neta)

            nvdll = np.zeros(nope, dtype=int)
            ibtypee = np.zeros(nope, dtype=int)
            nbdv = np.zeros((neta, nope), dtype=int)

            for i in range(nope):
                line = fid.readline().strip()
                varg = int(line.split()[0])
                nvdll[i] = varg
                nbdv[:nvdll[i], i] = np.loadtxt(fid, max_rows=nvdll[i], dtype=int)

            opedat = {
                "nope": nope,
                "neta": neta,
                "nvdll": nvdll,
                "ibtypee": ibtypee,
                "nbdv": nbdv[:np.max(nvdll), :],
            }

            # Leer frontera terrestre
            line_nbou = fid.readline().strip()
            nbou = int(line_nbou.split('=')[0].strip())  # Extraer el número antes del signo '='
            line_nvel = fid.readline().strip()
            nvel = int(line_nvel.split('=')[0].strip())  # Extraer el número antes del signo '='

            nvell = np.zeros(nbou, dtype=int)
            ibtype = np.zeros(nbou, dtype=int)
            nbvv = np.zeros((nvel, nbou), dtype=int)

            for i in range(nbou):
                line = fid.readline().strip()
                varg = list(map(int, line.split()[:2]))
                nvell[i], ibtype[i] = varg

                if ibtype[i] in [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 60, 61, 101, 52]:
                    nbvv[:nvell[i], i] = np.loadtxt(fid, max_rows=nvell[i], dtype=int)

            boudat = {
                "nbou": nbou,
                "nvel": nvel,
                "nvell": nvell,
                "ibtype": ibtype,
                "nbvv": nbvv[:np.max(nvell), :],
            }

    return EToV, VX, B, opedat, boudat, title
