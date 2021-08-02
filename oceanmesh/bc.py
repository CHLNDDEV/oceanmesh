import numpy as np
import scipy.spatial

from . import edges

__all__ = ["label_boundaries"]


def label_boundaries(shoreline, vertices, faces, classifier="distance"):
    """Labels the boundary edges with unique tags"""
    etbv = edges.get_boundary_edges(faces)
    x = vertices[:, 0]
    y = vertices[:, 1]
    x_mid = np.mean(x[etbv], [], 1)
    y_mid = np.mean(y[etbv], [], 1)
    eb_mid = np.array([x_mid, y_mid])

    cut_lim = 10
    dist_lim = 10 * shoreline.dx

    if classifier == "distance":
        if len(shoreline.mainland) > 0:
            land = shoreline.mainland
            land = land[~np.isnan(land[:, 0]), :]
        if len(shoreline.inner) > 0:
            inner = shoreline.inner
            inner = inner[~np.isnan(inner[:, 0]), :]
            land = np.append([land, inner])
        if len(land) > 0:
            land_tree = scipy.spatial.cKDTree(land, balanced_tree=False)
            ldst, _ = land_tree.query(eb_mid, k=1)
        else:
            # set distance to be larger than dist_lim
            # everywhere when no land exists
            ldst = eb_mid[:, 1] * 0 + 2 * dist_lim

        eb_class = ldst > dist_lim

        polys = edges.get_winded_boundary_loops(faces)
        polys = edges._convert_to_list(polys)

        set_etbv = set([tuple(x) for x in etbv])
        set_etbv_flr = set([tuple(x) for x in np.fliplr(etbv)])

        for poly in polys:

            eb_class_t = 0 * poly[:, 0]
            set_poly_ed = set([tuple(x) for x in poly])

            # make sure that the order of polygon and
            # the order of the edges in etbv are the same
            ed_id1 = [ind for ind, i in enumerate(set_etbv) if i in set_poly_ed]
            poly_id1 = [ind for ind, i in enumerate(set_poly_ed) if i in set_etbv]
            eb_class_t1 = eb_class[ed_id1]
            eb_class_t[poly_id1] = eb_class_t1

            ed_id2 = [ind for ind, i in enumerate(set_etbv_flr) if i in set_poly_ed]
            poly_id2 = [ind for ind, i in enumerate(set_poly_ed) if i in set_etbv_flr]
            eb_class_t2 = eb_class[ed_id2]
            eb_class_t[poly_id2] = eb_class_t2

            eb_sum = sum(eb_class_t)

            if eb_sum > 0:
                # Consists of both ocean and mainland
                # If a boundary segment is composed of both
                # ocean and mainland, then the whole segment is
                # 'cut up' into segments. Special care is
                # taken to ensure the ocean
                # type boundary starts following then by mainland
                # etc.
                cuts = np.where(np.diff(eb_class_t) != 0)[0]

                # Do not include open boundary that is
                # smaller than cutlim vertices across
                rm = np.zeros((len(cuts), 1), dtype=bool)
                st = 0
                if eb_class_t[1]:
                    st = 1
                for ii in range(st, len(cuts), 2):
                    if ii == len(cuts):
                        if len(idv) - cuts[ii] + cuts[0] - 1 < cut_lim:
                            rm[0] = 1
                            rm[-1] = 1
                        else:
                            if cuts[ii + 1] - cuts[ii] < cut_lim:
                                rm[ii : ii + 1] = 1
                cuts[rm] = []
            else:
                # then it's an island
