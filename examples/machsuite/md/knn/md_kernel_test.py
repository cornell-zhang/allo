import md
import allo
import numpy as np
import os

nAtoms = 256
maxNeighbors = 16
domainEdge = 20.0
lj1 = 1.5
lj2 = 2.0


def md_knn_force_ref(pos_x, pos_y, pos_z, NL):
    """Python reference for MD KNN force computation."""
    force_x = np.zeros(nAtoms, dtype=np.float64)
    force_y = np.zeros(nAtoms, dtype=np.float64)
    force_z = np.zeros(nAtoms, dtype=np.float64)

    for i in range(nAtoms):
        ix, iy, iz = pos_x[i], pos_y[i], pos_z[i]
        fx, fy, fz = 0.0, 0.0, 0.0
        for j in range(maxNeighbors):
            jidx = NL[i * maxNeighbors + j]
            jx, jy, jz = pos_x[jidx], pos_y[jidx], pos_z[jidx]
            dx = ix - jx
            dy = iy - jy
            dz = iz - jz
            r2 = dx*dx + dy*dy + dz*dz
            if r2 == 0:
                r2inv = (domainEdge * domainEdge * 3.0) * 1000
            else:
                r2inv = 1.0 / r2
            r6inv = r2inv * r2inv * r2inv
            potential = r6inv * (lj1 * r6inv - lj2)
            force = r2inv * potential
            fx += dx * force
            fy += dy * force
            fz += dz * force
        force_x[i] = fx
        force_y[i] = fy
        force_z[i] = fz

    return force_x, force_y, force_z


if __name__ == "__main__":
    np.random.seed(42)

    # Generate random atom positions
    np_pos_x = np.random.rand(nAtoms).astype(np.float64) * domainEdge
    np_pos_y = np.random.rand(nAtoms).astype(np.float64) * domainEdge
    np_pos_z = np.random.rand(nAtoms).astype(np.float64) * domainEdge

    # Generate neighbor list: for each atom, pick maxNeighbors distinct neighbors
    np_NL = np.zeros(nAtoms * maxNeighbors, dtype=np.int32)
    for i in range(nAtoms):
        others = list(range(nAtoms))
        others.remove(i)
        neighbors = np.random.choice(others, size=maxNeighbors, replace=False)
        for j in range(maxNeighbors):
            np_NL[i * maxNeighbors + j] = neighbors[j]

    s_x = allo.customize(md.md_x)
    mod_x = s_x.build()
    s_y = allo.customize(md.md_y)
    mod_y = s_y.build()
    s_z = allo.customize(md.md_z)
    mod_z = s_z.build()

    forceX = mod_x(np_pos_x, np_pos_y, np_pos_z, np_NL)
    forceY = mod_y(np_pos_x, np_pos_y, np_pos_z, np_NL)
    forceZ = mod_z(np_pos_x, np_pos_y, np_pos_z, np_NL)

    check_x, check_y, check_z = md_knn_force_ref(np_pos_x, np_pos_y, np_pos_z, np_NL)

    np.testing.assert_allclose(forceX, check_x, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forceY, check_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forceZ, check_z, rtol=1e-5, atol=1e-5)
    print("PASS!")
