import os
import sys
import json
import allo
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import md

lj1 = 1.5
lj2 = 2.0
domainEdge = 20.0


def md_force_ref(n_points, pos_x, pos_y, pos_z, blockSide):
    """Python reference for MD grid force computation."""
    force_x = np.zeros_like(pos_x)
    force_y = np.zeros_like(pos_y)
    force_z = np.zeros_like(pos_z)

    for b0x in range(blockSide):
        for b0y in range(blockSide):
            for b0z in range(blockSide):
                for b1x in range(max(0, b0x-1), min(blockSide, b0x+2)):
                    for b1y in range(max(0, b0y-1), min(blockSide, b0y+2)):
                        for b1z in range(max(0, b0z-1), min(blockSide, b0z+2)):
                            q_range = n_points[b1x, b1y, b1z]
                            for p_idx in range(n_points[b0x, b0y, b0z]):
                                px = pos_x[b0x, b0y, b0z, p_idx]
                                py = pos_y[b0x, b0y, b0z, p_idx]
                                pz = pos_z[b0x, b0y, b0z, p_idx]
                                sx, sy, sz = 0.0, 0.0, 0.0
                                for q_idx in range(q_range):
                                    qx = pos_x[b1x, b1y, b1z, q_idx]
                                    qy = pos_y[b1x, b1y, b1z, q_idx]
                                    qz = pos_z[b1x, b1y, b1z, q_idx]
                                    if qx != px or qy != py or qz != pz:
                                        dx = px - qx
                                        dy = py - qy
                                        dz = pz - qz
                                        r2inv = 1.0 / (dx*dx + dy*dy + dz*dz)
                                        r6inv = r2inv * r2inv * r2inv
                                        potential = r6inv * (lj1 * r6inv - lj2)
                                        f = r2inv * potential
                                        sx += f * dx
                                        sy += f * dy
                                        sz += f * dz
                                force_x[b0x, b0y, b0z, p_idx] += sx
                                force_y[b0x, b0y, b0z, p_idx] += sy
                                force_z[b0x, b0y, b0z, p_idx] += sz

    return force_x, force_y, force_z


def test_md_grid(psize="small"):
    setting_path = os.path.join(os.path.dirname(__file__), "..", "..", "psize.json")
    with open(setting_path, "r") as fp:
        sizes = json.load(fp)
    params = sizes["md_grid"][psize]

    blockSide = params["blockSide"]
    densityFactor = params["densityFactor"]
    blockEdge = domainEdge / blockSide

    # Patch md module constants
    md.blockSide = blockSide
    md.densityFactor = densityFactor
    md.domainEdge = domainEdge
    md.blockEdge = blockEdge
    md.nBlocks = blockSide * blockSide * blockSide

    np.random.seed(42)

    # Generate random atom positions within grid blocks
    np_n_points = np.full((blockSide, blockSide, blockSide), densityFactor, dtype=np.int32)
    np_pos_x = np.zeros((blockSide, blockSide, blockSide, densityFactor), dtype=np.float64)
    np_pos_y = np.zeros((blockSide, blockSide, blockSide, densityFactor), dtype=np.float64)
    np_pos_z = np.zeros((blockSide, blockSide, blockSide, densityFactor), dtype=np.float64)

    for bx in range(blockSide):
        for by in range(blockSide):
            for bz in range(blockSide):
                for a in range(densityFactor):
                    np_pos_x[bx, by, bz, a] = bx * blockEdge + np.random.rand() * blockEdge
                    np_pos_y[bx, by, bz, a] = by * blockEdge + np.random.rand() * blockEdge
                    np_pos_z[bx, by, bz, a] = bz * blockEdge + np.random.rand() * blockEdge

    s_x = allo.customize(md.md_x)
    mod_x = s_x.build()
    s_y = allo.customize(md.md_y)
    mod_y = s_y.build()
    s_z = allo.customize(md.md_z)
    mod_z = s_z.build()

    forceX = mod_x(np_n_points, np_pos_x, np_pos_y, np_pos_z)
    forceY = mod_y(np_n_points, np_pos_x, np_pos_y, np_pos_z)
    forceZ = mod_z(np_n_points, np_pos_x, np_pos_y, np_pos_z)

    check_x, check_y, check_z = md_force_ref(np_n_points, np_pos_x, np_pos_y, np_pos_z, blockSide)

    np.testing.assert_allclose(forceX, check_x, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forceY, check_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(forceZ, check_z, rtol=1e-5, atol=1e-5)
    print("PASS!")


if __name__ == "__main__":
    test_md_grid("full")
