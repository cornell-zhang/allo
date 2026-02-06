# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float64, int32

nAtoms: int32 = 256
maxNeighbors: int32 = 16
lj1: float64 = 1.5
lj2: float64 = 2.0
domainEdge: float64 = 20.0


def md_x(
    position_x: float64[nAtoms],
    position_y: float64[nAtoms],
    position_z: float64[nAtoms],
    NL: int32[nAtoms * maxNeighbors],
) -> float64[nAtoms]:
    # -> (float64[nAtoms],float64[nAtoms],float64[nAtoms]):

    i_x: float64 = 0.0
    i_y: float64 = 0.0
    i_z: float64 = 0.0
    jidx: int32 = 0
    j_x: float64 = 0.0
    j_y: float64 = 0.0
    j_z: float64 = 0.0
    delx: float64 = 0.0
    dely: float64 = 0.0
    delz: float64 = 0.0
    r2inv: float64 = 0.0
    r6inv: float64 = 0.0
    potential: float64 = 0.0
    force: float64 = 0.0
    fx: float64 = 0.0
    force_x: float64[nAtoms] = 0.0

    for i in range(nAtoms):
        i_x = position_x[i]
        i_y = position_y[i]
        i_z = position_z[i]
        fx = 0.0
        # fy:float64=0.0
        # fz:float64=0.0

        for j in range(maxNeighbors):
            # Get neighbor
            jidx = NL[i * maxNeighbors + j]
            # Look up x, y, z positions
            j_x = position_x[jidx]
            j_y = position_y[jidx]
            j_z = position_z[jidx]
            # Calculate distance
            delx = i_x - j_x
            dely = i_y - j_y
            delz = i_z - j_z
            if (delx * delx + dely * dely + delz * delz) == 0:
                r2inv = (domainEdge * domainEdge * 3.0) * 1000
            else:
                r2inv = 1.0 / (delx * delx + dely * dely + delz * delz)
            # Assume no cutoff and always account for all nodes in the area
            r6inv = r2inv * r2inv * r2inv
            potential = r6inv * (lj1 * r6inv - lj2)
            # Sum changes in force
            force = r2inv * potential
            fx = fx + delx * force
        # Update forces after all neighbors are accounted for
        force_x[i] = fx
        # force_y[i] = fy
        # force_z[i] = fz
    return force_x
    # print(f"dF={fx},{fy},{fz}")


def md_y(
    position_x: float64[nAtoms],
    position_y: float64[nAtoms],
    position_z: float64[nAtoms],
    NL: int32[nAtoms * maxNeighbors],
) -> float64[nAtoms]:
    # -> (float64[nAtoms],float64[nAtoms],float64[nAtoms]):
    i_x: float64 = 0.0
    i_y: float64 = 0.0
    i_z: float64 = 0.0
    jidx: int32 = 0
    j_x: float64 = 0.0
    j_y: float64 = 0.0
    j_z: float64 = 0.0
    delx: float64 = 0.0
    dely: float64 = 0.0
    delz: float64 = 0.0
    r2inv: float64 = 0.0
    r6inv: float64 = 0.0
    potential: float64 = 0.0
    force: float64 = 0.0
    fy: float64 = 0.0
    force_y: float64[nAtoms]

    for i in range(nAtoms):
        i_x = position_x[i]
        i_y = position_y[i]
        i_z = position_z[i]
        fy = 0.0
        # fy:float64=0.0
        # fz:float64=0.0

        for j in range(maxNeighbors):
            # Get neighbor
            jidx = NL[i * maxNeighbors + j]
            # Look up x, y, z positions
            j_x = position_x[jidx]
            j_y = position_y[jidx]
            j_z = position_z[jidx]
            # Calculate distance
            delx = i_x - j_x
            dely = i_y - j_y
            delz = i_z - j_z
            if (delx * delx + dely * dely + delz * delz) == 0:
                r2inv = (domainEdge * domainEdge * 3.0) * 1000
            else:
                r2inv = 1.0 / (delx * delx + dely * dely + delz * delz)
            # Assume no cutoff and always account for all nodes in the area
            r6inv = r2inv * r2inv * r2inv
            potential = r6inv * (lj1 * r6inv - lj2)
            # Sum changes in force
            force = r2inv * potential
            fy = fy + dely * force
        # Update forces after all neighbors are accounted for
        force_y[i] = fy
        # force_y[i] = fy
        # force_z[i] = fz
    return force_y
    # print(f"dF={fx},{fy},{fz}")


def md_z(
    position_x: float64[nAtoms],
    position_y: float64[nAtoms],
    position_z: float64[nAtoms],
    NL: int32[nAtoms * maxNeighbors],
) -> float64[nAtoms]:
    # -> (float64[nAtoms],float64[nAtoms],float64[nAtoms]):
    i_x: float64 = 0.0
    i_y: float64 = 0.0
    i_z: float64 = 0.0
    jidx: int32 = 0
    j_x: float64 = 0.0
    j_y: float64 = 0.0
    j_z: float64 = 0.0
    delx: float64 = 0.0
    dely: float64 = 0.0
    delz: float64 = 0.0
    r2inv: float64 = 0.0
    r6inv: float64 = 0.0
    potential: float64 = 0.0
    force: float64 = 0.0
    fz: float64 = 0.0
    force_z: float64[nAtoms]

    for i in range(nAtoms):
        i_x = position_x[i]
        i_y = position_y[i]
        i_z = position_z[i]
        fz = 0.0
        # fy:float64=0.0
        # fz:float64=0.0

        for j in range(maxNeighbors):
            # Get neighbor
            jidx = NL[i * maxNeighbors + j]
            # Look up x, y, z positions
            j_x = position_x[jidx]
            j_y = position_y[jidx]
            j_z = position_z[jidx]
            # Calculate distance
            delx = i_x - j_x
            dely = i_y - j_y
            delz = i_z - j_z
            if (delx * delx + dely * dely + delz * delz) == 0:
                r2inv = (domainEdge * domainEdge * 3.0) * 1000
            else:
                r2inv = 1.0 / (delx * delx + dely * dely + delz * delz)
            # Assume no cutoff and always account for all nodes in the area
            r6inv = r2inv * r2inv * r2inv
            potential = r6inv * (lj1 * r6inv - lj2)
            # Sum changes in force
            force = r2inv * potential
            fz = fz + delz * force
        # Update forces after all neighbors are accounted for
        force_z[i] = fz
        # force_y[i] = fy
        # force_z[i] = fz
    return force_z
    # print(f"dF={fx},{fy},{fz}")


if __name__ == "__main__":
    s_x = allo.customize(md_x)
    print(s_x.module)
    s_x.build()

    s_y = allo.customize(md_y)
    print(s_y.module)
    s_y.build()

    s_z = allo.customize(md_z)
    print(s_z.module)
    s_z.build()

    print("build success")
