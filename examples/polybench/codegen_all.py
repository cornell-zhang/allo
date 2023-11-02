from cedar.verify import codegen_pair

from two_mm import top_2mm
from three_mm import top_3mm
from adi import top_adi
from atax import top_atax
from bicg import top_bicg
from cholesky import top_cholesky
from correlation import top_correlation
from covariance import top_covariance
from deriche import top_deriche
from doitgen import top_doitgen
from durbin import top_durbin
from fdtd_2d import top_fdtd_2d
from floyd_warshall import top_floyd_warshall
from gemm import top_gemm
from gemver import top_gemver
from gesummv import top_gesummv
from gramschmidt import top_gramschmidt
from heat_3d import top_heat_3d
from jacobi_1d import top_jacobi_1d
from jacobi_2d import top_jacobi_2d
from lu import top_lu
from ludcmp import top_ludcmp
from mvt import top_mvt
from nussinov import top_nussinov
from seidel_2d import top_seidel_2d
from symm import top_symm
from syr2k import top_syr2k
from syrk import top_syrk
from trisolv import top_trisolv
from trmm import top_trmm


def codegen_all(size, codegen_dir, log_dir):
    cmd = "mkdir -p " + log_dir + "\n"

    orig, opt = top_2mm(size=size)
    cmd += codegen_pair(
        orig, opt, "2mm", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v4"
    )

    orig, opt = top_3mm(size=size)
    cmd += codegen_pair(
        orig, opt, "3mm", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v4"
    )

    orig, opt = top_adi(size=size)
    cmd += codegen_pair(
        orig, opt, "adi", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v3"
    )

    orig, opt = top_atax(size=size)
    cmd += codegen_pair(
        orig, opt, "atax", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v2"
    )

    orig, opt = top_bicg(size=size)
    cmd += codegen_pair(
        orig, opt, "bicg", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v4"
    )

    orig, opt = top_cholesky(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "cholesky",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v0",
    )

    orig, opt = top_correlation(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "correlation",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v3",
    )

    orig, opt = top_covariance(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "covariance",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v2",
    )

    orig, opt = top_deriche(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "deriche",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v1",
    )

    orig, opt = top_doitgen(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "doitgen",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v1",
    )

    orig, opt = top_durbin(size=size)
    cmd += codegen_pair(
        orig, opt, "durbin", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v1"
    )

    orig, opt = top_fdtd_2d(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "fdtd_2d",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v0,v1,v2",
    )

    orig, opt = top_floyd_warshall(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "floyd_warshall",
        options="--symbolic-conditionals",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v0",
    )

    orig, opt = top_gemm(size=size)
    cmd += codegen_pair(
        orig, opt, "gemm", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v2"
    )

    orig, opt = top_gemver(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "gemver",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v0,v5,v7",
    )

    orig, opt = top_gesummv(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "gesummv",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v3",
    )

    orig, opt = top_gramschmidt(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "gramschmidt",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v0,v1,v2",
    )

    orig, opt = top_heat_3d(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "heat_3d",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v0,v1",
    )

    orig, opt = top_jacobi_1d(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "jacobi_1d",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v0,v1",
    )

    orig, opt = top_jacobi_2d(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "jacobi_2d",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v0,v1",
    )

    orig, opt = top_lu(size=size)
    cmd += codegen_pair(
        orig, opt, "lu", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v0"
    )

    orig, opt = top_ludcmp(size=size)
    cmd += codegen_pair(
        orig, opt, "ludcmp", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v2"
    )

    orig, opt = top_mvt(size=size)
    cmd += codegen_pair(
        orig, opt, "mvt", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v3,v4"
    )

    orig, opt = top_nussinov(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "nussinov",
        options="--symbolic-conditionals",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v1",
    )

    orig, opt = top_seidel_2d(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "seidel_2d",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v0",
    )

    orig, opt = top_symm(size=size)
    cmd += codegen_pair(
        orig, opt, "symm", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v2"
    )

    orig, opt = top_syr2k(size=size)
    cmd += codegen_pair(
        orig, opt, "syr2k", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v2"
    )

    orig, opt = top_syrk(size=size)
    cmd += codegen_pair(
        orig, opt, "syrk", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v1"
    )

    orig, opt = top_trisolv(size=size)
    cmd += codegen_pair(
        orig,
        opt,
        "trisolv",
        codegen_dir=codegen_dir,
        log_dir=log_dir,
        liveout_vars="v2",
    )

    orig, opt = top_trmm(size=size)
    cmd += codegen_pair(
        orig, opt, "trmm", codegen_dir=codegen_dir, log_dir=log_dir, liveout_vars="v1"
    )

    return cmd


if __name__ == "__main__":
    cmd = codegen_all(size=None, codegen_dir="codegen_mini", log_dir="log_mini")
    with open("verify_mini.sh", "w") as f:
        f.write(cmd)
    print("generated verify_mini.sh")
    cmd = codegen_all(size="medium", codegen_dir="codegen_medium", log_dir="log_medium")
    with open("verify_medium.sh", "w") as f:
        f.write(cmd)
    print("generated verify_medium.sh")
