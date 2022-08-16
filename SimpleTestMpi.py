import lightweaver as lw
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from lightweaver.fal import Falc82
from lightweaver.rh_atoms import (
    Al_atom,
    C_atom,
    CaII_atom,
    Fe_atom,
    H_6_atom,
    He_9_atom,
    MgII_atom,
    N_atom,
    Na_atom,
    Si_atom,
    O_atom,
    S_atom,
)

from MpiStackedContext import MpiStackedContext

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
Nchunks = comm.Get_size()

atmos = Falc82()
atmos.quadrature(5)
aSet = lw.RadiativeSet(
    [
        H_6_atom(),
        C_atom(),
        O_atom(),
        Si_atom(),
        Al_atom(),
        CaII_atom(),
        Fe_atom(),
        He_9_atom(),
        MgII_atom(),
        N_atom(),
        Na_atom(),
        S_atom(),
    ]
)
aSet.set_active("H", "Ca")

stacked = MpiStackedContext(
    atmos,
    aSet,
    Nchunks,
    comm,
    ctx_kwargs={
        "Nthreads": 1,
        "formalSolver": "piecewise_besser_1d",
    },
    conserveCharge=True,
)
# for i in range(4):
#     a = stacked.sub_ctx[i].atmos
#     plt.plot(a.z, a.temperature, '+--' if i % 2 == 0 else 'x-.')

lw.iterate_ctx_se(stacked, prd=True, quiet=(rank != 0))

# dJ = 1.0
# while dJ > 1e-8:
#     dJ = stacked.formal_sol_gamma_matrices().dJMax

if rank == 0:
    spect = aSet.compute_wavelength_grid()
    eq_pops = aSet.compute_eq_pops(atmos)
    ctx = lw.Context(
        atmos,
        spect,
        eq_pops,
        Nthreads=4,
        formalSolver="piecewise_besser_1d",
        conserveCharge=True,
    )
    # ctx.depthData.fill = True
    lw.iterate_ctx_se(ctx, prd=True)
    plt.plot(ctx.spect.wavelength, stacked.spect.I[:, -1])
    plt.plot(ctx.spect.wavelength, ctx.spect.I[:, -1], "--")
    plt.show()

# # active_elems = [a.element for a in aSet.activeAtoms]
# # nlte_pops = stacked.gather_nlte_pops()
# # for elem in active_elems:
# #     eq_pops[elem][...] = nlte_pops[elem]
# dJ = 1.0
# while dJ > 1e-8:
#     dJ = ctx.formal_sol_gamma_matrices().dJMax
