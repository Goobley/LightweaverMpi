import os

import lightweaver as lw
import matplotlib.pyplot as plt
import numpy as np
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
    O_atom,
    S_atom,
    Si_atom,
)
from mpi4py import MPI

from MpiSerde import (
    IntensityVisitor,
    PickleLwVisitorSerializer,
    WavelengthVisitor,
    ZarrLwVisitorSerializer,
    make_pops_visitors,
    make_prd_visitors,
)
from MpiStackedContext import MpiStackedContext

atmos1d = Falc82()
xAxis = np.linspace(0, 100e3, 11)
temperature = np.zeros((atmos1d.z.shape[0], xAxis.shape[0]))
vx = np.zeros((atmos1d.z.shape[0], xAxis.shape[0]))
vz = np.zeros((atmos1d.z.shape[0], xAxis.shape[0]))
vturb = np.zeros((atmos1d.z.shape[0], xAxis.shape[0]))
ne = np.zeros((atmos1d.z.shape[0], xAxis.shape[0]))
nHTot = np.zeros((atmos1d.z.shape[0], xAxis.shape[0]))
temperature[...] = atmos1d.temperature[:, None]
vz[...] = atmos1d.vlos[:, None]
vturb[...] = atmos1d.vturb[:, None]
ne[...] = atmos1d.ne[:, None]
nHTot[...] = atmos1d.nHTot[:, None]

atmos = lw.Atmosphere.make_2d(
    height=atmos1d.z,
    x=xAxis,
    temperature=temperature,
    vx=vx,
    vz=vz,
    vturb=vturb,
    ne=ne,
    nHTot=nHTot,
)
atmos.quadrature(7)

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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = comm.Get_size()
Nthreads = os.cpu_count()


stacked = MpiStackedContext(
    atmos, aSet, num_nodes, comm, ctx_kwargs={"Nthreads": Nthreads}
)


visitors = {
    **make_pops_visitors(stacked),
    **make_prd_visitors(stacked),
    "wavelength": WavelengthVisitor(),
    "I": IntensityVisitor(),
}
root_only = ["wavelength", "I"]
ser = ZarrLwVisitorSerializer(visitors)

ser.load(stacked, "Mpi2d.zarr", ignore=root_only)

dJ = 1.0
while dJ > 1e-4:
    dJ = stacked.formal_sol_gamma_matrices().dJMax

if rank == 0:
    plt.ion()
    plt.plot(stacked.spect.wavelength, stacked.spect.I[:, -1, 5])

    import zarr

    d = zarr.convenience.open("Mpi2d.zarr", "r")

    plt.plot(d["wavelength"], d["I"][:, -1, 5], "--")
    plt.show()
