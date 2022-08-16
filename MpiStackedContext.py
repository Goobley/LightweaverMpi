import pickle
from enum import IntEnum
from typing import List, Optional

import lightweaver as lw
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Intracomm


class FixedBc(lw.BoundaryCondition):
    def __init__(self):
        super().__init__()
        self.I = np.array(())

    def compute_bc(self, atmos: "lw.Atmosphere", spect: "lw.LwSpectrum") -> np.ndarray:
        return self.I

    def set_bc(self, I: np.ndarray):
        if I.ndim == 2:
            self.I = I[:, :, None]
        else:
            self.I = I


def reduce_iteration_updates(updates: List[lw.IterationUpdate]):
    """Reduce the iteration updates from all nodes, to be rebroadcast"""
    # TODO(cmo): Handle offsets on MaxIdxs due to different ctxs
    u = updates[0]
    kwargs = {}
    kwargs["crsw"] = u.crsw

    dJMaxs = [x.dJMax for x in updates]
    dJMaxIdxs = [x.dJMaxIdx for x in updates]
    kwargs["updatedJ"] = u.updatedJ
    kwargs["dJMax"] = max(dJMaxs)
    kwargs["dJMaxIdx"] = dJMaxIdxs[dJMaxs.index(kwargs["dJMax"])]

    kwargs["updatedPops"] = u.updatedPops
    kwargs["ngAccelerated"] = u.ngAccelerated
    kwargs["dPops"] = []
    kwargs["dPopsMaxIdx"] = []
    for i in range(len(u.dPops)):
        dPops = [x.dPops[i] for x in updates]
        dPopsIdx = [x.dPopsMaxIdx[i] for x in updates]
        kwargs["dPops"].append(max(dPops))
        kwargs["dPopsMaxIdx"].append(dPopsIdx[dPops.index(kwargs["dPops"][-1])])

    dNe = [x.dNeMax for x in updates]
    dNeIdx = [x.dNeMaxIdx for x in updates]
    kwargs["updatedNe"] = u.updatedNe
    kwargs["dNeMax"] = max(dNe)
    kwargs["dNeMaxIdx"] = dNeIdx[dNe.index(kwargs["dNeMax"])]

    kwargs["updatedRho"] = u.updatedRho
    Nprd = [x.NprdSubIter for x in updates]
    kwargs["NprdSubIter"] = max(Nprd)
    kwargs["dRho"] = []
    kwargs["dRhoMaxIdx"] = []
    for i in range(len(u.dRho)):
        dRho = [x.dRho[i] for x in updates]
        dRhoIdx = [x.dRhoMaxIdx[i] for x in updates]
        kwargs["dRho"].append(max(dRho))
        kwargs["dRhoMaxIdx"].append(dRhoIdx[dRho.index(kwargs["dRho"][-1])])

    kwargs["updatedJPrd"] = u.updatedJPrd
    kwargs["dJPrdMax"] = []
    kwargs["dJPrdMaxIdx"] = []
    for i in range(len(u.dJPrdMax)):
        dJPrd = [x.dJPrdMax[i] for x in updates]
        dJPrdIdx = [x.dJPrdMaxIdx[i] for x in updates]
        kwargs["dJPrdMax"].append(max(dJPrd))
        kwargs["dJPrdMaxIdx"].append(dJPrdIdx[dJPrd.index(kwargs["dJPrdMax"][-1])])

    return lw.IterationUpdate(u.ctx, **kwargs)


def slice_atmos_1d(atmos: lw.Atmosphere, start_idx: int, end_idx: int, **kwargs):
    """Chunk a 1d atmosphere"""
    # NOTE(cmo): This ignores the polarisation variables.
    return lw.Atmosphere.make_1d(
        scale=lw.ScaleType.Geometric,
        depthScale=atmos.height[start_idx:end_idx],
        temperature=atmos.temperature[start_idx:end_idx],
        vlos=atmos.vlos[start_idx:end_idx],
        vturb=atmos.vturb[start_idx:end_idx],
        ne=np.copy(
            atmos.ne[start_idx:end_idx]
        ),  # copy to avoid issues with overlap during nr_post_iter
        nHTot=atmos.nHTot[start_idx:end_idx],
        **kwargs,
    )


def slice_atmos_2d(atmos: lw.Atmosphere, start_idx: int, end_idx: int, **kwargs):
    """Chunk a 2D atmosphere"""
    # NOTE(cmo): This ignores the polarisation variables.
    atmos = atmos.dimensioned_view()
    return lw.Atmosphere.make_2d(
        height=np.ascontiguousarray(atmos.height[start_idx:end_idx]),
        x=atmos.x,
        temperature=np.ascontiguousarray(atmos.temperature[start_idx:end_idx]),
        vx=np.ascontiguousarray(atmos.vx[start_idx:end_idx]),
        vz=np.ascontiguousarray(atmos.vz[start_idx:end_idx]),
        vturb=np.ascontiguousarray(atmos.vturb[start_idx:end_idx]),
        ne=np.copy(np.ascontiguousarray(atmos.ne[start_idx:end_idx])),
        nHTot=np.ascontiguousarray(atmos.nHTot[start_idx:end_idx]),
        **kwargs,
    )


def slice_atmos(atmos: lw.Atmosphere, start_idx: int, end_idx: int, **kwargs):
    """Extract correct chunk of atmosphere"""
    if atmos.Ndim == 1:
        return slice_atmos_1d(atmos, start_idx, end_idx, **kwargs)
    return slice_atmos_2d(atmos, start_idx, end_idx, **kwargs)


class LwMpiTags(IntEnum):
    # NOTE(cmo): These are from the sender's perspective
    UpperBc = 10
    LowerBc = 11

    LineStep = 31
    ElementStep = 100
    ElementStart = 1000  # NOTE(cmo): Elements go up in steps of 100, lines in 31
    NeTag = 9000
    RhoPrdTag = 8000


class MpiStackedContext(lw.Context):
    """Manages the MPI distributed simulation"""

    def __init__(
        self,
        atmos: lw.Atmosphere,
        aSet: lw.RadiativeSet,
        Nchunks: int,
        comm: Intracomm,
        conserveCharge: bool = False,
        ctx_kwargs=None,
        quiet=True,
    ):

        self.global_atmos = atmos.dimensioned_view()
        self.Nz = atmos.Nz
        self.aSet = aSet
        self.Nchunks = Nchunks
        self.conserveCharge = conserveCharge

        if Nchunks < 2:
            raise ValueError("Need at least 2 chunks")

        if ctx_kwargs is None:
            ctx_kwargs = {}
        ctx_kwargs["conserveCharge"] = conserveCharge
        self.comm = comm
        self.rank = self.comm.Get_rank()

        if self.Nchunks != self.comm.Get_size():
            raise ValueError(
                "Expected Nchunks and number of MPI processes to be the same."
            )

        ctx_size = int(atmos.Nz // Nchunks)
        ctx_sizes = [ctx_size] * Nchunks
        self.ctx_sizes = ctx_sizes
        i = 0
        while sum(ctx_sizes) != atmos.Nz:
            ctx_sizes[i] += 1
            i += 1

        # NOTE(cmo): We use one ghost cell on the +/- side of each Z domain.
        start_idx = 0
        for i, ctx_size in enumerate(ctx_sizes):
            size_with_bcs = ctx_size
            if i != 0:
                size_with_bcs += 1
            if i != self.Nchunks - 1:
                size_with_bcs += 1

            end_idx = start_idx + size_with_bcs
            end_idx = min(end_idx, atmos.Nz)

            if i == self.rank:
                break

            start_idx += ctx_size
            if i == 0:
                start_idx -= 1

        if not quiet:
            print(
                f"Node {self.rank}: z_range: ({start_idx}, {end_idx}) z_length: {end_idx - start_idx}"
            )
        upperBc = None if start_idx == 0 else FixedBc()
        lowerBc = None if end_idx == atmos.Nz else FixedBc()
        self.local_bcs = [x for x in [upperBc, lowerBc] if x is not None]

        self.cstart = 1
        self.cend: Optional[int] = -1
        if self.rank == 0:
            self.cstart = 0
        elif self.rank == self.Nchunks - 1:
            self.cend = None

        if atmos.Ndim == 1:
            self.sub_atmos = slice_atmos(
                atmos, start_idx, end_idx, lowerBc=lowerBc, upperBc=upperBc
            )
        else:
            self.sub_atmos = slice_atmos(
                atmos, start_idx, end_idx, zLowerBc=lowerBc, zUpperBc=upperBc
            )

        if atmos.Ndim == 1:
            self.sub_atmos.quadrature(atmos.Nrays)
        else:
            self.sub_atmos.quadrature(int(atmos.Nrays // 2))

        self.sub_spect = aSet.compute_wavelength_grid()
        self.sub_eq_pops = aSet.compute_eq_pops(self.sub_atmos)
        # HACK(cmo): Initialising the Context resets __dict__, so preserve it
        # and add the attrs back.
        prev_dict = self.__dict__
        super().__init__(self.sub_atmos, self.sub_spect, self.sub_eq_pops, **ctx_kwargs)
        for k, v in prev_dict.items():
            self.__dict__[k] = v

        self.zero_bcs()
        self.bc_storage = [self.new_z_bc_array() for _ in self.local_bcs]

    def zero_bcs(self):
        for bc in self.local_bcs:
            bc.set_bc(self.new_z_bc_array())

    def new_z_bc_array(self):
        Nwave = self.sub_spect.wavelength.shape[0]
        if self.atmos.Ndim == 1:
            return np.zeros((Nwave, self.atmos.Nrays))

        return np.zeros((Nwave, self.atmos.Nrays, self.atmos.Nx))

    def communicate_bcs(self):
        """Communicate the intensity boundary conditions between adjacent nodes"""
        reqs = []
        if self.rank == 0:
            lower_bc = self.bc_storage[0]
            reqs.append(
                self.comm.Isend(lower_bc, dest=self.rank + 1, tag=LwMpiTags.LowerBc)
            )

            lower_recv = self.local_bcs[0].I.squeeze()
            self.comm.Recv(lower_recv, source=self.rank + 1, tag=LwMpiTags.UpperBc)

            self.local_bcs[0].set_bc(lower_recv)
        elif self.rank == self.Nchunks - 1:
            upper_bc = self.bc_storage[0]
            reqs.append(
                self.comm.Isend(upper_bc, dest=self.rank - 1, tag=LwMpiTags.UpperBc)
            )

            upper_recv = self.local_bcs[0].I.squeeze()
            self.comm.Recv(upper_recv, source=self.rank - 1, tag=LwMpiTags.LowerBc)

            self.local_bcs[0].set_bc(upper_recv)
        else:
            upper_bc = self.bc_storage[0]
            lower_bc = self.bc_storage[1]
            reqs.append(
                self.comm.Isend(upper_bc, dest=self.rank - 1, tag=LwMpiTags.UpperBc)
            )
            reqs.append(
                self.comm.Isend(lower_bc, dest=self.rank + 1, tag=LwMpiTags.LowerBc)
            )

            upper_recv = self.local_bcs[0].I.squeeze()
            lower_recv = self.local_bcs[1].I.squeeze()
            self.comm.Recv(upper_recv, source=self.rank - 1, tag=LwMpiTags.LowerBc)
            self.comm.Recv(lower_recv, source=self.rank + 1, tag=LwMpiTags.UpperBc)

            self.local_bcs[0].set_bc(upper_recv)
            self.local_bcs[1].set_bc(lower_recv)

        MPI.Request.Waitall(reqs)

    def store_pops(self, nlte_elems):
        """Per node, store a copy of the local populations of `nlte_elems`"""
        return [np.copy(self.eqPops.dimensioned_view()[a]) for a in nlte_elems]

    def pops_change(self, prev_pops, nlte_elems) -> lw.IterationUpdate:
        """Per node, compute the relative change from the previous populations to the current ones"""
        pops_change = [
            np.abs(1.0 - prev_pops[j] / self.eqPops.dimensioned_view()[a])
            for j, a in enumerate(nlte_elems)
        ]
        pops_change = [p[:, self.cstart : self.cend] for p in pops_change]
        dPopsMaxIdx = [np.nanargmax(p) for p in pops_change]
        dPops = [p.reshape(-1)[idx] for p, idx in zip(pops_change, dPopsMaxIdx)]
        dPopsMaxIdx = [d + self.cstart * max(1, self.atmos.Nx) for d in dPopsMaxIdx]

        update = lw.IterationUpdate(
            self, updatedPops=True, dPops=dPops, dPopsMaxIdx=dPopsMaxIdx
        )
        return update

    def formal_sol_gamma_matrices(self) -> lw.IterationUpdate:
        """Specialisation of `formal_sol_gamma_matrices` handling the communication of the boundaries, and ignoring the ghost cells in dJ."""
        self.communicate_bcs()
        ctx = super()
        prev_J = np.copy(ctx.spect.J)

        extraParams = {"ZPlaneDecomposition": True}
        if self.rank == 0:
            extraParams["ZPlaneDown"] = self.bc_storage[0]
        elif self.rank == self.Nchunks - 1:
            extraParams["ZPlaneUp"] = self.bc_storage[0]
        else:
            extraParams["ZPlaneUp"] = self.bc_storage[0]
            extraParams["ZPlaneDown"] = self.bc_storage[1]

        _ = ctx.formal_sol_gamma_matrices(extraParams=extraParams)

        J_change = np.abs(1.0 - prev_J / ctx.spect.J)
        if self.atmos.Ndim == 2:
            J_change = J_change.reshape(J_change.shape[0], ctx.atmos.Nz, ctx.atmos.Nx)
        J_change = J_change[:, self.cstart : self.cend]
        dJMaxIdx = np.nanargmax(J_change)
        dJMax = J_change.reshape(-1)[dJMaxIdx]

        update = lw.IterationUpdate(ctx, updatedJ=True, dJMax=dJMax, dJMaxIdx=dJMaxIdx)
        update.ctx = None
        self.crswDone = update.crsw == 1.0

        updates = self.comm.gather(update, root=0)
        if self.rank == 0:
            update = reduce_iteration_updates(updates)
        update = self.comm.bcast(update, root=0)
        update.ctx = self

        return update

    def stat_equil(self) -> lw.IterationUpdate:
        """Specialised version of `stat_equil`, communicating the populations and electron density for the ghost points (if necessary), and ignoring the changes in these points."""
        active_elems = [a.element for a in self.aSet.activeAtoms]

        prev_pops = self.store_pops(active_elems)
        if self.conserveCharge:
            ne_prev = np.copy(self.sub_atmos.dimensioned_view().ne)
        _ = super().stat_equil()
        self.update_ghost_pops()
        if self.conserveCharge:
            self.update_ghost_ne()
        update = self.pops_change(prev_pops, active_elems)

        if self.conserveCharge:
            ne_change = np.abs(
                1.0
                - ne_prev[self.cstart : self.cend]
                / self.sub_atmos.dimensioned_view().ne[self.cstart : self.cend]
            )
            update.updatedNe = True
            update.dNeMaxIdx = np.nanargmax(ne_change)
            update.dNeMax = ne_change[update.dNeMaxIdx]
            update.dNeMaxIdx += self.cstart * max(1, self.atmos.Nx)

        update.ctx = None
        self.crswDone = update.crsw == 1.0

        updates = self.comm.gather(update, root=0)
        if self.rank == 0:
            update = reduce_iteration_updates(updates)
        update = self.comm.bcast(update, root=0)
        update.ctx = self

        if self.conserveCharge:
            self.sub_eq_pops.update_lte_atoms_Hmin_pops(
                self.sub_atmos, conserveCharge=False, quiet=True
            )

        return update

    def prd_redistribute(self, maxIter=3, tol=1e-2):
        """Specialised version of `prd_redistribute`. Has to manually manage the sub-iterations (rather than relying on the backend) due to the need to communicate the intensity boundaries between subiters."""
        dRho = []
        dRhoMaxIdx = []

        for iter in range(maxIter):
            prev_rho = []
            for ia, a in enumerate(self.activeAtoms):
                for it, t in enumerate(a.trans):
                    try:
                        prev_rho.append(np.copy(t.rhoPrd))
                    except AttributeError:
                        pass

            self.communicate_bcs()
            update = super().prd_redistribute(maxIter=1, tol=tol)

            inner_idx = 0
            for ia, a in enumerate(self.activeAtoms):
                for it, t in enumerate(a.trans):
                    try:
                        t.rhoPrd

                        if self.sub_atmos.Ndim == 1:
                            rho_new = t.rhoPrd
                            rho_prev = prev_rho[inner_idx]
                        else:
                            rho_new = t.rhoPrd.reshape(
                                -1, self.sub_atmos.Nz, self.sub_atmos.Nx
                            )
                            rho_prev = prev_rho[inner_idx].reshape(
                                -1, self.sub_atmos.Nz, self.sub_atmos.Nx
                            )
                        rho_change = np.abs(
                            1.0
                            - rho_prev[:, self.cstart : self.cend]
                            / rho_new[:, self.cstart : self.cend]
                        )
                        rho_max_idx = np.nanargmax(rho_change)
                        dRho.append(rho_change.reshape(-1)[rho_max_idx])
                        dRhoMaxIdx.append(rho_max_idx)
                        inner_idx += 1
                    except AttributeError:
                        pass

            update.dRho = dRho
            update.dRhoMaxIdx = dRhoMaxIdx
            update.NprdSubIter = iter + 1

            self.update_ghost_rho()

            update.ctx = None
            self.crswDone = update.crsw == 1.0

            updates = self.comm.gather(update, root=0)
            if self.rank == 0:
                update = reduce_iteration_updates(updates)
            update = self.comm.bcast(update, root=0)
            update.ctx = self

            if update.dRhoMax < tol:
                return update
        return update

    def update_ghost_pops(self):
        """Communicate the NLTE populations for the ghost points."""
        reqs = []
        active_elems = [a.element for a in self.aSet.activeAtoms]
        # NOTE(cmo): Communicate all using non-blocking (buffered) send
        for i, elem in enumerate(active_elems):
            if self.rank != 0:
                tag = LwMpiTags.ElementStart + LwMpiTags.ElementStep * i
                tag += LwMpiTags.UpperBc
                pops_source = np.ascontiguousarray(
                    self.sub_eq_pops.dimensioned_view()[elem][:, 1]
                )
                reqs.append(self.comm.Isend(pops_source, dest=self.rank - 1, tag=tag))

            if self.rank != self.Nchunks - 1:
                tag = LwMpiTags.ElementStart + LwMpiTags.ElementStep * i
                tag += LwMpiTags.LowerBc
                pops_source = np.ascontiguousarray(
                    self.sub_eq_pops.dimensioned_view()[elem][:, -2]
                )
                reqs.append(self.comm.Isend(pops_source, dest=self.rank + 1, tag=tag))

        # NOTE(cmo): Receive all
        for i, elem in enumerate(active_elems):
            if self.rank != 0:
                tag = LwMpiTags.ElementStart + LwMpiTags.ElementStep * i
                tag += LwMpiTags.LowerBc
                pops_ghost = self.sub_eq_pops.dimensioned_view()[elem][:, 0]
                pops_source = np.empty_like(pops_ghost)
                self.comm.Recv(pops_source, source=self.rank - 1, tag=tag)
                pops_ghost[...] = pops_source

            if self.rank != self.Nchunks - 1:
                tag = LwMpiTags.ElementStart + LwMpiTags.ElementStep * i
                tag += LwMpiTags.UpperBc
                pops_ghost = self.sub_eq_pops.dimensioned_view()[elem][:, -1]
                pops_source = np.empty_like(pops_ghost)
                self.comm.Recv(pops_source, source=self.rank + 1, tag=tag)
                pops_ghost[...] = pops_source

        MPI.Request.Waitall(reqs)

    def update_ghost_ne(self):
        """Communicate the electron density for the ghost points."""
        reqs = []
        # NOTE(cmo): Communicate all using non-blocking (buffered) send
        if self.rank != 0:
            tag = LwMpiTags.NeTag
            tag += LwMpiTags.UpperBc
            ne_source = np.ascontiguousarray(self.sub_atmos.dimensioned_view().ne[1])
            reqs.append(self.comm.Isend(ne_source, dest=self.rank - 1, tag=tag))

        if self.rank != self.Nchunks - 1:
            tag = LwMpiTags.NeTag
            tag += LwMpiTags.LowerBc
            ne_source = np.ascontiguousarray(self.sub_atmos.dimensioned_view().ne[-2])
            reqs.append(self.comm.Isend(ne_source, dest=self.rank + 1, tag=tag))

        # NOTE(cmo): Receive all
        if self.rank != 0:
            tag = LwMpiTags.NeTag
            tag += LwMpiTags.LowerBc
            ne_ghost = self.sub_atmos.dimensioned_view().ne
            ne_source = np.empty_like(ne_source[0])
            self.comm.Recv(ne_source, source=self.rank - 1, tag=tag)
            ne_ghost[0] = ne_source

        if self.rank != self.Nchunks - 1:
            tag = LwMpiTags.NeTag
            tag += LwMpiTags.UpperBc
            ne_ghost = self.sub_atmos.dimensioned_view().ne
            ne_source = np.empty_like(ne_ghost[-1])
            self.comm.Recv(ne_source, source=self.rank + 1, tag=tag)
            ne_ghost[-1] = ne_source

        MPI.Request.Waitall(reqs)

    def update_ghost_rho(self):
        """Communicate PRD rho for the ghost points."""
        reqs = []
        for ia, a in enumerate(self.activeAtoms):
            for it, t in enumerate(a.trans):
                try:
                    t.rhoPrd
                except AttributeError:
                    continue

                if self.rank != 0:
                    base_tag = LwMpiTags.RhoPrdTag
                    base_tag += LwMpiTags.UpperBc
                    if self.atmos.Ndim == 1:
                        rho_ghost = t.rhoPrd[:, 1]
                    else:
                        rho_ghost = t.rhoPrd.reshape(
                            -1, self.sub_atmos.Nz, self.sub_atmos.Nx
                        )[:, 1]

                    tag = (
                        base_tag
                        + LwMpiTags.ElementStart
                        + ia * LwMpiTags.ElementStep
                        + it * LwMpiTags.LineStep
                    )
                    rho_ghost = np.ascontiguousarray(rho_ghost)
                    reqs.append(self.comm.Isend(rho_ghost, dest=self.rank - 1, tag=tag))

                if self.rank != self.Nchunks - 1:
                    base_tag = LwMpiTags.RhoPrdTag
                    base_tag += LwMpiTags.LowerBc
                    if self.atmos.Ndim == 1:
                        rho_ghost = t.rhoPrd[:, -2]
                    else:
                        rho_ghost = t.rhoPrd.reshape(
                            -1, self.sub_atmos.Nz, self.sub_atmos.Nx
                        )[:, -2]

                    tag = (
                        base_tag
                        + LwMpiTags.ElementStart
                        + ia * LwMpiTags.ElementStep
                        + it * LwMpiTags.LineStep
                    )
                    rho_ghost = np.ascontiguousarray(rho_ghost)
                    reqs.append(self.comm.Isend(rho_ghost, dest=self.rank + 1, tag=tag))

                if self.rank != 0:
                    base_tag = LwMpiTags.RhoPrdTag
                    base_tag += LwMpiTags.LowerBc
                    if self.atmos.Ndim == 1:
                        rho_ghost = t.rhoPrd
                    else:
                        rho_ghost = t.rhoPrd.reshape(
                            -1, self.sub_atmos.Nz, self.sub_atmos.Nx
                        )

                    tag = (
                        base_tag
                        + LwMpiTags.ElementStart
                        + ia * LwMpiTags.ElementStep
                        + it * LwMpiTags.LineStep
                    )
                    rho_source = np.ones(rho_ghost[:, 0].shape)
                    self.comm.Recv(rho_source, source=self.rank - 1, tag=tag)
                    rho_ghost[:, 0] = rho_source

                if self.rank != self.Nchunks - 1:
                    base_tag = LwMpiTags.RhoPrdTag
                    base_tag += LwMpiTags.UpperBc
                    if self.atmos.Ndim == 1:
                        rho_ghost = t.rhoPrd
                    else:
                        rho_ghost = t.rhoPrd.reshape(
                            -1, self.sub_atmos.Nz, self.sub_atmos.Nx
                        )

                    tag = (
                        base_tag
                        + LwMpiTags.ElementStart
                        + ia * LwMpiTags.ElementStep
                        + it * LwMpiTags.LineStep
                    )
                    rho_source = np.ones(rho_ghost[:, -1].shape)
                    self.comm.Recv(rho_source, source=self.rank + 1, tag=tag)
                    rho_ghost[:, -1] = rho_source

        MPI.Request.Waitall(reqs)
