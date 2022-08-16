import pathlib
import pickle
from abc import abstractmethod
from logging import root
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar

import lightweaver as lw
import numpy as np
import zarr

from MpiStackedContext import MpiStackedContext


class LwVisitor(Protocol):
    """API for visitors for extracting and injecting state into a context.

    N.B. all state arrays coming through these functions is expected to have z as its second axis. This means that wavelength/number of levels remains the first axis, then populations as usual, z, y, x. For some, like ne, this requires the creation of a unit-length first axis."""

    def __call__(self, ctx: lw.Context, remove_ghost: bool = True) -> np.ndarray:
        raise NotImplementedError

    def set(self, ctx: lw.Context, val: np.ndarray):
        raise NotImplementedError

    def shape(self, ctx: lw.Context) -> Tuple:
        raise NotImplementedError


class PopsVisitor(LwVisitor):
    """Gather a specific population. See `make_pops_visitors`."""

    def __init__(self, pop_name: str):
        self.pop_name = pop_name

    def __call__(self, ctx: lw.Context, remove_ghost: bool = True) -> np.ndarray:
        result = ctx.eqPops.dimensioned_view()[self.pop_name]
        if remove_ghost:
            return result[:, ctx.cstart : ctx.cend]
        return result

    def set(self, ctx: lw.Context, val: np.ndarray):
        # NOTE(cmo): Expects to receive ghost points as part of val
        pops = ctx.eqPops.dimensioned_view()[self.pop_name]
        pops[...] = val

    def shape(self, ctx: lw.Context) -> Tuple:
        pops = ctx.eqPops.dimensioned_view()[self.pop_name]
        return pops.shape


class NeVisitor(LwVisitor):
    """Gather electron density."""

    @staticmethod
    def get_ne(ctx: lw.Context) -> np.ndarray:
        result = ctx.atmos.pyAtmos.dimensioned_view().ne.unsqueeze(0)
        return result

    def __call__(self, ctx: lw.Context, remove_ghost: bool = True) -> np.ndarray:
        result = self.get_ne(ctx)
        if remove_ghost:
            return result[:, ctx.cstart : ctx.cend]
        return result

    def set(self, ctx: lw.Context, val: np.ndarray):
        ne = self.get_ne(ctx)
        ne[...] = val

    def shape(self, ctx: lw.Context) -> Tuple:
        ne = self.get_ne(ctx)
        return ne.shape


class JVisitor(LwVisitor):
    """Gather J: warning, could be very big."""

    def __call__(self, ctx: lw.Context, remove_ghost: bool = True) -> np.ndarray:
        result = ctx.spect.J
        if remove_ghost:
            return result[:, ctx.cstart : ctx.cend]
        return result

    def set(self, ctx: lw.Context, val: np.ndarray):
        ctx.spect.J[...] = val

    def shape(self, ctx: lw.Context) -> Tuple:
        return ctx.spect.J.shape


class RhoPrdVisitor(LwVisitor):
    """Gather rho prd for a specific atom and transition. See `make_prd_visitors`."""

    def __init__(self, atom_idx: int, trans_idx: int):
        self.atom_idx = atom_idx
        self.trans_idx = trans_idx

    def get_rho(self, ctx: lw.Context):
        result = ctx.activeAtoms[self.atom_idx].trans[self.trans_idx].rhoPrd
        if ctx.atmos.Ndim != 1:
            result = result.reshape(-1, ctx.atmos.Nz, ctx.atmos.Nx)
        return result

    def __call__(self, ctx: lw.Context, remove_ghost: bool = True):
        result = self.get_rho(ctx)

        if remove_ghost:
            return result[:, ctx.cstart : ctx.cend]
        return result

    def set(self, ctx: lw.Context, val: np.ndarray):
        rho = self.get_rho(ctx)
        rho[...] = val

    def shape(self, ctx: lw.Context):
        return self.get_rho(ctx).shape


class WavelengthVisitor(LwVisitor):
    """Gather the wavelength array (same on every node, see `root_only`)"""

    def __call__(self, ctx: lw.Context, remove_ghost: bool = True) -> np.ndarray:
        return ctx.spect.wavelength

    def set(self, ctx: lw.Context, val: np.ndarray):
        pass

    def shape(self, ctx: lw.Context) -> Tuple:
        return ctx.spect.wavelength.shape


class IntensityVisitor(LwVisitor):
    """Gather the intensity array (probably only meaningful from the top node (conventionally the root), see `root_only`)"""

    def __call__(self, ctx: lw.Context, remove_ghost: bool = True) -> np.ndarray:
        return ctx.spect.I

    def set(self, ctx: lw.Context, val: np.ndarray):
        pass

    def shape(self, ctx: lw.Context) -> Tuple:
        return ctx.spect.I.shape


def gather_visitor_mpi_ctx(
    ctx: MpiStackedContext,
    visitor: LwVisitor,
    root: int = 0,
) -> Optional[np.ndarray]:
    """Use a visitor to gather a quantity from all nodes to the root node."""

    rank = ctx.rank
    comm = ctx.comm
    result = None

    if rank != root:
        send_buf = np.ascontiguousarray(visitor(ctx, remove_ghost=True))

        comm.Send(send_buf, dest=root)
    else:
        self_buffer = visitor(ctx, remove_ghost=True)
        if ctx.atmos.Ndim == 1:
            result = np.zeros((self_buffer.shape[0], ctx.global_atmos.Nz))
        else:
            result = np.zeros(
                (self_buffer.shape[0], ctx.global_atmos.Nz, ctx.global_atmos.Nx)
            )

        z_start = 0
        for recv_rank in range(ctx.Nchunks):
            if recv_rank != root:
                if ctx.atmos.Ndim == 1:
                    recv_buf = np.zeros(
                        (self_buffer.shape[0], ctx.ctx_sizes[recv_rank])
                    )
                else:
                    recv_buf = np.zeros(
                        (
                            self_buffer.shape[0],
                            ctx.ctx_sizes[recv_rank],
                            ctx.global_atmos.Nx,
                        )
                    )

                comm.Recv(recv_buf, source=recv_rank)
            else:
                recv_buf = self_buffer

            z_end = z_start + ctx.ctx_sizes[recv_rank]
            result[:, z_start:z_end] = recv_buf
            z_start += ctx.ctx_sizes[recv_rank]

    return result


def gather_mpi_to_dict(
    ctx: MpiStackedContext,
    visitors: Dict[Any, LwVisitor],
    root: int = 0,
    root_only: Optional[List[Any]] = None,
) -> Dict[Any, np.ndarray]:
    """Gather from a dict of visitors across all nodes to a dict of full arrays on the root node. Keys in `root_only` will only be run on the root (e.g. wavelength/I)."""

    if root_only is None:
        root_only = []

    result = {}

    for k, v in visitors.items():
        if k in root_only and ctx.rank == root:
            result[k] = v(ctx, remove_ghost=True)
            continue

        val = gather_visitor_mpi_ctx(ctx, v, root=root)
        if val is None:
            continue
        result[k] = val

    return result


def make_pops_visitors(ctx: MpiStackedContext):
    """Return a dict of visitors for the NLTE populations in a Context."""
    active_atoms = [a.element for a in ctx.aSet.activeAtoms]
    return {a.name: PopsVisitor(a.name) for a in active_atoms}


def make_prd_visitors(ctx: MpiStackedContext):
    """Return a dict of PRD rho visitors for PRD transitions in a Context."""
    active_atoms = [a.element for a in ctx.aSet.activeAtoms]

    visitors = {}
    for ia, a in enumerate(ctx.activeAtoms):
        for it, t in enumerate(a.trans):
            try:
                t.rhoPrd
            except AttributeError:
                continue
            visitors[(active_atoms[ia].name, it)] = RhoPrdVisitor(ia, it)
    return visitors


def scatter_visitor_mpi_ctx(
    ctx: MpiStackedContext,
    visitor: LwVisitor,
    data: Optional[np.ndarray],
    root: int = 0,
):
    """Use a visitor to scatter an array on the root node across all nodes."""
    if ctx.rank != root:
        shape = visitor.shape(ctx)
        data = np.zeros(shape)

        ctx.comm.Recv(data, source=root)
        visitor.set(ctx, data)
    else:
        z_start = 0
        for i, s in enumerate(ctx.ctx_sizes):
            z_ghost_start = z_start
            z_ghost_end = z_start + s

            if i != 0:
                z_ghost_start -= 1
            if i != ctx.Nchunks - 1:
                z_ghost_end += 1

            send_slice = np.ascontiguousarray(data[:, z_ghost_start:z_ghost_end])

            z_start += s

            if i != root:
                ctx.comm.Send(send_slice, dest=i)
            else:
                visitor.set(ctx, send_slice)


def scatter_mpi_from_dict(
    ctx: MpiStackedContext,
    visitors: Dict[Any, LwVisitor],
    data: Optional[Dict[Any, np.ndarray]],
    root: int = 0,
    ignore: Optional[List[Any]] = None,
):
    """Scatter data from a dict across all nodes using a dict of visitors, keys in the `ignore` list will be ingored."""
    if ignore is None:
        ignore = []
    if data is None:
        data = {k: None for k in visitors}

    for k, v in visitors.items():
        if k in ignore:
            continue
        scatter_visitor_mpi_ctx(ctx, v, data[k], root=root)


PathLike = TypeVar("PathLike", str, pathlib.Path)


class LwVisitorSerializer:
    """Base class for visitor serialization interface."""

    def __init__(
        self,
        visitors: Dict[Any, LwVisitor],
        root: int = 0,
    ):
        self.visitors = visitors
        self.root = root

    def serialize(
        self,
        instance: MpiStackedContext,
        root_only: Optional[List[Any]] = None,
    ) -> Dict[Any, np.ndarray]:
        """Serialize a class instance: gathers to root node."""

        return gather_mpi_to_dict(
            instance,
            self.visitors,
            root=self.root,
            root_only=root_only,
        )

    def deserialize(
        self,
        target: MpiStackedContext,
        data: Optional[Dict[Any, np.ndarray]],
        ignore: Optional[List[Any]] = None,
    ):
        """Deserialize an instance (by scattering) into the provided target per node."""

        scatter_mpi_from_dict(
            target,
            self.visitors,
            data,
            root=self.root,
            ignore=ignore,
        )

    @abstractmethod
    def save(
        self,
        instance: MpiStackedContext,
        path: PathLike,
        root_only: Optional[List[Any]] = None,
    ):
        """Serialize a class instance and save in path."""
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        target: MpiStackedContext,
        path: PathLike,
        ignore: Optional[List[Any]] = None,
    ):
        """Load data from path and deserialize into target per node."""
        raise NotImplementedError


class PickleLwVisitorSerializer(LwVisitorSerializer):
    """Pickle specialisation of serializer, probably not a good idea for large datasets!"""

    def save(
        self,
        instance: MpiStackedContext,
        path: PathLike,
        root_only: Optional[List[Any]] = None,
    ):
        """Gather the data and save to path via pickle."""
        data = self.serialize(instance, root_only=root_only)
        if instance.rank == self.root:
            with open(path, "wb") as pkl:
                pickle.dump(data, pkl)

    def load(
        self,
        target: MpiStackedContext,
        path: PathLike,
        ignore: Optional[List[Any]] = None,
    ):
        """Load from pickle and scatter across nodes into target."""
        data = None
        if target.rank == self.root:
            with open(path, "rb") as pkl:
                data = pickle.load(pkl)

        self.deserialize(target, data, ignore=ignore)


class ZarrLwVisitorSerializer(LwVisitorSerializer):
    """Zarr based backend for visitor-based serializer."""

    def save(
        self,
        instance: MpiStackedContext,
        path: PathLike,
        root_only: Optional[List[Any]] = None,
    ):
        """Gather the data and save to path via zarr."""
        data = self.serialize(instance, root_only=root_only)
        if instance.rank == self.root:
            file = zarr.convenience.open(path, "w")
            for k, v in data.items():
                file[k] = v

    def load(
        self,
        target: MpiStackedContext,
        path: PathLike,
        ignore: Optional[List[Any]] = None,
    ):
        """Load from zarr and scatter across nodes into target."""
        data = None
        if target.rank == self.root:
            file = zarr.convenience.open(path, "r")

            data = {}
            for k in self.visitors.keys():
                data[k] = file[k][...]

        self.deserialize(target, data, ignore=ignore)
