"""Eager EnvMACE: exact cloned foundation, explicit pre-product feature tap."""
from copy import deepcopy

import torch

from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import prepare_graph, compute_hessians_vmap

from ..adapters import foundation_adapter
from .config import EnvHeadConfig
from .head import EnvHead


class EnvMACE(ScaleShiftMACE):
    def __getstate__(self):
        state = super().__getstate__()
        # Rebuild backend execution caches after copy/save/load; canonical model
        # tensors and the explicit backend choice are sufficient.
        state.pop("_cueq_foundation_cache", None)
        state.pop("_cueq_head_cache", None)
        return state

    def _apply(self, fn, recurse=True):
        self.__dict__.pop("_cueq_foundation_cache", None)
        self.__dict__.pop("_cueq_head_cache", None)
        return super()._apply(fn, recurse=recurse)

    def __init__(self, foundation, config=None, dtype=None):
        source_dtype = next(foundation.parameters()).dtype
        dtype = source_dtype if dtype is None else dtype
        if source_dtype not in (torch.float32, torch.float64) or dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError(
                "EnvMACE supports float32/float64 foundations and computation"
            )
        if source_dtype == torch.float32 and dtype == torch.float64:
            raise ValueError(
                "A float32 foundation cannot be promoted to float64; use float32 or supply a float64 foundation"
            )
        adapter = foundation_adapter(foundation)
        adapter.validate(foundation)
        for module in foundation.modules():
            for key in ("cueq_config", "oeq_config"):
                backend = getattr(module, key, None)
                if backend is not None and getattr(backend, "enabled", False):
                    raise NotImplementedError("Accelerated backends are unsupported")
        # Deepcopy preserves constructor-specific modules and metadata, including
        # readouts and scale/shift. No foundation reconstruction or key filtering.
        clone = deepcopy(foundation).to(dtype=dtype)
        torch.nn.Module.__init__(self)
        self.__dict__.update(clone.__dict__)
        if not hasattr(self, "heads"):
            self.heads = ["Default"]
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._envhead_adapter_name = adapter.name
        self._envhead_active = False
        interaction = adapter.interaction(self)
        previous_dtype = torch.get_default_dtype()
        try:
            # Construct head constants in the same precision as the foundation.
            torch.set_default_dtype(dtype)
            self.env_head = EnvHead(
                interaction.target_irreps,
                interaction,
                (config or EnvHeadConfig()),
            ).to(device=self.r_max.device, dtype=dtype)
        finally:
            torch.set_default_dtype(previous_dtype)

    def forward(
        self,
        data,
        training=False,
        compute_force=True,
        compute_polarizability=False,
        compute_hessian=False,
        **flags,
    ):
        allowed = {
            "compute_virials",
            "compute_stress",
            "compute_displacement",
            "compute_edge_forces",
            "compute_atomic_stresses",
            "lammps_mliap",
        }
        for name, value in flags.items():
            if name not in allowed:
                raise TypeError(f"Unknown forward argument: {name}")
            if value:
                raise NotImplementedError(f"EnvMACE does not support {name}")
        if (
            torch.compiler.is_compiling()
            or torch.is_inference_mode_enabled()
            or torch.jit.is_scripting()
            or torch.jit.is_tracing()
        ):
            raise NotImplementedError(
                "EnvMACE requires eager autograd, including at inference"
            )
        if self._envhead_active:
            raise RuntimeError(
                "Concurrent/reentrant EnvMACE calls require separate model instances"
            )
        self._envhead_active = True
        try:
            with torch.enable_grad():
                return self._forward(
                    data, training, compute_force, compute_polarizability, compute_hessian
                )
        finally:
            self._envhead_active = False

    def _forward(self, data, training, compute_force, compute_polarizability, compute_hessian):
        data = dict(data)
        if data.get("pbc") is not None and data["pbc"].any():
            raise NotImplementedError("Periodic embedding is unsupported")
        positions = data["positions"]
        if positions.dtype != next(self.parameters()).dtype:
            raise ValueError("Input dtype must match the foundation and EnvHead")
        mm = data["mm_positions"]
        charges = data["mm_charges"]
        counts = data["mm_count"].reshape(-1)
        if mm.shape != (len(mm), 3) or charges.shape != (len(mm), 1):
            raise ValueError("MM positions/charges must have shapes [M,3]/[M,1]")
        if (
            counts.dtype not in (torch.int32, torch.int64)
            or (counts < 0).any()
            or counts.sum() != len(mm)
        ):
            raise ValueError("Invalid mm_count")
        if any(
            x.dtype != positions.dtype or x.device != positions.device
            for x in (mm, charges)
        ):
            raise ValueError("MM tensors must share QM dtype and device")
        if not all(torch.isfinite(x).all() for x in (positions, mm, charges)):
            raise ValueError("Nonfinite coordinates or source charges")
        mm.requires_grad_(True)
        ctx = prepare_graph(
            data,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            lammps_mliap=False,
        )
        ngraph = ctx.num_graphs
        if len(counts) != ngraph:
            raise ValueError("mm_count must contain one entry per graph")
        # MM data belong exclusively to EnvHead. Native foundation sees QM only.
        for key in ("external_field", "external_potential", "fermi_level"):
            if data.get(key) is not None and data[key].any():
                raise ValueError(
                    "Supply environment through MM sources, not foundation external fields"
                )
        base_data = {k: v for k, v in data.items() if not k.startswith("mm_")}
        base_data["atomic_numbers"] = self.atomic_numbers[data["node_attrs"].argmax(-1)]
        foundation_result, tap = foundation_adapter(self).evaluate(self, base_data)
        base = foundation_result["energy"]
        edge_attrs = self.spherical_harmonics(ctx.vectors)
        edge_feats, cutoff = self.radial_embedding(
            ctx.lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        probe = (
            positions.new_zeros(ngraph, 3, requires_grad=True)
            if compute_polarizability
            else None
        )
        shift = data.get("potential_shift", positions.new_zeros(ngraph)).reshape(-1)
        origins = data.get("reference_origin", positions.new_zeros(ngraph, 3))
        if shift.shape != (ngraph,) or origins.shape != (ngraph, 3):
            raise ValueError("Invalid potential_shift/reference_origin shapes")
        if any(
            x.dtype != positions.dtype or x.device != positions.device
            for x in (shift, origins)
        ):
            raise ValueError(
                "Potential shift and reference origin must share coordinate dtype/device"
            )
        if not torch.isfinite(shift).all() or not torch.isfinite(origins).all():
            raise ValueError("Nonfinite potential shift/reference origin")
        if data.get("total_spin") is not None and (data["total_spin"] != 1).any():
            raise NotImplementedError("Spin-dependent delta is not implemented")
        # Explicit smooth cutoff for all new head paths, including edge MLPs
        # whose bias could otherwise leave nonzero messages at the boundary.
        cutoff = self.radial_embedding.cutoff_fn(ctx.lengths)
        totals = data.get("total_charge", positions.new_zeros(ngraph)).reshape(-1)
        if len(totals) != ngraph or not torch.isfinite(totals).all():
            raise ValueError("total_charge must be finite, with one value per graph")
        group_keys = ("charge_group", "charge_group_totals", "charge_group_count")
        has_groups = [data.get(key) is not None for key in group_keys]
        if any(has_groups) and not all(has_groups):
            raise ValueError("All charge_group tensors must be supplied together")
        group_counts = None
        if all(has_groups):
            group_counts = data["charge_group_count"].reshape(-1)
            if (
                group_counts.dtype not in (torch.int32, torch.int64)
                or len(group_counts) != ngraph
                or (group_counts < 1).any()
                or group_counts.sum() != data["charge_group_totals"].numel()
                or data["charge_group_totals"].ndim != 1
                or data["charge_group"].shape != (len(positions),)
            ):
                raise ValueError("Invalid charge_group batch shapes or counts")
        head_forward = self.env_head
        if getattr(self, "_envhead_enable_cueq", False):
            from functools import partial
            from ..backends import call_cueq_head

            head_forward = partial(call_cueq_head, self)
        out = head_forward(
            tap,
            positions,
            mm,
            charges,
            totals,
            data["node_attrs"],
            data["edge_index"],
            edge_attrs,
            edge_feats,
            cutoff,
            data.get("charge_group"),
            data.get("charge_group_totals"),
            probe_field=probe,
            potential_shift=shift,
            reference_origin=origins,
            graph_ptr=data["ptr"],
            mm_counts=counts,
            charge_group_counts=group_counts,
            mm_cell=data.get("mm_cell"),
            rf_cutoff=data.get("rf_cutoff"),
            rf_dielectric=data.get("rf_dielectric", 78.4),
            atomic_numbers=base_data["atomic_numbers"],
        )
        total = base + out["delta_energy"]
        if not torch.isfinite(total).all():
            raise FloatingPointError("Nonfinite total energy")
        if compute_polarizability:
            dipole = -torch.autograd.grad(
                total.sum(), probe, create_graph=True, retain_graph=True
            )[0]
            alpha = torch.stack(
                [
                    torch.autograd.grad(
                        dipole[:, j].sum(),
                        probe,
                        create_graph=training,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    for j in range(3)
                ],
                dim=1,
            )
            out.update(energy_dipole=dipole, polarizability=alpha)
        forces, mm_forces = None, None
        if compute_force or compute_hessian:
            gradients = torch.autograd.grad(
                total.sum(),
                (positions, mm),
                create_graph=training or compute_hessian,
                retain_graph=training or compute_hessian,
                allow_unused=True,
            )
            forces, mm_forces = tuple(
                -grad if grad is not None else torch.zeros_like(pos)
                for grad, pos in zip(gradients, (positions, mm))
            )
        if (compute_force or compute_hessian) and not all(
            torch.isfinite(f).all() for f in (forces, mm_forces)
        ):
            raise FloatingPointError("Nonfinite coordinate forces")
        hessian = None
        if compute_hessian:
            try:
                hessian = compute_hessians_vmap(forces, positions).detach()
            except RuntimeError as error:
                raise RuntimeError(
                    "Analytic Hessian failed in the complete foundation/EnvHead "
                    "autograd path; no numerical or backend fallback was applied"
                ) from error
            if not torch.isfinite(hessian).all():
                raise FloatingPointError(
                    "Nonfinite coordinate Hessian in the complete foundation/EnvHead "
                    "energy. Check foundation second derivatives, including LES "
                    "zero-distance self terms; no numerical fallback was applied"
                )
        out.update(
            energy=total,
            base_energy=base,
            forces=forces,
            mm_forces=mm_forces,
            node_energy=None,
            stress=None,
            virials=None,
            hessian=hessian,
            displacement=None,
            node_feats=foundation_result.get("node_feats"),
        )
        if "les_energy" in foundation_result:
            out["base_les_energy"] = foundation_result["les_energy"]
        return out
