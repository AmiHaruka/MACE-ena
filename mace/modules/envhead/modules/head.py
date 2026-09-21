"""Direct delta energy with fixed-depth, charge-conserving density feedback."""
import math

import torch
import torch.nn.functional as F
from e3nn import o3
from mace.modules.wrapper_ops import Linear
from ..data import charge_groups
from .electrostatics import multipole_kernel, source_features, quadrupole_tensor
from .fusion import FusionBlock

# Charge-route conventions (eem/sqe). The solve only determines the q channel;
# the energy stays learned + p.v (no EEM self-energy, so nothing double-counts
# against the frozen foundation's long-range electrostatics).
CHI_BOUND_EV = 25.0  # tanh bound keeps chi outputs in a physical electronegativity range
HARDNESS_FLOOR_EV = 0.5  # positive hardness floor keeps the bordered system nonsingular
ZETA_FLOOR_EV = 0.5  # bond-hardness floor keeps the split-charge system positive definite
BOND_SCALE = 1.3  # covalent bond if r < BOND_SCALE * (r_i + r_j), D-SQE-style distance rule
COVALENT_RADII_ANGSTROM = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 16: 1.05}


def eem_group_solve(
    coulomb, hardness, chi, group_ids, group_totals, external_potential=None
):
    """Bordered EEM system for one graph; returns equalized atomic charges.

    Stationarity of chi.q + 0.5 q.(C+diag(eta)).q + (phi_MM.q) under per-group
    sum rules, i.e. chi enters as chi + phi_MM so the solved density carries
    the explicit EEM polarization response to the embedding potential. The
    Lagrange multipliers equalize dE/dq within every charge group; a constant
    shift of phi_MM is absorbed by the multipliers (1 lies in span(group_ids)).
    """
    n = chi.shape[0]
    if external_potential is not None:
        chi = chi + external_potential
    count = int(group_ids.max()) + 1
    rows = torch.arange(n, device=chi.device)
    system = torch.zeros(n + count, n + count, dtype=chi.dtype, device=chi.device)
    system[:n, :n] = coulomb + torch.diag(hardness)
    system[rows, n + group_ids] = 1.0
    system[n + group_ids, rows] = 1.0
    rhs = torch.cat((-chi, group_totals))
    return torch.linalg.solve(system, rhs)[:n]


def sqe_group_refine(
    coulomb, hardness, chi, zeta, q_start, senders, receivers, external_potential=None
):
    """Split-charge normal equations for one bond-connected charge group.

    Minimizes 0.5 zeta.d^2 + 0.5 (q0+Bd).C.(q0+Bd) + (chi+phi_MM).(q0+Bd) over
    split charges d; the bond driving voltage includes the embedding potential
    difference (the D-SQE battery term). Antisymmetric incidence makes 1.B.d
    identically zero, so the group sums of q_start are conserved exactly and no
    constraint row exists.
    """
    m = q_start.shape[0]
    e = senders.shape[0]
    if external_potential is not None:
        chi = chi + external_potential
    c_matrix = coulomb + torch.diag(hardness)
    b_matrix = torch.zeros(m, e, dtype=q_start.dtype, device=q_start.device)
    b_matrix[senders, torch.arange(e, device=q_start.device)] = 1.0
    b_matrix[receivers, torch.arange(e, device=q_start.device)] = -1.0
    system = zeta * torch.eye(e, dtype=q_start.dtype, device=q_start.device)
    system = system + b_matrix.T @ c_matrix @ b_matrix
    rhs = -(b_matrix.T @ (c_matrix @ q_start + chi))
    return b_matrix @ torch.linalg.solve(system, rhs)


def zero_readout(module):
    for p in module.parameters():
        torch.nn.init.zeros_(p)
    return module


def segment_sum(values, ids, size):
    return values.new_zeros((size,) + values.shape[1:]).index_add(0, ids, values)


def segment_max(values, ids, size):
    return values.new_full((size,), -torch.inf).scatter_reduce(
        0, ids, values, reduce="amax", include_self=True
    )


def centered(values, batch, sizes):
    means = segment_sum(values[:, 0], batch, len(sizes)) / sizes
    return torch.cat((values[:, :1] - means[batch, None], values[:, 1:]), -1)


class EnvHead(torch.nn.Module):
    def __init__(self, tap_irreps, foundation_interaction, config, cueq_config=None):
        super().__init__()
        self.config = config
        self.tap_irreps = o3.Irreps(tap_irreps)
        self.hidden = o3.Irreps([(config.channels, (l, (-1) ** l)) for l in range(3)])
        self.density = o3.Irreps(
            [(1, (l, (-1) ** l)) for l in range(config.density_max_l + 1)]
        )
        self.state_dim = self.density.dim
        self.geometry = Linear(self.tap_irreps, self.hidden, cueq_config=cueq_config)
        self.missing = [
            l
            for l in range(config.density_max_l + 1)
            if self.tap_irreps.count(o3.Irrep(l, (-1) ** l)) == 0
        ]
        if self.missing and not config.mmc:
            raise ValueError(
                f"Foundation tap lacks density irreps l={self.missing}; enable mmc or lower density_max_l"
            )
        if self.tap_irreps.count(o3.Irrep("0e")) == 0:
            raise ValueError("Foundation must provide scalar geometry features")
        self.mmc_irreps = o3.Irreps(
            [(config.channels, (l, (-1) ** l)) for l in self.missing]
        )
        self.mmc_map = None
        if self.missing:
            self.mmc_weights = torch.nn.Sequential(
                torch.nn.Linear(
                    config.channels + foundation_interaction.edge_feats_irreps.dim,
                    config.channels,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(config.channels, config.channels * len(self.missing)),
            )
            self.mmc_map = Linear(self.mmc_irreps, self.hidden, cueq_config=cueq_config)
        self.conditioner = torch.nn.Sequential(
            torch.nn.Linear(config.channels + 3, config.channels),
            torch.nn.SiLU(),
            torch.nn.Linear(config.channels, config.channels),
        )
        self.initial_readout = zero_readout(
            o3.Linear(self.hidden, self.density, biases=True)
        )
        self.blocks = torch.nn.ModuleList(
            [
                FusionBlock(
                    self.hidden, self.density, foundation_interaction, config,
                    cueq_config=cueq_config,
                )
                for _ in range(config.num_interactions)
            ]
        )
        self.density_readouts = torch.nn.ModuleList(
            [
                zero_readout(o3.Linear(self.hidden, self.density, biases=True))
                for _ in self.blocks
            ]
        )
        self.fukui_readouts = torch.nn.ModuleList(
            [
                zero_readout(torch.nn.Linear(config.channels, 1))
                for _ in range(config.num_interactions + 1)
            ]
        )
        self.route_gates = None
        if config.charge_route == "eem":
            # Hidden layers keep their default init: an all-zero readout is an
            # exact gradient stationary point through the solve (uniform chi is
            # absorbed by the multipliers). Zero final layers keep a fresh head
            # forward bit-identical to the fukui route.
            self.eem_chi = self._route_readout_module()
            self.eem_eta = self._route_readout_module()
            self.route_gates = self._route_gates()
        elif config.charge_route == "sqe":
            self.sqe_chi = self._route_readout_module()
            self.sqe_eta = self._route_readout_module()
            self.sqe_zeta = self._route_readout_module(2 * config.channels + 1)
            self.route_gates = self._route_gates()
        self.energy_maps = torch.nn.ModuleList(
            [
                Linear(ir, self.hidden, cueq_config=cueq_config)
                for ir in (
                    self.hidden,
                    self.hidden,
                    self.density,
                    self.density,
                    self.density,
                )
            ]
        )
        self.energy_readout = torch.nn.Sequential(
            torch.nn.Linear(3 * config.channels, config.channels),
            torch.nn.SiLU(),
            torch.nn.Linear(config.channels, 1),
        )
        zero_readout(self.energy_readout[-1])

    def _route_readout_module(self, input_dim=None):
        channels = self.config.channels
        return torch.nn.Sequential(
            torch.nn.Linear(channels if input_dim is None else input_dim, channels),
            torch.nn.SiLU(),
            zero_readout(torch.nn.Linear(channels, 1)),
        )

    def _route_gates(self):
        return torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.zeros(()))
                for _ in range(self.config.num_interactions + 1)
            ]
        )

    def _bound_chi(self, module, z):
        raw = module(z[:, : self.config.channels]).squeeze(-1)
        return CHI_BOUND_EV * torch.tanh(raw / CHI_BOUND_EV)

    def _charge_update(self, p_tilde, z, ids, totals, layer, ctx):
        if ctx is None:
            return self.equilibrate(p_tilde, z, ids, totals, layer)
        if self.config.charge_route == "eem":
            charge = self._eem_charges(z, ids, totals, layer, ctx)
        else:
            charge = self._sqe_charges(p_tilde, z, ids, totals, layer, ctx)
        return torch.cat((charge[:, None], p_tilde[:, 1:]), -1)

    def _eem_charges(self, z, ids, totals, layer, ctx):
        coulomb, bounds, group_sizes, _, _, _, _, ptr, potential = ctx
        chi = self._bound_chi(self.eem_chi, z)
        eta = (
            F.softplus(self.eem_eta(z[:, : self.config.channels]).squeeze(-1))
            + HARDNESS_FLOOR_EV
        )
        solved = torch.empty_like(chi)
        for g, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            solved[start:end] = eem_group_solve(
                coulomb[g],
                eta[start:end],
                chi[start:end],
                ids[start:end] - bounds[g][0],
                totals[bounds[g][0] : bounds[g][1]],
                external_potential=potential[start:end],
            )
        uniform = totals[ids] / group_sizes[ids]
        gate = self.route_gates[layer]
        return uniform + gate * (solved - uniform)

    def _covalent_radii(self, atomic_numbers):
        if atomic_numbers is None:
            raise ValueError("SQE route requires atomic_numbers from the foundation table")
        largest = int(atomic_numbers.max()) + 1
        table = torch.full(
            (largest,),
            -1.0,
            dtype=self.route_gates[0].dtype,
            device=atomic_numbers.device,
        )
        for number, value in COVALENT_RADII_ANGSTROM.items():
            if number < largest:
                table[number] = value
        radii = table[atomic_numbers]
        if (radii < 0).any():
            missing = sorted(set(atomic_numbers[radii < 0].tolist()))
            raise ValueError(
                "SQE route: no covalent radius for atomic numbers "
                f"{missing}; extend COVALENT_RADII_ANGSTROM"
            )
        return radii

    def _sqe_charges(self, p_tilde, z, ids, totals, layer, ctx):
        coulomb, bounds, _, positions, _, _, atomic_numbers, ptr, potential = ctx
        base = self.equilibrate(p_tilde, z, ids, totals, layer)[:, 0]
        chi = self._bound_chi(self.sqe_chi, z)
        eta = (
            F.softplus(self.sqe_eta(z[:, : self.config.channels]).squeeze(-1))
            + HARDNESS_FLOOR_EV
        )
        radii = self._covalent_radii(atomic_numbers)
        gate = self.route_gates[layer]
        channels = self.config.channels
        refined = base.clone()
        for g, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            n = end - start
            # Manual distance (cdist lacks a second derivative); live positions
            # keep the bond-length contribution to zeta in the force graph.
            difference = positions[start:end][:, None, :] - positions[start:end][None, :, :]
            distance = difference.square().sum(-1).clamp_min(1e-12).sqrt()
            local_radii = radii[start:end]
            bonds = (
                distance < BOND_SCALE * (local_radii[:, None] + local_radii[None, :])
            ) & ~torch.eye(n, dtype=torch.bool, device=distance.device)
            group_local = ids[start:end] - bounds[g][0]
            if (bonds & (group_local[:, None] != group_local[None, :])).any():
                raise ValueError(
                    "SQE route: covalent graph crosses charge_group boundaries"
                )
            sender, receiver = torch.triu(bonds, 1).nonzero(as_tuple=True)
            if sender.numel() == 0:
                continue
            bond_length = distance[sender, receiver]
            zeta = (
                F.softplus(
                    self.sqe_zeta(
                        torch.cat(
                            (
                                z[start + sender, :channels],
                                z[start + receiver, :channels],
                                bond_length[:, None],
                            ),
                            -1,
                        )
                    ).squeeze(-1)
                )
                + ZETA_FLOOR_EV
            )
            low, high = bounds[g]
            for gid in range(high - low):
                inside = group_local[sender] == gid
                if not inside.any():
                    continue
                members = (group_local == gid).nonzero(as_tuple=True)[0]
                lookup = torch.full(
                    (n,), -1, dtype=torch.long, device=members.device
                )
                lookup[members] = torch.arange(len(members), device=members.device)
                refined[start + members] += gate * sqe_group_refine(
                    coulomb[g][members[:, None], members[None, :]],
                    eta[start + members],
                    chi[start + members],
                    zeta[inside],
                    base[start + members],
                    lookup[sender[inside]],
                    lookup[receiver[inside]],
                    external_potential=potential[start + members],
                )
        return refined

    def flatten_tap(self, tap):
        if tap.ndim == 2:
            if tap.shape[-1] != self.tap_irreps.dim:
                raise ValueError("Invalid flat tap dimension")
            return tap
        pieces = []
        offset = 0
        for mul, ir in self.tap_irreps:
            if tap.shape[1] != mul:
                raise ValueError("Unsupported foundation interaction layout")
            pieces.append(
                tap[:, :, offset : offset + ir.dim].reshape(len(tap), mul * ir.dim)
            )
            offset += ir.dim
        if offset != tap.shape[2]:
            raise ValueError("Invalid tap irreps layout")
        return torch.cat(pieces, -1)

    def equilibrate(self, p, z, ids, totals, layer):
        logits = self.fukui_readouts[layer](z[:, : self.config.channels]).squeeze(-1)
        maxima = segment_max(logits.detach(), ids, len(totals))
        weights = (logits - maxima[ids]).exp()
        weights = weights / segment_sum(weights, ids, len(totals))[ids]
        residual = totals - segment_sum(p[:, 0], ids, len(totals))
        return torch.cat((p[:, :1] + (weights * residual[ids])[:, None], p[:, 1:]), -1)

    def forward(
        self,
        tap,
        positions,
        mm_positions,
        mm_charges,
        total_charge,
        node_attrs,
        edge_index,
        edge_attrs,
        edge_feats,
        cutoff,
        charge_group=None,
        charge_group_totals=None,
        probe_field=None,
        potential_shift=None,
        reference_origin=None,
        graph_ptr=None,
        mm_counts=None,
        charge_group_counts=None,
        mm_cell=None, rf_cutoff=None, rf_dielectric=78.4,
        atomic_numbers=None,
    ):
        single = graph_ptr is None
        ptr = [0, len(positions)] if single else graph_ptr.detach().cpu().tolist()
        counts = [len(mm_positions)] if single else mm_counts.detach().cpu().tolist()
        graph_totals = total_charge.reshape(-1)
        sizes = positions.new_tensor([b - a for a, b in zip(ptr[:-1], ptr[1:])])
        if (sizes <= 0).any():
            raise ValueError("Empty ML graphs are unsupported")
        batch = torch.repeat_interleave(
            torch.arange(len(sizes), device=positions.device), sizes.long()
        )
        group_counts = (
            [len(charge_group_totals)]
            if single and charge_group_totals is not None
            else charge_group_counts.detach().cpu().tolist()
            if charge_group_counts is not None
            else [1] * len(sizes)
        )
        all_ids, all_totals, group_graph = [], [], []
        route_group_bounds = []
        group_offset = 0
        kernels, external_parts = [], []
        mm_offset = 0
        cfg = self.config
        for g, (start, end) in enumerate(zip(ptr[:-1], ptr[1:])):
            local_ids, local_totals = charge_groups(
                positions[start:end],
                graph_totals[g],
                None if charge_group is None else charge_group[start:end],
                None
                if charge_group_totals is None
                else charge_group_totals[group_offset : group_offset + group_counts[g]],
            )
            all_ids.append(local_ids + group_offset)
            all_totals.append(local_totals)
            group_graph.extend([g] * len(local_totals))
            route_group_bounds.append((group_offset, group_offset + len(local_totals)))
            group_offset += len(local_totals)
            kernels.append(
                multipole_kernel(
                    positions[start:end], cfg.gaussian_width, cfg.density_max_l
                )
            )
            external_parts.append(
                source_features(
                    positions[start:end],
                    mm_positions[mm_offset : mm_offset + counts[g]],
                    mm_charges[mm_offset : mm_offset + counts[g]],
                    cfg.gaussian_width,
                    cfg.density_max_l,
                    cfg.mm_chunk_size,
                    cell=None if mm_cell is None else mm_cell[g],
                    rf_cutoff=rf_cutoff, rf_dielectric=rf_dielectric,
                )
            )
            mm_offset += counts[g]
        ids, totals = torch.cat(all_ids), torch.cat(all_totals)
        group_graph = torch.tensor(group_graph, device=positions.device)
        external = torch.cat(external_parts)

        def internal_field(p):
            # Independent dense blocks; no cross-graph N_total squared matrix.
            values = torch.cat(
                [
                    (kernel @ p[start:end].flatten()).reshape(end - start, -1)
                    for kernel, start, end in zip(kernels, ptr[:-1], ptr[1:])
                ]
            )
            return centered(values, batch, sizes)

        cfg = self.config
        c = cfg.channels
        n = len(positions)
        h = self.geometry(self.flatten_tap(tap))
        if self.mmc_map is not None:
            s, r = edge_index
            directions = positions[r] - positions[s]
            weights = self.mmc_weights(torch.cat((h[s, :c], edge_feats), -1)).reshape(
                -1, len(self.missing), c
            )
            edge_weight = h.new_ones(len(s)) if cutoff is None else cutoff.reshape(-1)
            degree = h.new_ones(n).index_add(0, r, edge_weight)
            components = []
            for idx, l in enumerate(self.missing):
                sh = o3.spherical_harmonics(
                    l, directions, normalize=True, normalization="component"
                )
                msg = (weights[:, idx, :, None] * sh[:, None, :]).flatten(1)
                if cutoff is not None:
                    msg = msg * cutoff.reshape(-1, 1)
                components.append(
                    h.new_zeros(n, c * (2 * l + 1)).index_add(0, r, msg)
                    / degree[:, None]
                )
            h = h + self.mmc_map(torch.cat(components, -1))
        group_sizes = torch.bincount(ids, minlength=len(totals)).to(h.dtype)
        qfeatures = torch.stack(
            (
                graph_totals[batch],
                graph_totals[batch] / sizes[batch],
                totals[ids] / group_sizes[ids],
            ),
            -1,
        )
        h = torch.cat(
            (
                h[:, :c] + self.conditioner(torch.cat((h[:, :c], qfeatures), -1)),
                h[:, c:],
            ),
            -1,
        )
        origins = (
            positions.new_zeros(len(sizes), 3)
            if reference_origin is None
            else reference_origin.reshape(-1, 3)
        )
        origin = origins[batch]
        if probe_field is not None:
            field = probe_field.reshape(-1, 3)[batch]
            probe = torch.cat(
                (
                    -((positions - origin) * field).sum(-1, keepdim=True),
                    -field,
                    positions.new_zeros(n, self.state_dim - 4),
                ),
                -1,
            )
            external = external + probe
        if potential_shift is not None:
            external = external + torch.cat(
                (
                    potential_shift.reshape(-1)[batch, None],
                    positions.new_zeros(n, self.state_dim - 1),
                ),
                -1,
            )
        # Physical coefficients are never normalized in the explicit energy.
        # Fixed unit scales: 1 e, 1 Angstrom, 1 eV.
        ext = centered(external, batch, sizes)
        z = h
        route_ctx = None
        if self.config.charge_route != "fukui":
            # Per-graph q-q Gaussian Coulomb blocks with the analytic self
            # term on the diagonal, assembled once and reused by every layer.
            # The uncentered embedding potential (probe/shift included) drives
            # the solves, so the solved density carries the explicit EEM/SQE
            # polarization response to the environment.
            self_term = 1.0 / (self.config.gaussian_width * math.sqrt(math.pi))
            blocks = []
            for kernel, (start, end) in zip(kernels, zip(ptr[:-1], ptr[1:])):
                n_g = end - start
                stride = self.density.dim
                block = kernel.reshape(n_g, stride, n_g, stride)[:, 0, :, 0]
                blocks.append(
                    block
                    + self_term * torch.eye(n_g, dtype=block.dtype, device=block.device)
                )
            route_ctx = (
                blocks,
                route_group_bounds,
                group_sizes,
                positions,
                node_attrs,
                edge_index,
                atomic_numbers,
                ptr,
                external[:, 0],
            )
        p = self._charge_update(self.initial_readout(z), z, ids, totals, 0, route_ctx)
        for k, block in enumerate(self.blocks):
            internal = internal_field(p)
            z = block(
                h,
                z,
                p,
                internal,
                ext,
                node_attrs,
                edge_index,
                edge_attrs,
                edge_feats,
                cutoff,
            )
            p = self._charge_update(
                p + self.density_readouts[k](z), z, ids, totals, k + 1, route_ctx
            )
        internal = internal_field(p)
        features = sum(
            layer(value)
            for layer, value in zip(self.energy_maps, (h, z, p, internal, ext))
        )
        invariants = torch.cat(
            (
                features[:, :c],
                features[:, c : 4 * c].reshape(n, c, 3).square().sum(-1),
                features[:, 4 * c :].reshape(n, c, 5).square().sum(-1),
            ),
            -1,
        )
        learned = segment_sum(
            self.energy_readout(invariants).squeeze(-1), batch, len(sizes)
        )
        coupling = segment_sum((p * external).sum(-1), batch, len(sizes))
        theta = (
            quadrupole_tensor(p[:, 4:])
            if cfg.density_max_l == 2
            else positions.new_zeros(n, 3, 3)
        )
        group_charge = p.new_zeros(len(totals)).index_add(0, ids, p[:, 0])
        if not torch.isfinite(p).all():
            raise FloatingPointError("Nonfinite multipole state")
        output = dict(
            delta_energy=learned + coupling,
            learned_energy=learned,
            external_energy=coupling,
            density_coefficients=p,
            charges=p[:, 0],
            atomic_dipoles=p[:, 1:4],
            atomic_quadrupoles=theta,
            density_dipole=segment_sum(
                p[:, :1] * (positions - origin) + p[:, 1:4], batch, len(sizes)
            ),
            max_charge=segment_max(p[:, 0].abs(), batch, len(sizes)),
            max_dipole=segment_max(p[:, 1:4].norm(dim=-1), batch, len(sizes)),
            max_quadrupole=segment_max(
                theta.flatten(1).norm(dim=-1), batch, len(sizes)
            ),
            charge_error=segment_max(
                (group_charge - totals).abs(), group_graph, len(sizes)
            ),
        )
        if single:
            atom_keys = {
                "density_coefficients",
                "charges",
                "atomic_dipoles",
                "atomic_quadrupoles",
            }
            output = {
                key: value if key in atom_keys else value[0]
                for key, value in output.items()
            }
        return output
