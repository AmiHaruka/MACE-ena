# EnvHead v2: implementation and physical contracts

Date: 2026-09-22. v1 code is replaced; Git commit `df05a17` preserves the old implementation.

## Energy ownership and execution

`EnvMACE` owns a copied native MACE/ScaleShiftMACE/MACELES foundation and trains it
jointly by default. `freeze_foundation` freezes parameters without detaching
geometry. A single native forward is instrumented with scoped interaction and
product hooks, removed in `finally`. Each interaction receives ENV messages
before its existing ACE product. Native readouts, E0 and LES sources see QM only;
the native LES energy is included once. No learned-charge Coulomb energy is added.

The model predicts embedded QM energy: internal QM energy in the polarized state
plus QM/background coupling, excluding background self energy. All QM coordinate
paths participate in autograd. Current implementation is eager, nonperiodic,
single energy head and singlet. Concurrent calls use separate model instances.

## Graph and interchangeable operators

`graph.py` selects ENV nodes within `foundation.r_max + r_env` of any QM atom;
`r_env=5 A`. Message edges retain the foundation cutoff. A per-graph cKDTree builds
one detached topology on CPU; selected geometry remains differentiable. It does
not allocate a dense all-node pair matrix. QM/all-MM selection is chunked.

ENV node support is a smooth union `1-prod_i(1-c_i)`, where each `c_i` is one
inside 75% of the coverage radius and smoothly vanishes at its boundary. A C2
quintic envelope also multiplies each edge. Radial modulation is independent:
constant/cosine/polynomial/mollifier/switch/bell, plus baseline/amplitude and
separate epsilon gain. Boundary support remains even for constant modulation.

ENV->QM messages carry the source charge. ENV->ENV messages retain elemental
and geometric context even at zero charge, as explicitly requested. A single
zero-charge ENV node may therefore influence charged neighbors; insertion
invariance is not imposed. All-zero background charges recover the vacuum path.
QM->ENV messages can be disabled. Atomic identity has an independent ENV embedding.
Node features do not imply dynamical MM charges or explicit environmental density.

ACE uses native MACE products after equivariant spherical convolution. PaiNN uses
scalar/vector messages and scalar-vector mixing. E2 uses the first-order identity
`T(r_i,sum_j a_ij h_j)-sum_j a_ij T(r_j,h_j)` with node tensor products and smooth
scalar attention. Explicit natural parity is retained. It is an adapted E2
operator, not the complete upstream architecture; no full Wigner6j/SO2 performance
claim is made. Optional UBio sparse Triton aggregation is lazy-loaded. Its kernels
use float32 and support a second backward in source; GPU forward and second-backward parity has now passed on RTX 3060; see
validation-cuda.md for the tested scope. License and exact source pointers are in `modules/triton_ops/`.

## Features and global response

Each native product's actual irreps are projected into common irreps and summed;
no scalar-only last layer is silently promoted to higher angular information.
`last_message` selects a final pre-product feature alternative. Scalar Q/N
conditioning enters native messages. Depth comes from the foundation.

Global modes are all-source physical phi/E or learned radial-angular projection.
The latter uses learned functions of reciprocal distance, charge weighting and
spherical harmonics, summed in chunks. Global features couple with QM features
through invariant scalar products and norms. Its final residual subtracts the
same readout at zero global descriptor, so zero sources recover the corresponding
vacuum path. Near-conditioned native energy is not called vacuum base energy.

## Response-defined diagnostics

For background charges b, compute V=dE/db only when requested. Define
`interaction_energy=sum(b*V)` and `internal_energy=E-interaction_energy`.
The identity corresponds to a Hamiltonian linear in background charges and a
stationary electronic state. For an approximate learned E, these are predictions
whose accuracy must be assessed independently. Gauge response, ground-state
concavity and nonnegative internal reorganization are not guaranteed by this
network architecture. Matching E/F at a few environments alone does not validate
dE/db. There is no factor 1/2 in the final-state cross interaction.

`response.py` projects V onto QM point charges with exact total-charge conservation.
The data term is normalized by the RMS Coulomb design matrix; a fixed Gaussian
Coulomb hardness matrix (including its analytic self term) plus positive atomic
hardness regularizes around Q/N. The dimensionless prior is a reconstruction
constraint, not an additional physical energy. Width=1 A, hardness=1 eV/e^2,
regularization=1e-4 are configurable numerical defaults, not fitted elemental
hardnesses. Diagnostics expose residual and charge-conserving design condition.
Fewer than N_QM-1 independent samples or condition >1e8 invalidate reconstruction.
Gas without sample locations remains valid for E/F and invalid for charges.
No invented zero-strength probe protocol or SQE bond topology is included.

## Data, training and extension boundaries

Native Configuration/extxyz/HDF5 carry total_charge, optional REF_charges and MM
positions/charges/atomic numbers. No custom charge key or charge-scheme field.
Mulliken extraction is standalone; MBIS can replace labels in prepared datasets.
Missing charge labels have zero weight. E/F/q loss uses valid reconstructions
only, and differentiates through energy response and KKT solve. No delta-q loss.

Training saves full models, optimizers, schedulers and RNG state. Source-group
splits avoid paired-frame leakage. v1 head-only checkpoints, legacy cuEq cache
adapters and ML/MM paths have been removed. Periodic embedding, dynamics, SCF,
SQE transfer graphs, higher density moments and gas probe grids remain extensions.

## Verification

Numerical tests cover data roundtrips, QM ownership, MACE/LES parity in neutral
vacuum, finite difference force/response, symmetry, mixed batch, charge gradients,
zero-charge chemical context, boundary smoothness and full-model checkpoints. The CUDA
parity test is skipped when CUDA is unavailable. Scientific smoke experiments
and limitations are recorded in validation-v2.md; runtime tests are not evidence
of production accuracy or a unique atomic charge interpretation.

## cuEq update

`backends.py` converts native/ENV ACE symmetric contractions with native CG weight
mapping and ENV ACE/E2 tensor products with `method="fused_tp"`. Parameters are
owned directly by the accelerated modules; convert before optimizer creation.
Foundation convolutions and linear layers remain in e3nn. Full-model checkpoints
retain backend state. CUDA float32 parity, force/response mixed gradients and
optimizer updates were tested on RTX 3060 outside the sandbox. See
validation-cuda.md; the earlier sandbox-only CUDA observation was not a host
hardware limitation.
