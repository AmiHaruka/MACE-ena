# EnvHead v2

Unified QM/environment messages condition native MACE energy readouts. Only QM
nodes own energy; the target includes QM/background interaction, excludes MM
self energy. Forces are derivatives of the complete predicted energy.

## Data

QM atoms use native MACE `REF_energy`, `REF_forces`, `total_charge` and optional
`REF_charges`. `atoms.info` holds `mm_positions`, `mm_charges`,
`mm_atomic_numbers`. Units are eV, Angstrom and elementary charge. Empty MM
arrays represent gas configurations. Atomic-charge labels are never inputs.

```bash
python mace/modules/envhead/script/orca.py /path/to/outputs \
  --pc-dir /path/to/Traj2 --output /tmp/training.extxyz
```

The ORCA converter extracts Mulliken charges and reads environment elements from
`frame.elements.json` alongside `frame.pc`, or the original `params.json`
trajectory metadata. Explicit sidecars contain a JSON array of element symbols
or atomic numbers in PC order. Missing metadata is not guessed from charges;
v2 requires it for embedded inputs. ORCA labels retain their existing energy
convention; `DoEQ` is rejected. `REF_charges` can later contain MBIS instead;
prepare each dataset consistently without mixing charge definitions.

## Training

Example configuration (`config.json`):

```json
{"channels":16,"backend":"ace","r_env":5.0,"bidirectional":true,
 "feature_source":"mixed_products","global_mode":"physical",
 "freeze_foundation":false}
```

```bash
python -m mace.modules.envhead.train --foundation /path/to/foundation.model \
  --train-file /path/to/training.extxyz --head-config config.json \
  --output-dir /tmp/envhead-run --device cuda --dtype float32 --enable-cueq --epochs 50
```

Use `--charges-weight 0.1` to enable optional charge supervision; default is zero.
The converter writes source `group_id` for Unit/frame naming. Automatic splitting
requires all group IDs; otherwise supply an explicitly separated `--valid-file`.
Gas/PC pairs and the same trajectory must remain in the same split.

Resume with `--resume /tmp/envhead-run/last.pt`, retaining the training options.
Checkpoints contain the full model, configuration, optimizer, scheduler and RNG
state. Load only trusted model files. v1 head-only checkpoints are incompatible.

Backends: `ace`, `painn`, `e2former`. The last is an O(3) first-order
node-factorized adaptation of the E2Former identity, not the full UBio model or
its Wigner/SO2 implementation. Optional `triton: true` uses vendored UBio sparse
aggregation (CUDA float32 only); forward and second-backward parity passed on
RTX 3060 with the deepfield environment outside the filesystem sandbox.

## cuEq acceleration

`--enable-cueq --device cuda --dtype float32` converts native and ENV ACE symmetric
contractions, plus ENV ACE/E2 fully connected tensor products (`fused_tp`). These
are live trainable modules, converted before optimizer creation. There is no
frozen duplicate model/cache. Foundation convolutions, linears, PaiNN scalar/vector
operations, graph construction and electrostatics retain their existing paths.

Programmatic: `enable_cueq(model.cuda())` from `mace.modules.envhead.backends`, before
creating an optimizer. ASE: `EnvHeadCalculator(model, device="cuda", enable_cueq=True)`.
Converted full checkpoints are loaded on CUDA; do not switch backend mid-resume.
A config with `cueq: true` requires constructing EnvMACE from a CUDA foundation.

On one Amino I frame, float32/default 16 ENV channels/RTX 3060:
E/F 319.6 -> 54.9 ms (**5.82x**), force-training forward/backward
568.4 -> 141.5 ms (**4.02x**).
This is a single-frame end-to-end benchmark, not a universal speed guarantee.
Details and tolerances: [CUDA validation](validation-cuda.md).

## Tensor and ASE inference

```python
from mace.modules.envhead.train.checkpoint import load_checkpoint
from mace.modules.envhead.runtime import EnvRuntime
from mace.modules.envhead.calculator import EnvHeadCalculator

model = load_checkpoint('/tmp/envhead-run/last.pt')['model'].eval()
result = EnvRuntime(model)(
    atomic_numbers, positions, mm_positions, mm_charges, mm_atomic_numbers,
    total_charge=0, compute_response=True, compute_charges=True,
)
atoms.calc = EnvHeadCalculator(model)
energy, forces = atoms.get_potential_energy(), atoms.get_forces()
```

Supply the environment arrays in `atoms.info` for ASE. Results optionally include
`response_potential`, `interaction_energy`, `internal_energy`, `charges`,
`charges_valid`, `charge_esp_rmse`, `charge_condition`. Gas or underdetermined
charge reconstructions return invalid status and NaN charges; E/F remain valid.
ASE `get_charges()` raises when reconstruction is invalid. Singlets and
nonperiodic systems are supported. Current scoped forward hooks require a
separate model instance for concurrent forwards.

## Validation

```bash
python -m pytest tests/extensions/envhead/test_v2.py \
  tests/extensions/envhead/test_orca_converter.py
```

See [architecture](ARCHITECTUREv2.md) and [validation](validation-v2.md).
