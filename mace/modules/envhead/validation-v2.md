# EnvHead v2 validation — 2026-09-22

**Update:** Host CUDA was subsequently verified outside the sandbox. See
[CUDA/cuEq measurements](validation-cuda.md); the initial CUDA-unavailable
observations below apply to the sandbox execution only.

## Implemented and tested

Final CPU reference tests: **129 passed, 1 CUDA skip, 2 deselected**. Two native DataLoader worker-process tests
were excluded after blocking in this sandbox; ordinary single-process loading,
native extxyz/HDF5 roundtrips and mixed batches passed. Ruff and `git diff --check` passed.

Tests cover QM forces and MM-charge derivatives by finite difference; optional
charge-loss backpropagation; full-model roundtrip; MACE and MACELES vacuum parity;
all three operator backends; rotation/translation and permutations; all-zero
charge vacuum recovery; retained zero-charge ENV chemical context; connected
three-layer outer-boundary propagation; and frozen foundation parameter behavior.

CLI E/F and E/F/q training completed epoch 1 and resumed epoch 2 with full
optimizer/scheduler state. Charge-loss weight 0.1 was restored without respecifying it. CUDA is unavailable (`torch.cuda.is_available() == False`). The vendored
Triton forward/force/response and double-backward parity test is present but skipped.
E2 is the O(3) first-order node-factorized adaptation, not the full UBio architecture.

## Real reference smoke study

26 local ORCA 6.1 HF/def2-SVP TightSCF EnGrad jobs: 18 water configurations
(three O-H lengths 0.94/0.98/1.02 A, six background scales 0/0.5/0.99/1/1.01/1.5),
and eight independent source-charge perturbations (+/-0.001 e) at held-out geometry.
Four fixed external sources are used, without PC-PC self energy. Scale-zero
frames retain zero-charge source locations and have the vacuum energy.

Eight configurations from lengths 0.94 and 0.98 A trained each model for 100 epochs;
all six scales at 1.02 A were held out. Intermediate +/-0.01 scales at training
geometries were excluded from training. Tiny randomly initialized two-layer
foundations and two-channel environment operators were used, on CPU with two
PyTorch threads. This narrow study tests the complete learning/diagnostic path;
it does not establish chemical transferability, production accuracy or backend
ranking. Labels and numerical summaries are in `validation/v2/`.

Energy RMSE is per-configuration eV; force RMSE is per Cartesian component eV/A.
ESP error is eV/e at the four existing source locations. q weight is 0.1 when
used; E/F weights are 1. No delta-q loss is used.

| Variant | E RMSE | F RMSE | predicted coupling (eV) | response ESP RMSE | q ESP RMSE | E/F inference ms |
|---|---:|---:|---:|---:|---:|---:|
| ACE E/F | 0.00734 | 0.12642 | -0.06473 | 0.2913 | 0.2558 | 24.74 |
| ACE E/F/q | 0.01826 | 0.12808 | -0.09523 | 0.3159 | 0.2857 | 26.43 |
| PaiNN E/F | 0.02103 | 0.12793 | -0.09151 | 1.0497 | 0.1487 | 18.91 |
| PaiNN E/F/q | 0.01744 | 0.11851 | -0.09240 | 1.0653 | 0.2033 | 17.97 |
| E2 first-order E/F | 0.00276 | 0.11581 | -0.05506 | 0.7318 | 0.1365 | 22.61 |
| E2 first-order E/F/q | 0.02003 | 0.12068 | -0.10166 | 0.8290 | 0.1935 | 22.63 |
| ACE last message | 0.01732 | 0.15348 | -0.03149 | 0.6502 | 0.3232 | 26.04 |
| ACE one-way | 0.00749 | 0.12613 | -0.06505 | 0.3069 | 0.2571 | 25.70 |
| ACE learned global | 0.00468 | 0.11366 | -0.03679 | 0.2451 | 0.2404 | 26.96 |

Reference coupling from scaling derivative: **-0.0508152 eV**; independently
perturbing each background charge gives **-0.0508149 eV**. Reference internal
reorganization estimated with that derivative is **+0.1055443 eV**. Mulliken
point charges reproduce the four reference potentials with RMSE **0.285070 eV/e**.

All variants reduced held-out E/F error. Improving Mulliken charge agreement did
not consistently improve response ESP or coupling energy. Projection can remove
some response-potential error, but a small projection residual alone is not a
physical accuracy check. These results support keeping charge supervision optional
and validating response separately. Only existing source locations were used;
independent spatial ESP grids, broader chemical environments, MBIS multipoles,
charge-state transfer, gauge/concavity checks and MD stability remain unvalidated.

Timing is the mean of five E/F calls after two warmups, tiny single-water CPU input.
It includes CPU neighbor construction. It is not a large-system scaling benchmark;
no GPU speed or memory improvement is claimed.

## Reproduction

```bash
python tests/extensions/envhead/v2/orca_sweep.py --orca /absolute/path/to/orca \
  --output /tmp/env-v2-orca
python tests/extensions/envhead/v2/experiment.py --orca-tree /tmp/env-v2-orca \
  --output /tmp/env-v2-experiment --epochs 100
python -m pytest tests/extensions/envhead/test_v2.py \
  tests/extensions/envhead/test_orca_converter.py tests/unit/test_data.py \
  tests/unit/test_data_utils.py -k 'not data_loader'
```

The supplied sweep script creates the 18 geometry/scale cases.
`orca_probes.py --orca /path/to/orca --output /tmp/env-v2-orca` creates the eight
individual charge perturbations; `analyze.py --orca-tree /tmp/env-v2-orca
--runs /tmp/env-v2-experiment` reproduces the ESP and inference timing diagnostics.
These scripts are alongside experiment.py. The independent
source perturbation protocol is +/-0.001 e at each of the four source charges of
g2_s1, keeping every coordinate fixed. Full local calculations and trained models
from this run remain under `/tmp/env-v2-orca` and `/tmp/env-v2-experiment-context`.
The source/energy reference is HF/def2-SVP, not the Amino dataset's production DFT.

A real Amino I/frame0044 gas/PC pair was separately converted successfully,
including Mulliken labels and environment species from original trajectory metadata.
No full Amino training or general QM accuracy claim is made.

## Existing foundation smoke inference

MACE-OFF23_small (float64), MACELES-OFF_small (float32), and DeepField_LR (float32)
loaded successfully and produced finite embedded energy, QM forces and background
charge derivatives on a water example. The added environment parameters were
untrained; this checks integration only. Values are in `validation/v2/foundations.json`.
