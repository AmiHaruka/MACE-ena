# Vendored sparse aggregation

Source: UBio-MolFM, commit `05d85f586ac949bb4c45bd9937817d276f0bf9c2`,
`src/molfm/models/e2former/triton_dr/`.

Files copied with trailing whitespace normalized; executable logic unchanged:
- `sparse_v_agg_lastdim_autograd.py`
- `triton_sparse_qk_autograd.py` (provides CSR metadata construction)

MIT license retained in LICENSE. Imported lazily by the optional E2Former
aggregation path. Kernels accumulate in float32 and include second-backward
support; v2 requires CUDA float32 and tests energy/forces/response gradients
against the eager reference. CUDA float32 forward/force/response and double-backward parity passed on the
host RTX 3060 in deepfield outside the filesystem sandbox. No standalone Triton
speed claim is made.
