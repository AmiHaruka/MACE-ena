"""Copy into your own module, implement and explicitly register an adapter.

Load it with --adapter-module my_package.my_adapter before loading checkpoints.
This file intentionally does not register PolarMACE or any unverified backend.
"""
from .native import NativeMACEAdapter


class UserFoundationAdapter(NativeMACEAdapter):
    """Template for foundations that expose native MACE graphs and blocks.

    Instantiate with name='my-foundation-v1', model_type=YourExactModelClass,
    then register_adapter(instance). Do not use broad isinstance dispatch.

    Required contract:
    - owner preserves all foundation state_dict keys and metadata;
    - interaction(owner) describes the native pre-product irreps/layout and
      node/edge attrs required to construct EnvHead's MACE blocks;
    - evaluate returns (output_dict, differentiable_pre_product_tensor), with
      output_dict['energy'] the FULL vacuum foundation energy in eV per graph;
    - never consume MM sources or externally supplied fields in the foundation;
    - preserve coordinate gradients through every short/long-range energy term;
    - native TensorDict/AtomicData, radial_embedding and spherical_harmonics
      remain required by EnvMACE. Other architectures need an explicit graph
      and feature bridge as well; this template is not a universal wrapper.

    PolarMACE needs a reviewed zero-external-field charge/spin contract, its
    complete electrostatic energy, and an explicit feature-layout mapping.
    Its latent multipoles must not be substituted for EnvHead's variational
    charges. Unknown capabilities must raise rather than return zeros.
    """

    def validate(self, foundation):
        raise NotImplementedError("Define foundation capabilities and energy ownership")

    def evaluate(self, owner, data):
        raise NotImplementedError(
            "Implement and verify complete energy plus feature tap"
        )
