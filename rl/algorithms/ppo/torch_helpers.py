"""Device and policy state-dict helpers for PPO and shared training code.

Device selection (get_device, torch_compile_available) is delegated to
rl.utils.device_utils. Policy state-dict helpers remain here as
they are PPO-specific (handle torch.compile wrapping of PPOPolicyNetwork).
"""

from rl.training_utils.device_utils import get_device, torch_compile_available


def get_policy_state_dict(policy):
    """Return a plain state dict from a policy, stripping any torch.compile prefix.

    torch.compile wraps modules in an OptimizedModule whose state_dict() keys
    all start with '_orig_mod.'.  This helper always returns plain keys so that
    saved checkpoints, pool files, and cross-process weight transfers remain
    portable regardless of whether the module was compiled.
    """
    return getattr(policy, '_orig_mod', policy).state_dict()


def load_policy_state_dict(policy, state_dict):
    """Load a state dict into a policy, handling torch.compile transparently.

    Accepts both plain keys and '_orig_mod.*' prefixed keys (backward compat
    with checkpoints saved before torch.compile was added).  Always loads into
    the underlying (uncompiled) module so keys match regardless of compilation.
    """
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}
    # Backward compatibility with unified DualHeadResNet which nests layers under 'net.'
    if not any(k.startswith('net.') for k in state_dict):
        state_dict = {f"net.{k}": v for k, v in state_dict.items()}
    getattr(policy, '_orig_mod', policy).load_state_dict(state_dict)
