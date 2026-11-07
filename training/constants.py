"""Training loop constants.

MythosConfig is frozen, so training-specific toggles live here as module-level
constants. Flip USE_GRAD_CHECKPOINT=True to wrap RecurrentBlock forwards with
torch.utils.checkpoint.checkpoint(..., use_reentrant=False) for memory savings
at the cost of a recompute pass. Actual wiring at the forward call site is
deferred because FSDP + activation-checkpoint interaction needs validation
before it's turned on by default.
"""

USE_GRAD_CHECKPOINT: bool = False
