"""Injection factory. Stage 4.3 (P4.3).

Currently only LTIInjection is available. This factory exists so that
alternative injection schemes (identity, gated, SSM) can be plugged in
without touching RecurrentBlock.
"""
from __future__ import annotations

from open_mythos.config import MythosConfig
from open_mythos.recurrence.lti import LTIInjection
from open_mythos.recurrence.protocol import InjectionProtocol


def build_injection(cfg: MythosConfig) -> InjectionProtocol:
    """Return an injection module for the recurrent loop."""
    # Stage 4.3: only LTI is implemented. Future alternates dispatch here.
    return LTIInjection(cfg.dim)
