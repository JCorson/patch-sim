"""Shared Traub-Miles (1991) analytical rate-function helpers.

All four cell-type-specific families (Pospischil, Thalamic Relay, TRN,
Purkinje, SNc Dopaminergic) use the same six Traub-Miles analytical forms —
they differ only in the voltage threshold parameter ``vt``.  These private
helpers encode each form once; the public wrappers delegate to them with the
appropriate VT constant.
"""

from ..utils import safe_exp

# Threshold for detecting near-singularity in GHK-style rate equations.
# When the denominator voltage term is within this tolerance of zero, the
# L'Hôpital limit is used instead to avoid division by zero.
SINGULARITY_THRESHOLD: float = 1e-6


def _traub_miles_alpha_m(V: float, vt: float) -> float:
    """Traub-Miles forward rate for Na⁺ activation gate m.

    Has a removable singularity at V = vt + 13; the L'Hôpital limit (1.28)
    is returned when ``|V − vt − 13| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Forward rate in 1/ms.
    """
    x = V - vt - 13
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.28
    return -0.32 * x / (safe_exp(-x / 4) - 1)


def _traub_miles_beta_m(V: float, vt: float) -> float:
    """Traub-Miles backward rate for Na⁺ activation gate m.

    Has a removable singularity at V = vt + 40; the L'Hôpital limit (1.4)
    is returned when ``|V − vt − 40| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Backward rate in 1/ms.
    """
    x = V - vt - 40
    if abs(x) < SINGULARITY_THRESHOLD:
        return 1.4
    return 0.28 * x / (safe_exp(x / 5) - 1)


def _traub_miles_alpha_h(V: float, vt: float) -> float:
    """Traub-Miles forward rate for Na⁺ inactivation gate h.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Forward rate in 1/ms.
    """
    return 0.128 * safe_exp(-(V - vt - 17) / 18)


def _traub_miles_beta_h(V: float, vt: float) -> float:
    """Traub-Miles backward rate for Na⁺ inactivation gate h.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Backward rate in 1/ms.
    """
    return 4.0 / (1 + safe_exp(-(V - vt - 40) / 5))


def _traub_miles_alpha_n(V: float, vt: float) -> float:
    """Traub-Miles forward rate for K⁺ delayed-rectifier activation gate n.

    Has a removable singularity at V = vt + 15; the L'Hôpital limit (0.16)
    is returned when ``|V − vt − 15| < SINGULARITY_THRESHOLD``.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Forward rate in 1/ms.
    """
    x = V - vt - 15
    if abs(x) < SINGULARITY_THRESHOLD:
        return 0.16
    return -0.032 * x / (safe_exp(-x / 5) - 1)


def _traub_miles_beta_n(V: float, vt: float) -> float:
    """Traub-Miles backward rate for K⁺ delayed-rectifier activation gate n.

    Args:
        V: Membrane voltage in mV.
        vt: Cell-type voltage threshold in mV.

    Returns:
        Backward rate in 1/ms.
    """
    return 0.5 * safe_exp(-(V - vt - 10) / 40)
