"""
Harmonic mixing utilities for the OpenMythos DJ engine.

Implements the Camelot wheel used by DJs for harmonic mixing. Every musical
key maps to a Camelot code such as ``8A`` (A minor) or ``8B`` (C major). Two
tracks mix harmonically when their codes are *compatible*: identical, adjacent
on the wheel (+/- 1 with the same letter), or the relative major/minor switch
(same number, different letter).

The functions here are deliberately dependency-free so the engine runs anywhere.
"""

from __future__ import annotations

# Camelot code -> canonical key name (for display).
CAMELOT_TO_KEY: dict[str, str] = {
    "1A": "Abm", "1B": "B",
    "2A": "Ebm", "2B": "F#",
    "3A": "Bbm", "3B": "Db",
    "4A": "Fm", "4B": "Ab",
    "5A": "Cm", "5B": "Eb",
    "6A": "Gm", "6B": "Bb",
    "7A": "Dm", "7B": "F",
    "8A": "Am", "8B": "C",
    "9A": "Em", "9B": "G",
    "10A": "Bm", "10B": "D",
    "11A": "F#m", "11B": "A",
    "12A": "Dbm", "12B": "E",
}

# Reverse + alias map: many spellings of the same key -> Camelot code.
_KEY_TO_CAMELOT: dict[str, str] = {}


def _register(code: str, *names: str) -> None:
    for n in names:
        _KEY_TO_CAMELOT[n.lower()] = code


# Minor keys (A) and their enharmonic spellings.
_register("1A", "abm", "g#m", "ab minor", "g# minor")
_register("2A", "ebm", "d#m", "eb minor", "d# minor")
_register("3A", "bbm", "a#m", "bb minor", "a# minor")
_register("4A", "fm", "f minor")
_register("5A", "cm", "c minor")
_register("6A", "gm", "g minor")
_register("7A", "dm", "d minor")
_register("8A", "am", "a minor")
_register("9A", "em", "e minor")
_register("10A", "bm", "b minor")
_register("11A", "f#m", "gbm", "f# minor", "gb minor")
_register("12A", "dbm", "c#m", "db minor", "c# minor")
# Major keys (B).
_register("1B", "b", "b major")
_register("2B", "f#", "gb", "f# major", "gb major")
_register("3B", "db", "c#", "db major", "c# major")
_register("4B", "ab", "g#", "ab major", "g# major")
_register("5B", "eb", "d#", "eb major", "d# major")
_register("6B", "bb", "a#", "bb major", "a# major")
_register("7B", "f", "f major")
_register("8B", "c", "c major")
_register("9B", "g", "g major")
_register("10B", "d", "d major")
_register("11B", "a", "a major")
_register("12B", "e", "e major")


def to_camelot(key: str | None) -> str | None:
    """Normalize a key string to its Camelot code, or ``None`` if unknown.

    Accepts Camelot codes directly (``"8A"``, ``"8a"``) and common key
    spellings (``"Am"``, ``"A minor"``, ``"C"``, ``"F#m"``).
    """
    if not key:
        return None
    k = key.strip()
    upper = k.upper()
    if upper in CAMELOT_TO_KEY:
        return upper
    return _KEY_TO_CAMELOT.get(k.lower())


def _parse(code: str) -> tuple[int, str] | None:
    if not code or len(code) < 2:
        return None
    letter = code[-1].upper()
    if letter not in ("A", "B"):
        return None
    try:
        number = int(code[:-1])
    except ValueError:
        return None
    if not 1 <= number <= 12:
        return None
    return number, letter


def compatibility(a: str | None, b: str | None) -> float:
    """Score harmonic compatibility of two keys in ``[0.0, 1.0]``.

    * ``1.0``  same key (perfect blend / energy hold)
    * ``0.9``  relative major/minor (same number, flipped letter)
    * ``0.85`` adjacent on the wheel (+/- 1, same letter)
    * ``0.5``  two steps away (mixable with care)
    * ``0.2``  everything else (clash)
    * ``0.5``  if either key is unknown (neutral, don't over-penalize)
    """
    ca, cb = to_camelot(a), to_camelot(b)
    if ca is None or cb is None:
        return 0.5
    pa, pb = _parse(ca), _parse(cb)
    if pa is None or pb is None:
        return 0.5
    (na, la), (nb, lb) = pa, pb
    if na == nb and la == lb:
        return 1.0
    if na == nb and la != lb:
        return 0.9
    # Circular distance around the 12-hour wheel.
    dist = min((na - nb) % 12, (nb - na) % 12)
    if la == lb and dist == 1:
        return 0.85
    if dist == 2:
        return 0.5
    return 0.2


def compatible_codes(code: str) -> list[str]:
    """Return the Camelot codes that mix cleanly with ``code``."""
    p = _parse(to_camelot(code) or "")
    if p is None:
        return []
    n, letter = p
    other = "B" if letter == "A" else "A"
    up = (n % 12) + 1
    down = ((n - 2) % 12) + 1
    return [f"{n}{letter}", f"{n}{other}", f"{up}{letter}", f"{down}{letter}"]
