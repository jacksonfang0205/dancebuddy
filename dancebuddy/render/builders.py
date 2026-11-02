"""Registry of procedural character builders. Each returns the animation nodes named in the character's spec."""

from __future__ import annotations


def build_character(name: str, **kwargs) -> dict:
    if name == "doudou":
        from . import doudou

        return doudou.build(**kwargs)
    raise ValueError(f"no builder for character {name!r} yet")
