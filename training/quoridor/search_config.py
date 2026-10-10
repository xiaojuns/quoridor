"""Map an already validated YAML snapshot to the versioned PUCT contract."""

from __future__ import annotations

from .project_config import ProjectConfig
from .puct import SearchConfig


def search_config_from_project(config: ProjectConfig) -> SearchConfig:
    """Use the snapshot's search fields and seed without tuning overrides.

    Arena uses deterministic evaluation. Smoke and experiment use self-play
    semantics, with root noise and temperature still explicitly set by YAML.
    The v1 self-play Dirichlet defaults are alpha=0.3 and fraction=0.25; they
    are search-version defaults, not edits to the frozen arena YAML.
    Budget ownership remains with the caller's shared runtime Deadline.
    """

    if type(config) is not ProjectConfig:
        raise TypeError("config must be a validated ProjectConfig")
    document = config.document
    mode = document["mode"]
    if mode not in {"arena", "smoke", "experiment"}:
        raise ValueError("unsupported project mode")
    search = document["search"]
    return SearchConfig(
        simulations=search["simulations"],
        c_puct=search["c_puct"],
        mode="evaluation" if mode == "arena" else "selfplay",
        root_noise=search["root_noise"],
        temperature=search["temperature"],
        seed=document["seed"],
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
    )


__all__ = ["search_config_from_project"]
