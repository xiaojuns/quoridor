"""M0 YAML contract; deliberately independent of legacy training config."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARENA_DIGEST_PATH = PROJECT_ROOT / "configs/arena.sha256"


class ConfigError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ConfigError("YAML keys must be strings")
        if key in result:
            raise ConfigError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


@dataclass(frozen=True)
class ProjectConfig:
    source_path: Path
    yaml_bytes: bytes
    canonical_json: str

    @property
    def document(self) -> dict[str, Any]:
        # Return a copy so callers cannot mutate a validated frozen snapshot.
        return json.loads(self.canonical_json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.yaml_bytes).hexdigest()

    @property
    def mode(self) -> str:
        return self.document["mode"]


def _keys(value: Any, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ConfigError(f"{label} must have exactly these keys: {sorted(expected)}")
    return value


def _integer(value: Any, label: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{label} must be an integer >= {minimum}")


def _number(value: Any, label: str, minimum: float = 0.0, *, strict: bool = True) -> None:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if (
        not finite
        or (value <= minimum if strict else value < minimum)
    ):
        relation = ">" if strict else ">="
        raise ConfigError(f"{label} must be finite and {relation} {minimum}")


def _relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigError(f"{label} must be a nonempty project-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ConfigError(f"{label} must be a nonempty project-relative path")
    try:
        inside_project = (PROJECT_ROOT / path).resolve().is_relative_to(PROJECT_ROOT)
    except (OSError, ValueError, RuntimeError) as error:
        raise ConfigError(f"invalid {label}: {error}") from error
    if not inside_project:
        raise ConfigError(f"{label} must remain within the project")
    return path


def load_project_config(path: str | Path, *, expected_mode: str | None = None) -> ProjectConfig:
    source = Path(path)
    try:
        source = source.resolve()
        raw = source.read_bytes()
        document = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, ValueError, RuntimeError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read YAML config {source}: {error}") from error

    if not isinstance(document, dict):
        raise ConfigError("config must be a YAML mapping")
    mode = document.get("mode")
    if not isinstance(mode, str) or mode not in {"smoke", "arena", "experiment"}:
        raise ConfigError("mode must be smoke, arena, or experiment")
    if expected_mode is not None and mode != expected_mode:
        raise ConfigError(f"expected mode {expected_mode}, got {mode}")
    keys = {"schema_version", "mode", "seed", "device", "rules", "budget", "output", "search"}
    _keys(document, keys | ({"arena"} if mode == "arena" else set()), "config")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ConfigError("schema_version must be integer 1")
    _integer(document["seed"], "seed", 0)
    if document["seed"] >= 2**32:
        raise ConfigError("seed must be less than 2**32")
    if not isinstance(document["device"], str) or document["device"] not in {"cpu", "cuda"}:
        raise ConfigError("device must be cpu or cuda (no implicit auto selection)")

    expected_rules = {
        "board_size": 9,
        "walls_per_player": 10,
        "action_size": 209,
        "rules_version": "quoridor-v1",
        "codec_version": "az209-v1",
        "value_perspective": "current-player-v1",
        "repetition_limit": 3,
    }
    rules = _keys(document["rules"], set(expected_rules), "rules")
    for key, value in expected_rules.items():
        if type(rules[key]) is not type(value) or rules[key] != value:
            raise ConfigError(f"rules.{key} must be {value!r}")

    budget = _keys(document["budget"], {"wall_time_seconds", "max_game_plies"}, "budget")
    _number(budget["wall_time_seconds"], "budget.wall_time_seconds")
    _integer(budget["max_game_plies"], "budget.max_game_plies")
    output = _keys(document["output"], {"run_root"}, "output")
    _relative_path(output["run_root"], "output.run_root")
    if output["run_root"] != "runs":
        raise ConfigError("output.run_root must be runs (isolated from legacy training/runs)")
    search = _keys(
        document["search"], {"simulations", "c_puct", "root_noise", "temperature"}, "search"
    )
    _integer(search["simulations"], "search.simulations")
    _number(search["c_puct"], "search.c_puct")
    _number(search["temperature"], "search.temperature", strict=False)
    if type(search["root_noise"]) is not bool:
        raise ConfigError("search.root_noise must be a boolean")

    if mode == "arena":
        arena = _keys(
            document["arena"],
            {"protocol_id", "opening_suite", "opening_suite_sha256", "pairs",
             "bootstrap_samples", "promotion_score", "promotion_confidence"},
            "arena",
        )
        if search["root_noise"] or search["temperature"] != 0:
            raise ConfigError("arena requires root_noise=false and temperature=0")
        if not isinstance(arena["protocol_id"], str) or not arena["protocol_id"].strip():
            raise ConfigError("arena.protocol_id must be nonempty")
        for key in ("pairs", "bootstrap_samples"):
            _integer(arena[key], f"arena.{key}")
        for key in ("promotion_score", "promotion_confidence"):
            _number(arena[key], f"arena.{key}", minimum=0.5)
            if arena[key] >= 1:
                raise ConfigError(f"arena.{key} must be less than 1")
        opening = PROJECT_ROOT / _relative_path(arena["opening_suite"], "arena.opening_suite")
        try:
            frozen_digest = ARENA_DIGEST_PATH.read_text(encoding="ascii").strip()
            opening_digest = hashlib.sha256(opening.read_bytes()).hexdigest()
        except (OSError, UnicodeError) as error:
            raise ConfigError(f"missing frozen arena asset: {error}") from error
        if hashlib.sha256(raw).hexdigest() != frozen_digest:
            raise ConfigError("arena config differs from configs/arena.sha256; create a reviewed new protocol")
        if opening_digest != arena["opening_suite_sha256"]:
            raise ConfigError("frozen arena opening suite SHA-256 mismatch")

    return ProjectConfig(source, raw, json.dumps(document, sort_keys=True, allow_nan=False))
