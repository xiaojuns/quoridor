"""Immutable Phase 4E1 CUDA calibration contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping

from .run_manifest import canonical_json_bytes, sha256_bytes


PHASE4E1_CONTRACT_SCHEMA_VERSION = 1
PHASE4E1_CONTRACT_KIND = "phase4e1-calibration-contract"
PHASE4E1_STAGE = "phase4e1"
PHASE4E1_STATUS = "confirmed"

GLOBAL_GPU_HOUR_LIMIT = 72.0
INITIAL_GLOBAL_GPU_HOURS = 8.459698904922698
INITIAL_GPU_LEDGER_SHA256 = (
    "8b1e870fc183a4819276e97430323213e800ff322667c3d4046d9663993fd5bd"
)
CALIBRATION_GPU_HOUR_LIMIT = 6.0
ARTIFACT_LIMIT_BYTES = 8 * 1024**3
MINIMUM_FREE_DISK_BYTES = 20 * 1024**3

PREPARED_EXECUTION_CONTRACT_SHA256 = (
    "636ecdd40dc999614963e8f64eb7e58e533dc5302b7b2adb0e7ebaa46a8f3d87"
)
BOOTSTRAP_CONFIG_SEMANTIC_SHA256 = (
    "af10dd79d2b87002419b11d277213f98e85991e5adb1d799eb3da7d010ce715e"
)
STANDARD_CONFIG_SEMANTIC_SHA256 = (
    "93a208a1b9b14ec8052fd10edaf3dbf7a614b8ab6ff78d5e5765a6d71efbc68d"
)
BASELINE_MANIFEST_SELF_SHA256 = (
    "dc1c33619997732e5525840d1c867969ce82ede6d7ecb25092e962f63dd15294"
)
BASELINE_MODEL_FILE_SHA256 = (
    "8db99ac64c5099bdf0df787350150e36c1b3c2343a826526522e027057cc8124"
)
DEV_SUITE_HASH = (
    "sha256:9cde8e3302916d023957b60d4a433f31f0dfc995881389aca95daee4730b11c4"
)
ARENA_SUITE_HASH = (
    "sha256:8c15fa0f1b006dfcc32a7e4a46028e744be4291f9bc56e4172c098d67f74252d"
)

SELFPLAY_GAMES = 128
SELFPLAY_ACTORS = 64
SELFPLAY_SIMULATIONS = 512
SELFPLAY_TEMPERATURE_MOVES = 24
ARENA_OPENING_PREFIX_COUNT = 40
ARENA_GAMES = 80
ARENA_ACTORS = 64
ARENA_SIMULATIONS = 800
WORKLOAD_SEED = 20_260_714

FREEZE_RESERVE_GPU_HOURS = 2.0
FREEZE_MINIMUM_STAGE_GPU_HOURS = 48.0
FREEZE_MAXIMUM_GENERATIONS = 64


class Phase4E1ContractError(ValueError):
    """Raised when the reviewed Phase 4E1 contract is malformed or changed."""


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4E1ContractError(f"{label} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise Phase4E1ContractError(f"{label} fields are invalid")


def _fixed(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise Phase4E1ContractError(f"{label} differs from the frozen contract")


def _finite_hours(value: object, label: str, *, positive: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        or (positive and float(value) <= 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise Phase4E1ContractError(f"{label} must be finite and {qualifier}")
    return float(value)


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise Phase4E1ContractError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Phase4E1ContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _suite_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise Phase4E1ContractError(f"{label} must be a content-addressed SHA-256")
    _sha256(value.removeprefix("sha256:"), label)
    return value


def _workspace_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Phase4E1ContractError(
            f"{label} must be a canonical workspace-relative path"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise Phase4E1ContractError(
            f"{label} must be a canonical workspace-relative path without '..'"
        )
    return value


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    path: str
    total_gpu_hours: float
    ledger_sha256: str

    @classmethod
    def from_dict(cls, value: object) -> "LedgerSnapshot":
        data = _mapping(value, "resources.initialLedger")
        _exact_fields(
            data,
            {"path", "totalGpuHours", "ledgerSha256"},
            "resources.initialLedger",
        )
        result = cls(
            path=_workspace_relative_path(
                data.get("path"), "resources.initialLedger.path"
            ),
            total_gpu_hours=_finite_hours(
                data.get("totalGpuHours"),
                "resources.initialLedger.totalGpuHours",
                positive=False,
            ),
            ledger_sha256=_sha256(
                data.get("ledgerSha256"),
                "resources.initialLedger.ledgerSha256",
            ),
        )
        _fixed(
            result.path,
            "training/runs/phase4/gpu-budget-ledger.json",
            "resources.initialLedger.path",
        )
        _fixed(
            result.total_gpu_hours,
            INITIAL_GLOBAL_GPU_HOURS,
            "resources.initialLedger.totalGpuHours",
        )
        _fixed(
            result.ledger_sha256,
            INITIAL_GPU_LEDGER_SHA256,
            "resources.initialLedger.ledgerSha256",
        )
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "totalGpuHours": self.total_gpu_hours,
            "ledgerSha256": self.ledger_sha256,
        }


@dataclass(frozen=True, slots=True)
class ResourceContract:
    global_gpu_hour_limit: float
    calibration_gpu_hour_limit: float
    artifact_limit_bytes: int
    minimum_free_disk_bytes: int
    initial_ledger: LedgerSnapshot

    @classmethod
    def from_dict(cls, value: object) -> "ResourceContract":
        data = _mapping(value, "resources")
        _exact_fields(
            data,
            {
                "globalGpuHourLimit",
                "calibrationGpuHourLimit",
                "artifactLimitBytes",
                "minimumFreeDiskBytes",
                "initialLedger",
            },
            "resources",
        )
        result = cls(
            global_gpu_hour_limit=_finite_hours(
                data.get("globalGpuHourLimit"),
                "resources.globalGpuHourLimit",
                positive=True,
            ),
            calibration_gpu_hour_limit=_finite_hours(
                data.get("calibrationGpuHourLimit"),
                "resources.calibrationGpuHourLimit",
                positive=True,
            ),
            artifact_limit_bytes=_positive_int(
                data.get("artifactLimitBytes"), "resources.artifactLimitBytes"
            ),
            minimum_free_disk_bytes=_positive_int(
                data.get("minimumFreeDiskBytes"),
                "resources.minimumFreeDiskBytes",
            ),
            initial_ledger=LedgerSnapshot.from_dict(data.get("initialLedger")),
        )
        _fixed(
            result.global_gpu_hour_limit,
            GLOBAL_GPU_HOUR_LIMIT,
            "resources.globalGpuHourLimit",
        )
        _fixed(
            result.calibration_gpu_hour_limit,
            CALIBRATION_GPU_HOUR_LIMIT,
            "resources.calibrationGpuHourLimit",
        )
        _fixed(
            result.artifact_limit_bytes,
            ARTIFACT_LIMIT_BYTES,
            "resources.artifactLimitBytes",
        )
        _fixed(
            result.minimum_free_disk_bytes,
            MINIMUM_FREE_DISK_BYTES,
            "resources.minimumFreeDiskBytes",
        )
        if result.initial_ledger.total_gpu_hours >= result.global_gpu_hour_limit:
            raise Phase4E1ContractError(
                "initial GPU ledger total must remain below the global limit"
            )
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "globalGpuHourLimit": self.global_gpu_hour_limit,
            "calibrationGpuHourLimit": self.calibration_gpu_hour_limit,
            "artifactLimitBytes": self.artifact_limit_bytes,
            "minimumFreeDiskBytes": self.minimum_free_disk_bytes,
            "initialLedger": self.initial_ledger.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class ProvenanceContract:
    prepared_execution_contract_path: str
    prepared_execution_contract_sha256: str
    bootstrap_config_path: str
    bootstrap_config_sha256: str
    standard_config_path: str
    standard_config_sha256: str
    baseline_manifest_path: str
    baseline_manifest_sha256: str
    baseline_model_path: str
    baseline_model_sha256: str
    dev_suite_path: str
    dev_suite_hash: str

    @classmethod
    def from_dict(cls, value: object) -> "ProvenanceContract":
        data = _mapping(value, "provenance")
        _exact_fields(
            data,
            {
                "preparedExecutionContract",
                "bootstrapConfig",
                "standardConfig",
                "bootstrapBaseline",
                "devOpeningSuite",
            },
            "provenance",
        )
        prepared = _mapping(
            data.get("preparedExecutionContract"),
            "provenance.preparedExecutionContract",
        )
        bootstrap = _mapping(
            data.get("bootstrapConfig"), "provenance.bootstrapConfig"
        )
        standard = _mapping(
            data.get("standardConfig"), "provenance.standardConfig"
        )
        baseline = _mapping(
            data.get("bootstrapBaseline"), "provenance.bootstrapBaseline"
        )
        dev_suite = _mapping(
            data.get("devOpeningSuite"), "provenance.devOpeningSuite"
        )
        _exact_fields(
            prepared,
            {"path", "canonicalSha256"},
            "provenance.preparedExecutionContract",
        )
        _exact_fields(
            bootstrap,
            {"path", "semanticSha256"},
            "provenance.bootstrapConfig",
        )
        _exact_fields(
            standard,
            {"path", "semanticSha256"},
            "provenance.standardConfig",
        )
        _exact_fields(
            baseline,
            {
                "manifestPath",
                "manifestSelfSha256",
                "modelPath",
                "modelFileSha256",
            },
            "provenance.bootstrapBaseline",
        )
        _exact_fields(
            dev_suite,
            {"path", "suiteHash"},
            "provenance.devOpeningSuite",
        )
        result = cls(
            prepared_execution_contract_path=_workspace_relative_path(
                prepared.get("path"),
                "provenance.preparedExecutionContract.path",
            ),
            prepared_execution_contract_sha256=_sha256(
                prepared.get("canonicalSha256"),
                "provenance.preparedExecutionContract.canonicalSha256",
            ),
            bootstrap_config_path=_workspace_relative_path(
                bootstrap.get("path"), "provenance.bootstrapConfig.path"
            ),
            bootstrap_config_sha256=_sha256(
                bootstrap.get("semanticSha256"),
                "provenance.bootstrapConfig.semanticSha256",
            ),
            standard_config_path=_workspace_relative_path(
                standard.get("path"), "provenance.standardConfig.path"
            ),
            standard_config_sha256=_sha256(
                standard.get("semanticSha256"),
                "provenance.standardConfig.semanticSha256",
            ),
            baseline_manifest_path=_workspace_relative_path(
                baseline.get("manifestPath"),
                "provenance.bootstrapBaseline.manifestPath",
            ),
            baseline_manifest_sha256=_sha256(
                baseline.get("manifestSelfSha256"),
                "provenance.bootstrapBaseline.manifestSelfSha256",
            ),
            baseline_model_path=_workspace_relative_path(
                baseline.get("modelPath"),
                "provenance.bootstrapBaseline.modelPath",
            ),
            baseline_model_sha256=_sha256(
                baseline.get("modelFileSha256"),
                "provenance.bootstrapBaseline.modelFileSha256",
            ),
            dev_suite_path=_workspace_relative_path(
                dev_suite.get("path"), "provenance.devOpeningSuite.path"
            ),
            dev_suite_hash=_suite_hash(
                dev_suite.get("suiteHash"),
                "provenance.devOpeningSuite.suiteHash",
            ),
        )
        expected = {
            "prepared_execution_contract_path": (
                "training/configs/phase4e-execution-prepared-v1.json"
            ),
            "prepared_execution_contract_sha256": (
                PREPARED_EXECUTION_CONTRACT_SHA256
            ),
            "bootstrap_config_path": (
                "training/configs/phase4e-bootstrap-v1.json"
            ),
            "bootstrap_config_sha256": BOOTSTRAP_CONFIG_SEMANTIC_SHA256,
            "standard_config_path": "training/configs/phase4e-standard-v1.json",
            "standard_config_sha256": STANDARD_CONFIG_SEMANTIC_SHA256,
            "baseline_manifest_path": (
                "training/runs/phase4/baselines/bootstrap-warm-v2/manifest.json"
            ),
            "baseline_manifest_sha256": BASELINE_MANIFEST_SELF_SHA256,
            "baseline_model_path": (
                "training/runs/phase4/baselines/bootstrap-warm-v2/model.pt"
            ),
            "baseline_model_sha256": BASELINE_MODEL_FILE_SHA256,
            "dev_suite_path": "training/fixtures/openings/dev.json",
            "dev_suite_hash": DEV_SUITE_HASH,
        }
        for field, expected_value in expected.items():
            _fixed(
                getattr(result, field),
                expected_value,
                f"provenance.{field}",
            )
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "preparedExecutionContract": {
                "path": self.prepared_execution_contract_path,
                "canonicalSha256": self.prepared_execution_contract_sha256,
            },
            "bootstrapConfig": {
                "path": self.bootstrap_config_path,
                "semanticSha256": self.bootstrap_config_sha256,
            },
            "standardConfig": {
                "path": self.standard_config_path,
                "semanticSha256": self.standard_config_sha256,
            },
            "bootstrapBaseline": {
                "manifestPath": self.baseline_manifest_path,
                "manifestSelfSha256": self.baseline_manifest_sha256,
                "modelPath": self.baseline_model_path,
                "modelFileSha256": self.baseline_model_sha256,
            },
            "devOpeningSuite": {
                "path": self.dev_suite_path,
                "suiteHash": self.dev_suite_hash,
            },
        }


@dataclass(frozen=True, slots=True)
class SelfPlayWorkload:
    channels: int
    residual_blocks: int
    games: int
    actors: int
    simulations: int
    temperature_moves: int
    seed: int

    @classmethod
    def from_dict(cls, value: object) -> "SelfPlayWorkload":
        data = _mapping(value, "workloads.selfPlay")
        _exact_fields(
            data,
            {
                "network",
                "games",
                "actors",
                "simulations",
                "temperatureMoves",
                "seed",
            },
            "workloads.selfPlay",
        )
        network = _mapping(data.get("network"), "workloads.selfPlay.network")
        _exact_fields(
            network,
            {"channels", "residualBlocks"},
            "workloads.selfPlay.network",
        )
        result = cls(
            channels=_positive_int(
                network.get("channels"), "workloads.selfPlay.network.channels"
            ),
            residual_blocks=_positive_int(
                network.get("residualBlocks"),
                "workloads.selfPlay.network.residualBlocks",
            ),
            games=_positive_int(data.get("games"), "workloads.selfPlay.games"),
            actors=_positive_int(data.get("actors"), "workloads.selfPlay.actors"),
            simulations=_positive_int(
                data.get("simulations"), "workloads.selfPlay.simulations"
            ),
            temperature_moves=_positive_int(
                data.get("temperatureMoves"),
                "workloads.selfPlay.temperatureMoves",
            ),
            seed=_positive_int(data.get("seed"), "workloads.selfPlay.seed"),
        )
        expected = {
            "channels": 192,
            "residual_blocks": 16,
            "games": SELFPLAY_GAMES,
            "actors": SELFPLAY_ACTORS,
            "simulations": SELFPLAY_SIMULATIONS,
            "temperature_moves": SELFPLAY_TEMPERATURE_MOVES,
            "seed": WORKLOAD_SEED,
        }
        for field, expected_value in expected.items():
            _fixed(
                getattr(result, field),
                expected_value,
                f"workloads.selfPlay.{field}",
            )
        if result.games < result.actors * 2:
            raise Phase4E1ContractError(
                "self-play must run at least two games per configured actor"
            )
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "network": {
                "channels": self.channels,
                "residualBlocks": self.residual_blocks,
            },
            "games": self.games,
            "actors": self.actors,
            "simulations": self.simulations,
            "temperatureMoves": self.temperature_moves,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ArenaWorkload:
    opening_prefix_count: int
    suite_hash: str
    games: int
    actors: int
    simulations: int
    seed: int
    candidate_model: str
    opponent_model: str
    require_actor_refill: bool

    @classmethod
    def from_dict(cls, value: object) -> "ArenaWorkload":
        data = _mapping(value, "workloads.arena")
        _exact_fields(
            data,
            {
                "openingPrefixCount",
                "suiteHash",
                "games",
                "actors",
                "simulations",
                "seed",
                "candidateModel",
                "opponentModel",
                "requireActorRefill",
            },
            "workloads.arena",
        )
        candidate = data.get("candidateModel")
        opponent = data.get("opponentModel")
        require_refill = data.get("requireActorRefill")
        if not isinstance(candidate, str) or not candidate:
            raise Phase4E1ContractError(
                "workloads.arena.candidateModel must be non-empty"
            )
        if not isinstance(opponent, str) or not opponent:
            raise Phase4E1ContractError(
                "workloads.arena.opponentModel must be non-empty"
            )
        if type(require_refill) is not bool:
            raise Phase4E1ContractError(
                "workloads.arena.requireActorRefill must be boolean"
            )
        result = cls(
            opening_prefix_count=_positive_int(
                data.get("openingPrefixCount"),
                "workloads.arena.openingPrefixCount",
            ),
            suite_hash=_suite_hash(
                data.get("suiteHash"), "workloads.arena.suiteHash"
            ),
            games=_positive_int(data.get("games"), "workloads.arena.games"),
            actors=_positive_int(data.get("actors"), "workloads.arena.actors"),
            simulations=_positive_int(
                data.get("simulations"), "workloads.arena.simulations"
            ),
            seed=_positive_int(data.get("seed"), "workloads.arena.seed"),
            candidate_model=candidate,
            opponent_model=opponent,
            require_actor_refill=require_refill,
        )
        if result.games != result.opening_prefix_count * 2:
            raise Phase4E1ContractError(
                "arena games must contain exactly two legs per opening"
            )
        if result.require_actor_refill and result.games <= result.actors:
            raise Phase4E1ContractError(
                "actor-refill smoke requires more games than actors"
            )
        expected = {
            "opening_prefix_count": ARENA_OPENING_PREFIX_COUNT,
            "suite_hash": ARENA_SUITE_HASH,
            "games": ARENA_GAMES,
            "actors": ARENA_ACTORS,
            "simulations": ARENA_SIMULATIONS,
            "seed": WORKLOAD_SEED,
            "candidate_model": "bootstrap-baseline",
            "opponent_model": "bootstrap-baseline",
            "require_actor_refill": True,
        }
        for field, expected_value in expected.items():
            _fixed(
                getattr(result, field),
                expected_value,
                f"workloads.arena.{field}",
            )
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "openingPrefixCount": self.opening_prefix_count,
            "suiteHash": self.suite_hash,
            "games": self.games,
            "actors": self.actors,
            "simulations": self.simulations,
            "seed": self.seed,
            "candidateModel": self.candidate_model,
            "opponentModel": self.opponent_model,
            "requireActorRefill": self.require_actor_refill,
        }


@dataclass(frozen=True, slots=True)
class FreezePolicy:
    reserve_gpu_hours: float
    minimum_stage_gpu_hours: float
    maximum_generations: int

    @classmethod
    def from_dict(cls, value: object) -> "FreezePolicy":
        data = _mapping(value, "freeze")
        _exact_fields(
            data,
            {
                "reserveGpuHours",
                "minimumStageGpuHours",
                "maximumGenerations",
            },
            "freeze",
        )
        result = cls(
            reserve_gpu_hours=_finite_hours(
                data.get("reserveGpuHours"),
                "freeze.reserveGpuHours",
                positive=True,
            ),
            minimum_stage_gpu_hours=_finite_hours(
                data.get("minimumStageGpuHours"),
                "freeze.minimumStageGpuHours",
                positive=True,
            ),
            maximum_generations=_positive_int(
                data.get("maximumGenerations"), "freeze.maximumGenerations"
            ),
        )
        _fixed(
            result.reserve_gpu_hours,
            FREEZE_RESERVE_GPU_HOURS,
            "freeze.reserveGpuHours",
        )
        _fixed(
            result.minimum_stage_gpu_hours,
            FREEZE_MINIMUM_STAGE_GPU_HOURS,
            "freeze.minimumStageGpuHours",
        )
        _fixed(
            result.maximum_generations,
            FREEZE_MAXIMUM_GENERATIONS,
            "freeze.maximumGenerations",
        )
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "reserveGpuHours": self.reserve_gpu_hours,
            "minimumStageGpuHours": self.minimum_stage_gpu_hours,
            "maximumGenerations": self.maximum_generations,
        }


@dataclass(frozen=True, slots=True)
class Phase4E1CalibrationContract:
    schema_version: int
    kind: str
    stage: str
    status: str
    formal_promotion: bool
    training_allowed: bool
    resources: ResourceContract
    provenance: ProvenanceContract
    self_play: SelfPlayWorkload
    arena: ArenaWorkload
    freeze: FreezePolicy

    @classmethod
    def from_dict(cls, value: object) -> "Phase4E1CalibrationContract":
        data = _mapping(value, "Phase 4E1 calibration contract")
        _exact_fields(
            data,
            {
                "schemaVersion",
                "kind",
                "stage",
                "status",
                "formalPromotion",
                "trainingAllowed",
                "resources",
                "provenance",
                "workloads",
                "freeze",
            },
            "Phase 4E1 calibration contract",
        )
        workloads = _mapping(data.get("workloads"), "workloads")
        _exact_fields(workloads, {"selfPlay", "arena"}, "workloads")
        result = cls(
            schema_version=data.get("schemaVersion"),
            kind=data.get("kind"),
            stage=data.get("stage"),
            status=data.get("status"),
            formal_promotion=data.get("formalPromotion"),
            training_allowed=data.get("trainingAllowed"),
            resources=ResourceContract.from_dict(data.get("resources")),
            provenance=ProvenanceContract.from_dict(data.get("provenance")),
            self_play=SelfPlayWorkload.from_dict(workloads.get("selfPlay")),
            arena=ArenaWorkload.from_dict(workloads.get("arena")),
            freeze=FreezePolicy.from_dict(data.get("freeze")),
        )
        _fixed(
            result.schema_version,
            PHASE4E1_CONTRACT_SCHEMA_VERSION,
            "schemaVersion",
        )
        _fixed(result.kind, PHASE4E1_CONTRACT_KIND, "kind")
        _fixed(result.stage, PHASE4E1_STAGE, "stage")
        _fixed(result.status, PHASE4E1_STATUS, "status")
        _fixed(result.formal_promotion, False, "formalPromotion")
        _fixed(result.training_allowed, False, "trainingAllowed")
        committed_hours = (
            result.resources.initial_ledger.total_gpu_hours
            + result.resources.calibration_gpu_hour_limit
            + result.freeze.reserve_gpu_hours
            + result.freeze.minimum_stage_gpu_hours
        )
        if committed_hours > result.resources.global_gpu_hour_limit:
            raise Phase4E1ContractError(
                "calibration, reserve, and minimum stage budgets exceed "
                "the global GPU-hour limit"
            )
        return result

    @property
    def canonical_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_payload()))

    @property
    def sha256(self) -> str:
        return self.canonical_sha256

    def to_payload(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "kind": self.kind,
            "stage": self.stage,
            "status": self.status,
            "formalPromotion": self.formal_promotion,
            "trainingAllowed": self.training_allowed,
            "resources": self.resources.to_payload(),
            "provenance": self.provenance.to_payload(),
            "workloads": {
                "selfPlay": self.self_play.to_payload(),
                "arena": self.arena.to_payload(),
            },
            "freeze": self.freeze.to_payload(),
        }


def load_phase4e1_contract(path: str | Path) -> Phase4E1CalibrationContract:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4E1ContractError(
            f"cannot load Phase 4E1 calibration contract {source}"
        ) from error
    return Phase4E1CalibrationContract.from_dict(payload)


__all__ = [
    "PHASE4E1_CONTRACT_SCHEMA_VERSION",
    "PHASE4E1_CONTRACT_KIND",
    "PHASE4E1_STAGE",
    "PHASE4E1_STATUS",
    "Phase4E1ContractError",
    "LedgerSnapshot",
    "ResourceContract",
    "ProvenanceContract",
    "SelfPlayWorkload",
    "ArenaWorkload",
    "FreezePolicy",
    "Phase4E1CalibrationContract",
    "load_phase4e1_contract",
]
