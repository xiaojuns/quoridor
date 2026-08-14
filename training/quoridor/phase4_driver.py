"""Phase 4E candidate gates and atomic champion publication.

Training always advances from its latest checkpoint.  This controller only
decides whether a cadence checkpoint is eligible for the independent champion
archive; rejection never rewinds training state.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .champion_store import (
    DEFAULT_MILESTONES,
    DEFAULT_RECENT_CHAMPIONS,
    ArtifactReference,
    ChampionRecord,
    ChampionStore,
)
from .phase4_arena import (
    Phase4ArenaError,
    verify_completed_arena_result,
)
from .phase4_checkpoint import (
    PHASE4_CANDIDATE_MODEL_FORMAT,
    PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION,
    PHASE4_CHECKPOINT_INDEX_SCHEMA_VERSION,
    CheckpointIdentity,
    Phase4CheckpointStore,
    canonical_model_state_sha256,
)
from .run_manifest import (
    DEFAULT_ARTIFACT_LIMIT_BYTES,
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    atomic_replace_bytes,
    canonical_json_bytes,
    GenerationReportStore,
    load_run_manifest,
    normalize_json_mapping,
    sha256_bytes,
    sha256_file,
    validate_sha256,
)
from .statistics import PairedGameResult, summarize_paired_results


PHASE4_PROMOTION_SCHEMA_VERSION = 1
PHASE4_CANDIDATE_CADENCE_GENERATIONS = 5
PHASE4_SCREEN_OPENING_PAIRS = 32
PHASE4_SCREEN_GAMES = 64
PHASE4_SCREEN_SIMULATIONS = 256
PHASE4_SCREEN_MINIMUM_WIN_RATE = 0.55
PHASE4_CONFIRMATION_OPENING_PAIRS = 200
PHASE4_CONFIRMATION_GAMES = 400
PHASE4_CONFIRMATION_SIMULATIONS = 800
PHASE4_CONFIRMATION_MINIMUM_WIN_RATE = 0.55
PHASE4_CONFIRMATION_MINIMUM_LOWER = 0.50
PHASE4_HISTORY_MINIMUM_WIN_RATE = 0.50
PHASE4_HISTORY_MINIMUM_LOWER = 0.45
PHASE4_HISTORY_RECENT = 5
PHASE4_ARENA_BOOTSTRAP_SEED = 20_260_714
PHASE4_ARENA_BOOTSTRAP_SAMPLES = 10_000
PHASE4_ARENA_CONFIDENCE = 0.90
PHASE4_SCREEN_SUITE_HASH = (
    "sha256:bef13630f0655e4a5da75d5d82bd2abaffdb11dc057fbb50965d949d1550311a"
)
PHASE4_CONFIRMATION_SUITE_HASH = (
    "sha256:9cde8e3302916d023957b60d4a433f31f0dfc995881389aca95daee4730b11c4"
)
PHASE4_BOOTSTRAP_BASELINE_ID = "bootstrap-warm-v2"
PHASE4_BOOTSTRAP_MODEL_RELATIVE = Path(
    "baselines/bootstrap-warm-v2/model.pt"
)
PHASE4_BOOTSTRAP_MODEL_SHA256 = (
    "8db99ac64c5099bdf0df787350150e36c1b3c2343a826526522e027057cc8124"
)
PHASE4_BOOTSTRAP_MODEL_STATE_SHA256 = (
    "a69073538072b191845a21e1ce35f583bc954af7b484c64ed83bdc8388e59b0a"
)


class Phase4PromotionError(RuntimeError):
    """Base class for Phase 4E promotion-controller failures."""


class Phase4PromotionEvidenceError(Phase4PromotionError):
    """Raised when candidate or arena evidence is invalid."""


@dataclass(frozen=True, slots=True)
class PromotionStatus:
    state: str
    next_gate: str | None
    failed_gate: str | None
    completed_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "nextGate": self.next_gate,
            "failedGate": self.failed_gate,
            "completedGates": list(self.completed_gates),
        }


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    payload = dict(core)
    payload[field] = sha256_bytes(canonical_json_bytes(core))
    return payload


def _verify_self_hash(
    payload: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> str:
    try:
        digest = validate_sha256(payload.get(field), f"{label} {field}")
    except ValueError as error:
        raise Phase4PromotionEvidenceError(
            f"{label} has an invalid self hash"
        ) from error
    core = dict(payload)
    core.pop(field, None)
    if sha256_bytes(canonical_json_bytes(core)) != digest:
        raise Phase4PromotionEvidenceError(f"{label} self hash mismatch")
    return digest


def _load_canonical(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise Phase4PromotionEvidenceError(f"{label} is missing or is a symlink")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase4PromotionEvidenceError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping) or canonical_json_bytes(payload) != raw:
        raise Phase4PromotionEvidenceError(f"{label} is not canonical JSON")
    return dict(payload)


def _validate_pair_evidence(
    path: Path,
    artifact: Mapping[str, object],
    *,
    result: Mapping[str, object],
    candidate_identity: Mapping[str, object],
    opponent_identity: Mapping[str, object],
    candidate_runtime: Mapping[str, object],
    opponent_runtime: Mapping[str, object],
) -> tuple[PairedGameResult, ...]:
    payload = _load_canonical(path, f"pair artifact {path.name}")
    pair_sha256 = _verify_self_hash(
        payload,
        field="pairSha256",
        label=f"pair artifact {path.name}",
    )
    ordinal = artifact.get("ordinal")
    opening_id = artifact.get("openingId")
    games = payload.get("games")
    expected_name = (
        f"pair-{ordinal:06d}-{opening_id}-{pair_sha256[:12]}.json"
        if type(ordinal) is int and isinstance(opening_id, str)
        else ""
    )
    if (
        payload.get("schemaVersion") != 1
        or payload.get("kind") != "phase4-arena-opening-pair"
        or payload.get("contractSha256") != result.get("contractSha256")
        or payload.get("suiteHash") != result.get("suiteHash")
        or payload.get("commitUnit") != "complete-opening-pair-v1"
        or payload.get("ordinal") != ordinal
        or payload.get("openingId") != opening_id
        or path.name != expected_name
        or not isinstance(games, list)
        or len(games) != 2
        or any(not isinstance(game, Mapping) for game in games)
    ):
        raise Phase4PromotionEvidenceError(
            f"pair artifact contract mismatch: {path.name}"
        )
    ordered = sorted(games, key=lambda game: game.get("leg", -1))
    results: list[PairedGameResult] = []
    for leg, game in enumerate(ordered):
        candidate_player = game.get("candidatePlayer")
        opponent_player = game.get("opponentPlayer")
        winner = game.get("winner")
        if (
            game.get("schemaVersion") != 1
            or game.get("recordType") != "paired-arena-game"
            or game.get("openingId") != opening_id
            or game.get("suiteHash") != result.get("suiteHash")
            or game.get("leg") != leg
            or type(candidate_player) is not int
            or type(opponent_player) is not int
            or {candidate_player, opponent_player} != {0, 1}
            or type(winner) is not int
            or winner not in (0, 1)
            or game.get("candidateWon") != (winner == candidate_player)
            or game.get("candidateIdentity") != candidate_identity
            or game.get("opponentIdentity") != opponent_identity
            or game.get("candidateRuntime") != candidate_runtime
            or game.get("opponentRuntime") != opponent_runtime
        ):
            raise Phase4PromotionEvidenceError(
                f"pair artifact game identity mismatch: {path.name}"
            )
        results.append(
            PairedGameResult(
                opening_id=opening_id,
                leg=leg,
                candidate_player=candidate_player,
                winner=winner,
            )
        )
    if {game.get("candidatePlayer") for game in ordered} != {0, 1}:
        raise Phase4PromotionEvidenceError(
            f"pair artifact does not exchange player ownership: {path.name}"
        )
    return tuple(results)


def _safe_file(path: str | Path, root: Path, label: str) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise Phase4PromotionEvidenceError(f"{label} must not be a symlink")
    source = source.resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise Phase4PromotionEvidenceError(
            f"{label} must be inside the Phase 4 artifact root"
        ) from error
    if not source.is_file() or source.is_symlink():
        raise Phase4PromotionEvidenceError(f"{label} is missing")
    return source


def _digest_from_suite_hash(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise Phase4PromotionEvidenceError(f"{label} must be a suite hash")
    digest = value.removeprefix("sha256:")
    try:
        return validate_sha256(digest, label)
    except ValueError as error:
        raise Phase4PromotionEvidenceError(f"{label} is invalid") from error


def _opponent_descriptor(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4PromotionEvidenceError("opponent descriptor must be a mapping")
    expected = {"id", "kind", "artifactSha256", "modelStateSha256", "championId"}
    if set(value) != expected:
        raise Phase4PromotionEvidenceError("opponent descriptor fields are invalid")
    opponent_id = value.get("id")
    kind = value.get("kind")
    champion_id = value.get("championId")
    if not isinstance(opponent_id, str) or not opponent_id:
        raise Phase4PromotionEvidenceError("opponent id must be non-empty")
    if kind not in {"bootstrap-baseline", "champion"}:
        raise Phase4PromotionEvidenceError("opponent kind is invalid")
    if (kind == "champion") != isinstance(champion_id, str):
        raise Phase4PromotionEvidenceError(
            "champion opponent must bind exactly one championId"
        )
    return {
        "id": opponent_id,
        "kind": kind,
        "artifactSha256": validate_sha256(
            value.get("artifactSha256"), "opponent artifact sha256"
        ),
        "modelStateSha256": validate_sha256(
            value.get("modelStateSha256"), "opponent model-state sha256"
        ),
        "championId": champion_id,
    }


def bootstrap_opponent_descriptor(
    *,
    baseline_id: str,
    artifact_sha256: str,
    model_state_sha256: str,
) -> dict[str, object]:
    return _opponent_descriptor(
        {
            "id": baseline_id,
            "kind": "bootstrap-baseline",
            "artifactSha256": artifact_sha256,
            "modelStateSha256": model_state_sha256,
            "championId": None,
        }
    )


def _verified_bootstrap_opponent(artifact_root: Path) -> dict[str, object]:
    path = _safe_file(
        artifact_root / PHASE4_BOOTSTRAP_MODEL_RELATIVE,
        artifact_root,
        "Phase 4E bootstrap model",
    )
    if sha256_file(path) != PHASE4_BOOTSTRAP_MODEL_SHA256:
        raise Phase4PromotionEvidenceError(
            "Phase 4E bootstrap model SHA-256 mismatch"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise Phase4PromotionEvidenceError(
            "Phase 4E bootstrap model cannot be safely deserialized"
        ) from error
    state = payload.get("modelState") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schemaVersion") != 1
        or payload.get("kind") != "phase4e-bootstrap-baseline"
        or payload.get("format")
        != "quoridor-phase4e-bootstrap-model-only"
        or payload.get("formalPromotion") is not False
        or payload.get("modelStateSha256")
        != PHASE4_BOOTSTRAP_MODEL_STATE_SHA256
        or not isinstance(state, Mapping)
    ):
        raise Phase4PromotionEvidenceError(
            "Phase 4E bootstrap model contract is invalid"
        )
    try:
        model_state_sha256 = canonical_model_state_sha256(state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Phase4PromotionEvidenceError(
            "Phase 4E bootstrap model state is invalid"
        ) from error
    if model_state_sha256 != PHASE4_BOOTSTRAP_MODEL_STATE_SHA256:
        raise Phase4PromotionEvidenceError(
            "Phase 4E bootstrap tensor state hash mismatch"
        )
    return bootstrap_opponent_descriptor(
        baseline_id=PHASE4_BOOTSTRAP_BASELINE_ID,
        artifact_sha256=PHASE4_BOOTSTRAP_MODEL_SHA256,
        model_state_sha256=PHASE4_BOOTSTRAP_MODEL_STATE_SHA256,
    )


def champion_opponent_descriptor(record: ChampionRecord) -> dict[str, object]:
    model_state = record.metadata.get("candidateModelStateSha256")
    if not isinstance(model_state, str):
        model_state = record.model_sha256
    return _opponent_descriptor(
        {
            "id": record.champion_id,
            "kind": "champion",
            "artifactSha256": record.model_sha256,
            "modelStateSha256": model_state,
            "championId": record.champion_id,
        }
    )


def phase4_arena_contract_sha256(
    *,
    gate: str,
    execution_contract_sha256: str,
    candidate_artifact_sha256: str,
    opponent_artifact_sha256: str,
    suite_hash: str,
) -> str:
    if gate not in {"screen", "confirmation", "history"}:
        raise ValueError("arena gate must be screen, confirmation, or history")
    suite_sha256 = _digest_from_suite_hash(suite_hash, "arena suite sha256")
    confirmation_shape = gate != "screen"
    pairs = (
        PHASE4_CONFIRMATION_OPENING_PAIRS
        if confirmation_shape
        else PHASE4_SCREEN_OPENING_PAIRS
    )
    games = PHASE4_CONFIRMATION_GAMES if confirmation_shape else PHASE4_SCREEN_GAMES
    simulations = (
        PHASE4_CONFIRMATION_SIMULATIONS
        if confirmation_shape
        else PHASE4_SCREEN_SIMULATIONS
    )
    core = {
        "schemaVersion": 1,
        "kind": "phase4e-arena-contract",
        "gate": gate,
        "executionContractSha256": validate_sha256(
            execution_contract_sha256,
            "execution contract sha256",
        ),
        "candidateArtifactSha256": validate_sha256(
            candidate_artifact_sha256,
            "candidate artifact sha256",
        ),
        "opponentArtifactSha256": validate_sha256(
            opponent_artifact_sha256,
            "opponent artifact sha256",
        ),
        "suiteHash": f"sha256:{suite_sha256}",
        "openingPairs": pairs,
        "games": games,
        "simulationsPerSide": simulations,
        "scheduler": "opening-pair-actor-refill-v1",
        "commitUnit": "complete-opening-pair-v1",
        "bootstrap": {
            "method": "opening-pair-cluster-percentile",
            "seed": PHASE4_ARENA_BOOTSTRAP_SEED,
            "samples": PHASE4_ARENA_BOOTSTRAP_SAMPLES,
            "confidence": PHASE4_ARENA_CONFIDENCE,
        },
    }
    return sha256_bytes(canonical_json_bytes(core))


def select_history_opponents(
    store: ChampionStore,
    *,
    exclude_model_sha256: Sequence[str] = (),
) -> tuple[dict[str, object], ...]:
    """Recent five plus best/product, deduplicated by canonical model identity."""

    excluded = set(exclude_model_sha256)
    selected: list[dict[str, object]] = []
    seen: set[str] = set()

    def append_if_distinct(record: ChampionRecord) -> bool:
        descriptor = champion_opponent_descriptor(record)
        model_sha256 = str(descriptor["modelStateSha256"])
        if model_sha256 in excluded or model_sha256 in seen:
            return False
        if not record.retained or record.file_name is None:
            raise Phase4PromotionEvidenceError(
                f"required history champion model was pruned: {record.champion_id}"
            )
        model_path = store.root / record.file_name
        if (
            model_path.is_symlink()
            or not model_path.is_file()
            or model_path.stat().st_size != record.byte_size
            or sha256_file(model_path) != record.model_sha256
        ):
            raise Phase4PromotionEvidenceError(
                f"required history champion model is unavailable: {record.champion_id}"
            )
        seen.add(model_sha256)
        selected.append(descriptor)
        return True

    recent_count = 0
    for record in reversed(store.records):
        if append_if_distinct(record):
            recent_count += 1
            if recent_count == PHASE4_HISTORY_RECENT:
                break
    for role in ("best", "product"):
        champion_id = store.pointers.get(role)
        if champion_id is not None:
            append_if_distinct(store.resolve(champion_id))
    return tuple(selected)


def _load_candidate_model_artifact(
    artifact_path: Path,
    *,
    artifact_sha256: str,
    artifact_bytes: int,
    model_state_sha256: str,
    checkpoint_sha256: str,
    generation: int,
    step: int,
    identity: CheckpointIdentity,
) -> tuple[Mapping[str, torch.Tensor], str]:
    if (
        type(artifact_bytes) is not int
        or artifact_bytes <= 0
        or artifact_path.stat().st_size != artifact_bytes
        or sha256_file(artifact_path) != artifact_sha256
    ):
        raise Phase4PromotionEvidenceError(
            "candidate model artifact metadata mismatch"
        )
    try:
        payload = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise Phase4PromotionEvidenceError(
            "candidate model artifact cannot be deserialized safely"
        ) from error
    expected_fields = {
        "schemaVersion",
        "format",
        "generation",
        "step",
        "identity",
        "checkpointSha256",
        "modelStateSha256",
        "modelState",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_fields
        or payload.get("schemaVersion")
        != PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION
        or payload.get("format") != PHASE4_CANDIDATE_MODEL_FORMAT
        or payload.get("generation") != generation
        or payload.get("step") != step
        or payload.get("identity") != identity.to_payload()
        or payload.get("checkpointSha256") != checkpoint_sha256
        or payload.get("modelStateSha256") != model_state_sha256
    ):
        raise Phase4PromotionEvidenceError(
            "candidate model artifact does not match checkpoint lineage"
        )
    state = payload.get("modelState")
    if not isinstance(state, Mapping):
        raise Phase4PromotionEvidenceError(
            "candidate model artifact does not contain a model state"
        )
    try:
        actual_model_state_sha256 = canonical_model_state_sha256(state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise Phase4PromotionEvidenceError(
            "candidate model tensors are invalid"
        ) from error
    if actual_model_state_sha256 != model_state_sha256:
        raise Phase4PromotionEvidenceError(
            "candidate model tensor-state hash mismatch"
        )
    return state, actual_model_state_sha256


def _load_training_checkpoint_evidence(
    checkpoint_path: Path,
    *,
    run_manifest_path: Path,
    artifact_root: Path,
    execution_contract_sha256: str,
) -> tuple[Mapping[str, torch.Tensor], dict[str, object]]:
    checkpoint = _safe_file(
        checkpoint_path, artifact_root, "candidate checkpoint"
    )
    manifest_path = _safe_file(
        run_manifest_path, artifact_root, "candidate run manifest"
    )
    run_root = manifest_path.parent
    if manifest_path.name != "run-manifest.json" or checkpoint.parent != (
        run_root / "checkpoints"
    ).resolve():
        raise Phase4PromotionEvidenceError(
            "candidate checkpoint is not owned by the supplied run manifest"
        )
    try:
        manifest = load_run_manifest(manifest_path)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise Phase4PromotionEvidenceError(
            "candidate run manifest is invalid"
        ) from error
    if canonical_json_bytes(manifest.payload) != manifest_path.read_bytes():
        raise Phase4PromotionEvidenceError(
            "candidate run manifest is not canonical JSON"
        )
    provenance = manifest.payload.get("provenance")
    metadata = manifest.payload.get("metadata")
    config = provenance.get("config") if isinstance(provenance, Mapping) else None
    source = provenance.get("source") if isinstance(provenance, Mapping) else None
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("pipeline") != "phase4"
        or metadata.get("stage") != "phase4e"
        or metadata.get("formalPromotion") is not True
        or metadata.get("executionContractSha256")
        != execution_contract_sha256
        or not isinstance(config, Mapping)
        or not isinstance(source, Mapping)
    ):
        raise Phase4PromotionEvidenceError(
            "candidate run manifest is not a formal Phase 4E lineage"
        )
    try:
        config_sha256 = validate_sha256(
            config.get("sha256"), "candidate config sha256"
        )
        source_sha256 = validate_sha256(
            source.get("sha256"), "candidate source sha256"
        )
    except ValueError as error:
        raise Phase4PromotionEvidenceError(
            "candidate run provenance hashes are invalid"
        ) from error
    identity = CheckpointIdentity(
        run_manifest_sha256=manifest.sha256,
        config_sha256=config_sha256,
        source_sha256=source_sha256,
    )

    metadata_path = checkpoint.with_suffix(".json")
    checkpoint_metadata = _load_canonical(
        metadata_path, "candidate checkpoint metadata"
    )
    expected_metadata_fields = {
        "schemaVersion",
        "generation",
        "step",
        "file",
        "bytes",
        "sha256",
        "identity",
        "candidateModel",
    }
    if (
        set(checkpoint_metadata) != expected_metadata_fields
        or checkpoint_metadata.get("schemaVersion")
        != PHASE4_CHECKPOINT_INDEX_SCHEMA_VERSION
        or checkpoint_metadata.get("file") != checkpoint.name
        or checkpoint_metadata.get("bytes") != checkpoint.stat().st_size
        or checkpoint_metadata.get("identity") != identity.to_payload()
    ):
        raise Phase4PromotionEvidenceError(
            "candidate checkpoint metadata does not match its lineage"
        )
    try:
        checkpoint_sha256 = validate_sha256(
            checkpoint_metadata.get("sha256"),
            "candidate checkpoint sha256",
        )
    except ValueError as error:
        raise Phase4PromotionEvidenceError(
            "candidate checkpoint metadata hash is invalid"
        ) from error
    generation = checkpoint_metadata.get("generation")
    step = checkpoint_metadata.get("step")
    if (
        type(generation) is not int
        or generation < 0
        or (generation + 1) % PHASE4_CANDIDATE_CADENCE_GENERATIONS != 0
        or type(step) is not int
        or step <= 0
        or sha256_file(checkpoint) != checkpoint_sha256
    ):
        raise Phase4PromotionEvidenceError(
            "candidate checkpoint is not a committed cadence checkpoint"
        )
    candidate_model = checkpoint_metadata.get("candidateModel")
    expected_candidate_fields = {
        "schemaVersion",
        "format",
        "file",
        "bytes",
        "sha256",
        "modelStateSha256",
    }
    if (
        not isinstance(candidate_model, Mapping)
        or set(candidate_model) != expected_candidate_fields
        or candidate_model.get("schemaVersion")
        != PHASE4_CANDIDATE_MODEL_SCHEMA_VERSION
        or candidate_model.get("format") != PHASE4_CANDIDATE_MODEL_FORMAT
    ):
        raise Phase4PromotionEvidenceError(
            "candidate model metadata is invalid"
        )
    candidate_model_name = candidate_model.get("file")
    expected_candidate_name = f"candidate-model-{checkpoint.stem}.pt"
    if (
        not isinstance(candidate_model_name, str)
        or Path(candidate_model_name).name != candidate_model_name
        or candidate_model_name != expected_candidate_name
    ):
        raise Phase4PromotionEvidenceError(
            "candidate model file name is invalid"
        )
    try:
        candidate_model_sha256 = validate_sha256(
            candidate_model.get("sha256"), "candidate model sha256"
        )
        model_state_sha256 = validate_sha256(
            candidate_model.get("modelStateSha256"),
            "candidate model state sha256",
        )
    except ValueError as error:
        raise Phase4PromotionEvidenceError(
            "candidate model metadata hashes are invalid"
        ) from error
    candidate_model_bytes = candidate_model.get("bytes")
    candidate_model_path = _safe_file(
        checkpoint.parent / candidate_model_name,
        artifact_root,
        "candidate model artifact",
    )
    try:
        state, actual_model_state_sha256 = _load_candidate_model_artifact(
            candidate_model_path,
            artifact_sha256=candidate_model_sha256,
            artifact_bytes=candidate_model_bytes,
            model_state_sha256=model_state_sha256,
            checkpoint_sha256=checkpoint_sha256,
            generation=generation,
            step=step,
            identity=identity,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise Phase4PromotionEvidenceError(
            "candidate model artifact is invalid"
        ) from error
    if actual_model_state_sha256 != model_state_sha256:
        raise Phase4PromotionEvidenceError(
            "candidate model state differs from checkpoint metadata"
        )

    report_root = run_root / "reports"
    if (
        not report_root.is_dir()
        or report_root.is_symlink()
    ):
        raise Phase4PromotionEvidenceError(
            "candidate generation report directory is missing"
        )
    report_store = GenerationReportStore(
        report_root,
        run_manifest_sha256=manifest.sha256,
    )
    try:
        report_record = report_store.load(generation)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise Phase4PromotionEvidenceError(
            "candidate generation report is invalid"
        ) from error
    if canonical_json_bytes(report_record.payload) != report_record.path.read_bytes():
        raise Phase4PromotionEvidenceError(
            "candidate generation report is not canonical JSON"
        )
    report = report_record.payload.get("report")
    report_checkpoint = report.get("checkpoint") if isinstance(report, Mapping) else None
    report_candidate_model = (
        report_checkpoint.get("candidateModel")
        if isinstance(report_checkpoint, Mapping)
        else None
    )
    generation_data = (
        report.get("generationData") if isinstance(report, Mapping) else None
    )
    if (
        not isinstance(report, Mapping)
        or not isinstance(report_checkpoint, Mapping)
        or not isinstance(report_candidate_model, Mapping)
        or not isinstance(generation_data, Mapping)
        or report_checkpoint.get("path") != checkpoint.name
        or report_checkpoint.get("sha256") != checkpoint_sha256
        or report_checkpoint.get("bytes") != checkpoint.stat().st_size
        or report_checkpoint.get("step") != step
        or dict(report_candidate_model)
        != {
            "path": candidate_model_path.name,
            "bytes": candidate_model_bytes,
            "sha256": candidate_model_sha256,
            "modelStateSha256": model_state_sha256,
        }
    ):
        raise Phase4PromotionEvidenceError(
            "candidate generation report does not commit the checkpoint"
        )
    try:
        replay_snapshot = generation_data.get("replay")
        if not isinstance(replay_snapshot, Mapping):
            raise TypeError("generation replay snapshot must be a mapping")
        Phase4CheckpointStore._validate_generation_contract(
            generation,
            step,
            generation_data,
            replay_snapshot,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise Phase4PromotionEvidenceError(
            "candidate generation report contract is invalid"
        ) from error
    evidence = {
        "generation": generation,
        "step": step,
        "checkpoint": ArtifactReference.from_file(
            checkpoint, relative_to=artifact_root
        ).to_payload(),
        "checkpointMetadata": ArtifactReference.from_file(
            metadata_path, relative_to=artifact_root
        ).to_payload(),
        "candidateModel": {
            **ArtifactReference.from_file(
                candidate_model_path, relative_to=artifact_root
            ).to_payload(),
            "bytes": candidate_model_bytes,
            "modelStateSha256": model_state_sha256,
        },
        "runManifest": ArtifactReference.from_file(
            manifest_path, relative_to=artifact_root
        ).to_payload(),
        "runManifestSelfSha256": manifest.sha256,
        "generationReport": ArtifactReference.from_file(
            report_record.path, relative_to=artifact_root
        ).to_payload(),
        "generationReportSelfSha256": report_record.sha256,
        "modelStateSha256": model_state_sha256,
        "configSha256": config_sha256,
        "sourceSha256": source_sha256,
    }
    return state, evidence


class Phase4PromotionController:
    """Durable, fail-fast gate state for one cadence candidate."""

    def __init__(
        self,
        root: str | Path,
        *,
        artifact_root: str | Path,
        execution_contract_sha256: str,
        champion_store: ChampionStore,
    ) -> None:
        self.root = Path(root)
        self.artifact_root = Path(artifact_root).resolve()
        self.execution_contract_sha256 = validate_sha256(
            execution_contract_sha256,
            "execution contract sha256",
        )
        if not isinstance(champion_store, ChampionStore):
            raise TypeError("champion_store must be a ChampionStore")
        if champion_store.artifact_root != self.artifact_root:
            raise ValueError("champion store and promotion controller need one artifact root")
        if (
            champion_store.budget_bytes != DEFAULT_ARTIFACT_LIMIT_BYTES
            or champion_store.recent_to_keep != DEFAULT_RECENT_CHAMPIONS
            or champion_store.milestones_to_keep != DEFAULT_MILESTONES
            or champion_store.global_artifact_limit_bytes
            != DEFAULT_ARTIFACT_LIMIT_BYTES
            or champion_store.minimum_free_disk_bytes
            != DEFAULT_MINIMUM_FREE_DISK_BYTES
        ):
            raise Phase4PromotionEvidenceError(
                "formal promotion requires the frozen champion resource contract"
            )
        self.champion_store = champion_store
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "candidate-manifest.json"

    def prepare_candidate(
        self,
        *,
        checkpoint_path: str | Path,
        run_manifest_path: str | Path,
        primary_opponent: Mapping[str, object],
    ) -> dict[str, object]:
        _, lineage = _load_training_checkpoint_evidence(
            Path(checkpoint_path),
            run_manifest_path=Path(run_manifest_path),
            artifact_root=self.artifact_root,
            execution_contract_sha256=self.execution_contract_sha256,
        )
        generation = int(lineage["generation"])
        model_state_sha256 = str(lineage["modelStateSha256"])
        primary = _opponent_descriptor(primary_opponent)
        latest_id = self.champion_store.pointers["latest"]
        if latest_id is not None:
            expected_primary = champion_opponent_descriptor(
                self.champion_store.resolve(latest_id)
            )
            if primary != expected_primary:
                raise Phase4PromotionEvidenceError(
                    "primary opponent must be the current latest champion"
                )
        elif primary != _verified_bootstrap_opponent(self.artifact_root):
            raise Phase4PromotionEvidenceError(
                "the first formal candidate requires the frozen bootstrap baseline"
            )
        history = select_history_opponents(
            self.champion_store,
            exclude_model_sha256=(
                model_state_sha256,
                str(primary["modelStateSha256"]),
            ),
        )
        core = {
            "schemaVersion": PHASE4_PROMOTION_SCHEMA_VERSION,
            "kind": "phase4e-promotion-candidate",
            "formalPromotion": True,
            "executionContractSha256": self.execution_contract_sha256,
            "generation": generation,
            "committedGenerations": generation + 1,
            "step": lineage["step"],
            "checkpoint": lineage["checkpoint"],
            "checkpointMetadata": lineage["checkpointMetadata"],
            "candidateModel": lineage["candidateModel"],
            "runManifest": lineage["runManifest"],
            "runManifestSelfSha256": lineage["runManifestSelfSha256"],
            "generationReport": lineage["generationReport"],
            "generationReportSelfSha256": lineage[
                "generationReportSelfSha256"
            ],
            "modelStateSha256": model_state_sha256,
            "configSha256": lineage["configSha256"],
            "sourceSha256": lineage["sourceSha256"],
            "screenSuiteHash": PHASE4_SCREEN_SUITE_HASH,
            "screenSuiteSha256": _digest_from_suite_hash(
                PHASE4_SCREEN_SUITE_HASH, "screen suite sha256"
            ),
            "confirmationSuiteHash": PHASE4_CONFIRMATION_SUITE_HASH,
            "confirmationSuiteSha256": _digest_from_suite_hash(
                PHASE4_CONFIRMATION_SUITE_HASH,
                "confirmation suite sha256",
            ),
            "primaryOpponent": primary,
            "historyOpponents": list(history),
            "trainingPolicy": {
                "candidateCadenceGenerations": PHASE4_CANDIDATE_CADENCE_GENERATIONS,
                "actorsTrainLatestCheckpoint": True,
                "rejectionRollsBackTraining": False,
            },
        }
        payload = _self_hashed(core, "candidateSha256")
        encoded = canonical_json_bytes(payload)
        if self.manifest_path.exists():
            if self.manifest_path.read_bytes() != encoded:
                raise Phase4PromotionEvidenceError(
                    "candidate manifest already exists with different evidence"
                )
        else:
            atomic_replace_bytes(self.manifest_path, encoded)
        return self.load_candidate()

    def load_candidate(self) -> dict[str, object]:
        payload = _load_canonical(self.manifest_path, "candidate manifest")
        _verify_self_hash(
            payload,
            field="candidateSha256",
            label="candidate manifest",
        )
        expected_fields = {
            "schemaVersion",
            "kind",
            "formalPromotion",
            "executionContractSha256",
            "generation",
            "committedGenerations",
            "step",
            "checkpoint",
            "checkpointMetadata",
            "candidateModel",
            "runManifest",
            "runManifestSelfSha256",
            "generationReport",
            "generationReportSelfSha256",
            "modelStateSha256",
            "configSha256",
            "sourceSha256",
            "screenSuiteHash",
            "screenSuiteSha256",
            "confirmationSuiteHash",
            "confirmationSuiteSha256",
            "primaryOpponent",
            "historyOpponents",
            "trainingPolicy",
            "candidateSha256",
        }
        if (
            set(payload) != expected_fields
            or payload.get("schemaVersion") != PHASE4_PROMOTION_SCHEMA_VERSION
            or payload.get("kind") != "phase4e-promotion-candidate"
            or payload.get("formalPromotion") is not True
            or payload.get("executionContractSha256")
            != self.execution_contract_sha256
        ):
            raise Phase4PromotionEvidenceError("candidate manifest contract mismatch")
        checkpoint = payload.get("checkpoint")
        run_manifest = payload.get("runManifest")
        if not isinstance(checkpoint, Mapping) or not isinstance(
            run_manifest, Mapping
        ):
            raise Phase4PromotionEvidenceError(
                "candidate training references are invalid"
            )
        _, lineage = _load_training_checkpoint_evidence(
            self.artifact_root / str(checkpoint.get("path")),
            run_manifest_path=self.artifact_root
            / str(run_manifest.get("path")),
            artifact_root=self.artifact_root,
            execution_contract_sha256=self.execution_contract_sha256,
        )
        lineage_fields = {
            "generation": lineage["generation"],
            "step": lineage["step"],
            "checkpoint": lineage["checkpoint"],
            "checkpointMetadata": lineage["checkpointMetadata"],
            "candidateModel": lineage["candidateModel"],
            "runManifest": lineage["runManifest"],
            "runManifestSelfSha256": lineage["runManifestSelfSha256"],
            "generationReport": lineage["generationReport"],
            "generationReportSelfSha256": lineage[
                "generationReportSelfSha256"
            ],
            "modelStateSha256": lineage["modelStateSha256"],
            "configSha256": lineage["configSha256"],
            "sourceSha256": lineage["sourceSha256"],
        }
        if any(payload.get(key) != value for key, value in lineage_fields.items()):
            raise Phase4PromotionEvidenceError(
                "candidate manifest differs from its committed training lineage"
            )
        if (
            payload.get("committedGenerations")
            != int(lineage["generation"]) + 1
            or payload.get("trainingPolicy")
            != {
                "candidateCadenceGenerations": (
                    PHASE4_CANDIDATE_CADENCE_GENERATIONS
                ),
                "actorsTrainLatestCheckpoint": True,
                "rejectionRollsBackTraining": False,
            }
            or payload.get("screenSuiteHash") != PHASE4_SCREEN_SUITE_HASH
            or payload.get("confirmationSuiteHash")
            != PHASE4_CONFIRMATION_SUITE_HASH
            or payload.get("screenSuiteSha256")
            != _digest_from_suite_hash(
                payload.get("screenSuiteHash"),
                "screen suite sha256",
            )
            or payload.get("confirmationSuiteSha256")
            != _digest_from_suite_hash(
                payload.get("confirmationSuiteHash"),
                "confirmation suite sha256",
            )
        ):
            raise Phase4PromotionEvidenceError(
                "candidate fixed promotion policy is invalid"
            )
        primary = payload.get("primaryOpponent")
        history = payload.get("historyOpponents")
        if not isinstance(primary, Mapping) or not isinstance(history, list):
            raise Phase4PromotionEvidenceError(
                "candidate opponent evidence is invalid"
            )
        normalized_primary = _opponent_descriptor(primary)
        latest_id = self.champion_store.pointers["latest"]
        latest_record = (
            None
            if latest_id is None
            else self.champion_store.resolve(latest_id)
        )
        candidate_already_published = (
            latest_record is not None
            and latest_record.metadata.get("candidateModelStateSha256")
            == lineage["modelStateSha256"]
            and latest_record.metadata.get("candidateSha256")
            == payload.get("candidateSha256")
        )
        if (
            latest_record is not None
            and not candidate_already_published
            and normalized_primary
            != champion_opponent_descriptor(latest_record)
        ):
            raise Phase4PromotionEvidenceError(
                "candidate primary opponent is no longer the latest champion"
            )
        if (
            latest_id is None
            and normalized_primary
            != _verified_bootstrap_opponent(self.artifact_root)
        ):
            raise Phase4PromotionEvidenceError(
                "the first formal candidate requires the frozen bootstrap baseline"
            )
        if candidate_already_published:
            for value in history:
                opponent = _opponent_descriptor(value)
                champion_id = opponent.get("championId")
                if (
                    not isinstance(champion_id, str)
                    or champion_opponent_descriptor(
                        self.champion_store.resolve(champion_id)
                    )
                    != opponent
                ):
                    raise Phase4PromotionEvidenceError(
                        "published candidate history opponent changed"
                    )
        else:
            expected_history = list(
                select_history_opponents(
                    self.champion_store,
                    exclude_model_sha256=(
                        str(lineage["modelStateSha256"]),
                        str(normalized_primary["modelStateSha256"]),
                    ),
                )
            )
            if history != expected_history:
                raise Phase4PromotionEvidenceError(
                    "candidate history opponents do not match the champion archive"
                )
        return payload

    def record_health(self) -> dict[str, object]:
        candidate = self.load_candidate()
        return self._write_gate(
            "health",
            None,
            self._health_gate_core(candidate),
        )

    def _health_gate_core(
        self, candidate: Mapping[str, object]
    ) -> dict[str, object]:
        checkpoint = candidate.get("checkpoint")
        run_manifest = candidate.get("runManifest")
        if not isinstance(checkpoint, Mapping) or not isinstance(
            run_manifest, Mapping
        ):
            raise Phase4PromotionEvidenceError(
                "candidate training references are invalid"
            )
        _, lineage = _load_training_checkpoint_evidence(
            self.artifact_root / str(checkpoint.get("path")),
            run_manifest_path=self.artifact_root
            / str(run_manifest.get("path")),
            artifact_root=self.artifact_root,
            execution_contract_sha256=self.execution_contract_sha256,
        )
        passed = (
            lineage["modelStateSha256"] == candidate.get("modelStateSha256")
            and lineage["candidateModel"] == candidate.get("candidateModel")
        )
        core = {
            "schemaVersion": PHASE4_PROMOTION_SCHEMA_VERSION,
            "kind": "phase4e-gate-evidence",
            "gate": "health",
            "candidateSha256": candidate["candidateSha256"],
            "passed": passed,
            "checks": {
                "checkpointSha256": True,
                "trainingLineage": True,
                "candidateModelBound": passed,
                "safeDeserialization": True,
                "finiteModelState": True,
                "modelStateSha256": passed,
            },
        }
        return core

    def _gate_path(self, gate: str, opponent_id: str | None) -> Path:
        suffix = "" if opponent_id is None else f"-{sha256_bytes(opponent_id.encode())[:12]}"
        return self.root / f"gate-{gate}{suffix}.json"

    def _write_gate(
        self,
        gate: str,
        opponent_id: str | None,
        core: Mapping[str, object],
    ) -> dict[str, object]:
        payload = _self_hashed(core, "gateSha256")
        path = self._gate_path(gate, opponent_id)
        encoded = canonical_json_bytes(payload)
        if path.exists():
            if path.read_bytes() != encoded:
                raise Phase4PromotionEvidenceError(
                    f"{gate} gate evidence already differs"
                )
        else:
            atomic_replace_bytes(path, encoded)
        return payload

    def _expected_opponent(
        self,
        candidate: Mapping[str, object],
        gate: str,
        opponent_id: str | None,
    ) -> dict[str, object]:
        if gate in {"screen", "confirmation"}:
            value = candidate.get("primaryOpponent")
        elif gate == "history":
            values = candidate.get("historyOpponents")
            if not isinstance(values, list) or opponent_id is None:
                raise Phase4PromotionEvidenceError("history opponent list is invalid")
            value = next(
                (
                    item
                    for item in values
                    if isinstance(item, Mapping) and item.get("id") == opponent_id
                ),
                None,
            )
        else:
            raise ValueError("gate must be screen, confirmation, or history")
        if not isinstance(value, Mapping):
            raise Phase4PromotionEvidenceError(
                f"candidate does not bind the requested {gate} opponent"
            )
        opponent = _opponent_descriptor(value)
        if opponent_id is not None and opponent["id"] != opponent_id:
            raise Phase4PromotionEvidenceError("arena opponent id mismatch")
        return opponent

    def record_arena_gate(
        self,
        gate: str,
        arena_result_path: str | Path,
        *,
        opponent_id: str | None = None,
    ) -> dict[str, object]:
        candidate = self.load_candidate()
        current = self.status()
        if current.failed_gate is not None:
            raise Phase4PromotionError(
                f"candidate already failed {current.failed_gate}; gates are fail-fast"
            )
        expected_next = current.next_gate
        requested = gate if gate != "history" else f"history:{opponent_id}"
        if expected_next != requested:
            raise Phase4PromotionError(
                f"gate order mismatch: expected {expected_next!r}, got {requested!r}"
            )
        core = self._arena_gate_core(
            candidate,
            gate,
            arena_result_path,
            opponent_id=opponent_id,
        )
        opponent = core["opponent"]
        if not isinstance(opponent, Mapping):
            raise Phase4PromotionEvidenceError("arena opponent evidence is invalid")
        return self._write_gate(gate, str(opponent["id"]), core)

    def _arena_gate_core(
        self,
        candidate: Mapping[str, object],
        gate: str,
        arena_result_path: str | Path,
        *,
        opponent_id: str | None,
    ) -> dict[str, object]:
        opponent = self._expected_opponent(candidate, gate, opponent_id)
        result_path = _safe_file(
            arena_result_path, self.artifact_root, "arena result"
        )
        try:
            result = verify_completed_arena_result(result_path)
        except (OSError, TypeError, ValueError, RuntimeError, Phase4ArenaError) as error:
            raise Phase4PromotionEvidenceError(
                "arena result failed independent transcript verification"
            ) from error
        result_sha256 = validate_sha256(
            result.get("resultSha256"), "arena result self hash"
        )
        if (
            result.get("status") != "completed"
            or result.get("kind") != "phase4-resumable-paired-arena-result"
        ):
            raise Phase4PromotionEvidenceError("arena result is not completed")
        candidate_identity = result.get("candidateIdentity")
        opponent_identity = result.get("opponentIdentity")
        candidate_runtime = result.get("candidateRuntime")
        opponent_runtime = result.get("opponentRuntime")
        if not all(
            isinstance(value, Mapping)
            for value in (
                candidate_identity,
                opponent_identity,
                candidate_runtime,
                opponent_runtime,
            )
        ):
            raise Phase4PromotionEvidenceError("arena participant evidence is invalid")
        candidate_model = candidate.get("candidateModel")
        if not isinstance(candidate_model, Mapping):
            raise Phase4PromotionEvidenceError(
                "candidate model artifact reference is invalid"
            )
        try:
            candidate_artifact_sha256 = validate_sha256(
                candidate_model.get("sha256"),
                "candidate model artifact sha256",
            )
        except ValueError as error:
            raise Phase4PromotionEvidenceError(
                "candidate model artifact hash is invalid"
            ) from error
        expected_suite = (
            candidate["screenSuiteHash"]
            if gate == "screen"
            else candidate["confirmationSuiteHash"]
        )
        expected_pairs = (
            PHASE4_SCREEN_OPENING_PAIRS
            if gate == "screen"
            else PHASE4_CONFIRMATION_OPENING_PAIRS
        )
        expected_games = (
            PHASE4_SCREEN_GAMES
            if gate == "screen"
            else PHASE4_CONFIRMATION_GAMES
        )
        expected_simulations = (
            PHASE4_SCREEN_SIMULATIONS
            if gate == "screen"
            else PHASE4_CONFIRMATION_SIMULATIONS
        )
        pair_artifacts = result.get("pairArtifacts")
        pair_files_valid = (
            isinstance(pair_artifacts, list)
            and len(pair_artifacts) == expected_pairs
        )
        pair_results: list[PairedGameResult] = []
        if pair_files_valid:
            seen_ordinals: set[int] = set()
            for pair in pair_artifacts:
                if not isinstance(pair, Mapping):
                    pair_files_valid = False
                    break
                ordinal = pair.get("ordinal")
                file_name = pair.get("file")
                if (
                    type(ordinal) is not int
                    or ordinal in seen_ordinals
                    or not 0 <= ordinal < expected_pairs
                    or not isinstance(file_name, str)
                    or Path(file_name).name != file_name
                    or type(pair.get("bytes")) is not int
                    or pair.get("bytes") <= 0
                ):
                    pair_files_valid = False
                    break
                seen_ordinals.add(ordinal)
                pair_path = result_path.parent / file_name
                try:
                    pair_digest = validate_sha256(
                        pair.get("sha256"), "pair artifact sha256"
                    )
                except ValueError:
                    pair_files_valid = False
                    break
                if (
                    not pair_path.is_file()
                    or pair_path.is_symlink()
                    or pair_path.stat().st_size != pair.get("bytes")
                    or sha256_file(pair_path) != pair_digest
                ):
                    pair_files_valid = False
                    break
                pair_results.extend(
                    _validate_pair_evidence(
                        pair_path,
                        pair,
                        result=result,
                        candidate_identity=candidate_identity,
                        opponent_identity=opponent_identity,
                        candidate_runtime=candidate_runtime,
                        opponent_runtime=opponent_runtime,
                    )
                )
            pair_files_valid = pair_files_valid and seen_ordinals == set(
                range(expected_pairs)
            )
        recomputed_statistics = (
            summarize_paired_results(
                pair_results,
                seed=PHASE4_ARENA_BOOTSTRAP_SEED,
                bootstrap_samples=PHASE4_ARENA_BOOTSTRAP_SAMPLES,
                confidence=PHASE4_ARENA_CONFIDENCE,
            ).to_dict()
            if pair_files_valid and len(pair_results) == expected_games
            else None
        )
        statistics = result.get("statistics")
        if not isinstance(statistics, Mapping):
            raise Phase4PromotionEvidenceError("arena statistics are missing")
        interval = statistics.get("winRateCI")
        if not isinstance(interval, Mapping):
            raise Phase4PromotionEvidenceError("arena confidence interval is missing")
        win_rate = statistics.get("winRate")
        lower = interval.get("lower")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (win_rate, lower)
        ):
            raise Phase4PromotionEvidenceError("arena strength statistics are invalid")
        bootstrap = statistics.get("bootstrap")
        candidate_wins = statistics.get("candidateWins")
        candidate_losses = statistics.get("candidateLosses")
        draws = statistics.get("draws")
        upper = interval.get("upper")
        statistics_valid = (
            type(candidate_wins) is int
            and type(candidate_losses) is int
            and candidate_wins >= 0
            and candidate_losses >= 0
            and type(draws) is int
            and draws == 0
            and candidate_wins + candidate_losses == expected_games
            and math.isclose(
                float(win_rate),
                candidate_wins / expected_games,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and isinstance(upper, (int, float))
            and not isinstance(upper, bool)
            and math.isfinite(float(upper))
            and 0.0 <= float(lower) <= float(win_rate) <= float(upper) <= 1.0
            and interval.get("confidence") == PHASE4_ARENA_CONFIDENCE
            and isinstance(bootstrap, Mapping)
            and bootstrap.get("method") == "opening-pair-cluster-percentile"
            and bootstrap.get("seed") == PHASE4_ARENA_BOOTSTRAP_SEED
            and bootstrap.get("samples") == PHASE4_ARENA_BOOTSTRAP_SAMPLES
            and dict(statistics) == recomputed_statistics
        )
        try:
            candidate_backend_sha256 = validate_sha256(
                candidate_runtime.get("backendHash"),
                "candidate runtime backend hash",
            )
            opponent_backend_sha256 = validate_sha256(
                opponent_runtime.get("backendHash"),
                "opponent runtime backend hash",
            )
        except ValueError:
            candidate_backend_sha256 = ""
            opponent_backend_sha256 = "invalid"
        expected_arena_contract_sha256 = phase4_arena_contract_sha256(
            gate=gate,
            execution_contract_sha256=self.execution_contract_sha256,
            candidate_artifact_sha256=candidate_artifact_sha256,
            opponent_artifact_sha256=str(opponent["artifactSha256"]),
            suite_hash=str(expected_suite),
        )
        workload_checks = {
            "contractSha256": (
                result.get("contractSha256")
                == expected_arena_contract_sha256
            ),
            "suiteHash": result.get("suiteHash") == expected_suite,
            "pairArtifacts": pair_files_valid,
            "openingPairs": result.get("openingPairs") == expected_pairs
            and statistics.get("openingPairs") == expected_pairs,
            "games": result.get("games") == expected_games
            and statistics.get("games") == expected_games,
            "candidateIdentity": candidate_identity.get("artifactHash")
            == candidate_artifact_sha256
            and candidate_identity.get("configHash") == candidate["configSha256"]
            and candidate_identity.get("sourceHash") == candidate["sourceSha256"]
            and candidate_identity.get("verified") is True,
            "opponentIdentity": opponent_identity.get("artifactHash")
            == opponent["artifactSha256"]
            and opponent_identity.get("verified") is True,
            "candidateRuntime": candidate_runtime.get("kind") == "native-mcts"
            and candidate_runtime.get("identityBound") is True
            and candidate_runtime.get("configuredSimulations")
            == expected_simulations
            and candidate_runtime.get("trusted") is True
            and candidate_backend_sha256 == opponent_backend_sha256,
            "opponentRuntime": opponent_runtime.get("kind") == "native-mcts"
            and opponent_runtime.get("identityBound") is True
            and opponent_runtime.get("configuredSimulations")
            == expected_simulations
            and opponent_runtime.get("trusted") is True
            and opponent_backend_sha256 == candidate_backend_sha256,
            "statistics": statistics_valid,
        }
        if gate == "screen":
            strength_checks = {
                "winRate": float(win_rate) >= PHASE4_SCREEN_MINIMUM_WIN_RATE,
            }
        elif gate == "confirmation":
            strength_checks = {
                "winRate": float(win_rate)
                >= PHASE4_CONFIRMATION_MINIMUM_WIN_RATE,
                "oneSided95Lower": float(lower)
                > PHASE4_CONFIRMATION_MINIMUM_LOWER,
            }
        else:
            strength_checks = {
                "winRate": float(win_rate) >= PHASE4_HISTORY_MINIMUM_WIN_RATE,
                "oneSided95Lower": float(lower) > PHASE4_HISTORY_MINIMUM_LOWER,
            }
        checks = {**workload_checks, **strength_checks}
        reference = ArtifactReference.from_file(
            result_path,
            relative_to=self.artifact_root,
        )
        core = {
            "schemaVersion": PHASE4_PROMOTION_SCHEMA_VERSION,
            "kind": "phase4e-gate-evidence",
            "gate": gate,
            "opponent": opponent,
            "candidateSha256": candidate["candidateSha256"],
            "passed": all(checks.values()),
            "checks": checks,
            "statistics": {
                "winRate": float(win_rate),
                "oneSided95Lower": float(lower),
            },
            "arenaResult": reference.to_payload(),
            "arenaResultSelfSha256": result_sha256,
        }
        return core

    def _load_gate(
        self,
        gate: str,
        opponent_id: str | None,
    ) -> dict[str, object] | None:
        path = self._gate_path(gate, opponent_id)
        if not path.exists():
            return None
        payload = _load_canonical(path, f"{gate} gate evidence")
        _verify_self_hash(payload, field="gateSha256", label=f"{gate} gate")
        candidate = self.load_candidate()
        if payload.get("candidateSha256") != candidate.get("candidateSha256"):
            raise Phase4PromotionEvidenceError("gate belongs to another candidate")
        if gate == "health":
            expected_core = self._health_gate_core(candidate)
        else:
            result_reference = payload.get("arenaResult")
            if not isinstance(result_reference, Mapping):
                raise Phase4PromotionEvidenceError(
                    "arena gate result reference is missing"
                )
            result_path = self.artifact_root / str(
                result_reference.get("path")
            )
            expected_core = self._arena_gate_core(
                candidate,
                gate,
                result_path,
                opponent_id=opponent_id,
            )
        if payload != _self_hashed(expected_core, "gateSha256"):
            raise Phase4PromotionEvidenceError(
                f"{gate} gate evidence does not reproduce from source artifacts"
            )
        return payload

    def status(self) -> PromotionStatus:
        candidate = self.load_candidate()
        completed: list[str] = []
        health = self._load_gate("health", None)
        if health is None:
            return PromotionStatus("in-progress", "health", None, ())
        completed.append("health")
        if health.get("passed") is not True:
            return PromotionStatus("rejected", None, "health", tuple(completed))
        primary = _opponent_descriptor(candidate["primaryOpponent"])
        for gate in ("screen", "confirmation"):
            evidence = self._load_gate(gate, str(primary["id"]))
            if evidence is None:
                return PromotionStatus("in-progress", gate, None, tuple(completed))
            completed.append(gate)
            if evidence.get("passed") is not True:
                return PromotionStatus("rejected", None, gate, tuple(completed))
        history = candidate.get("historyOpponents")
        if not isinstance(history, list):
            raise Phase4PromotionEvidenceError("candidate history list is invalid")
        for opponent_value in history:
            opponent = _opponent_descriptor(opponent_value)
            gate_name = f"history:{opponent['id']}"
            evidence = self._load_gate("history", str(opponent["id"]))
            if evidence is None:
                return PromotionStatus(
                    "in-progress", gate_name, None, tuple(completed)
                )
            completed.append(gate_name)
            if evidence.get("passed") is not True:
                return PromotionStatus(
                    "rejected", None, gate_name, tuple(completed)
                )
        bundle = self.root / "promotion-bundle.json"
        return PromotionStatus(
            "ready-to-publish" if not bundle.exists() else "bundle-complete",
            "promotion-bundle" if not bundle.exists() else "publish",
            None,
            tuple(completed),
        )

    def build_promotion_bundle(self) -> dict[str, object]:
        status = self.status()
        if status.failed_gate is not None or status.next_gate not in {
            "promotion-bundle",
            "publish",
        }:
            raise Phase4PromotionError("candidate has not passed every promotion gate")
        candidate = self.load_candidate()
        primary = _opponent_descriptor(candidate["primaryOpponent"])
        expected_gates: list[tuple[str, str | None]] = [
            ("health", None),
            ("screen", str(primary["id"])),
            ("confirmation", str(primary["id"])),
        ]
        history = candidate.get("historyOpponents")
        if not isinstance(history, list):
            raise Phase4PromotionEvidenceError(
                "candidate history list is invalid"
            )
        expected_gates.extend(
            ("history", str(_opponent_descriptor(value)["id"]))
            for value in history
        )
        evidence = []
        for gate, opponent_id in expected_gates:
            payload = self._load_gate(gate, opponent_id)
            if payload is None or payload.get("passed") is not True:
                raise Phase4PromotionError(
                    "promotion bundle cannot omit or include a failed gate"
                )
            path = self._gate_path(gate, opponent_id)
            evidence.append(
                {
                    "gate": payload["gate"],
                    "opponent": payload.get("opponent"),
                    "path": path.relative_to(self.artifact_root).as_posix(),
                    "sha256": sha256_file(path),
                    "gateSha256": payload["gateSha256"],
                }
            )
        core = {
            "schemaVersion": PHASE4_PROMOTION_SCHEMA_VERSION,
            "kind": "phase4e-promotion-bundle",
            "status": "accepted",
            "formalPromotion": True,
            "candidateSha256": candidate["candidateSha256"],
            "executionContractSha256": self.execution_contract_sha256,
            "generation": candidate["generation"],
            "modelStateSha256": candidate["modelStateSha256"],
            "gates": evidence,
        }
        payload = _self_hashed(core, "promotionBundleSha256")
        path = self.root / "promotion-bundle.json"
        encoded = canonical_json_bytes(payload)
        if path.exists() and path.read_bytes() != encoded:
            raise Phase4PromotionEvidenceError("promotion bundle already differs")
        if not path.exists():
            atomic_replace_bytes(path, encoded)
        return payload

    def publish(self) -> ChampionRecord:
        bundle = self.build_promotion_bundle()
        candidate = self.load_candidate()
        bundle_path = self.root / "promotion-bundle.json"
        bundle_file_sha256 = sha256_file(bundle_path)
        for record in self.champion_store.records:
            if (
                record.metadata.get("candidateModelStateSha256")
                == candidate["modelStateSha256"]
                and record.metadata.get("promotionBundleFileSha256")
                == bundle_file_sha256
            ):
                return record
        primary = _opponent_descriptor(candidate["primaryOpponent"])
        latest = self.champion_store.pointers["latest"]
        expected_parent = (
            primary["championId"] if primary["kind"] == "champion" else None
        )
        if latest != expected_parent:
            raise Phase4PromotionError(
                "champion lineage changed after candidate evaluation"
            )
        checkpoint = candidate["checkpoint"]
        run_manifest = candidate["runManifest"]
        state, lineage = _load_training_checkpoint_evidence(
            self.artifact_root / str(checkpoint["path"]),
            run_manifest_path=self.artifact_root
            / str(run_manifest["path"]),
            artifact_root=self.artifact_root,
            execution_contract_sha256=self.execution_contract_sha256,
        )
        if (
            lineage["modelStateSha256"] != candidate["modelStateSha256"]
            or lineage["candidateModel"] != candidate["candidateModel"]
        ):
            raise Phase4PromotionEvidenceError(
                "candidate model changed before champion publication"
            )
        arena_reference = ArtifactReference.from_file(
            bundle_path,
            relative_to=self.artifact_root,
        )
        return self.champion_store.publish(
            state,
            generation=int(candidate["generation"]),
            parent_id=latest,
            config_sha256=str(candidate["configSha256"]),
            source_sha256=str(candidate["sourceSha256"]),
            opening_suite_sha256=str(candidate["confirmationSuiteSha256"]),
            arena_report=arena_reference,
            milestone=False,
            best=False,
            product=False,
            metadata=normalize_json_mapping(
                {
                    "formalPromotion": True,
                    "candidateSha256": candidate["candidateSha256"],
                    "candidateCheckpointSha256": checkpoint["sha256"],
                    "candidateModelArtifactSha256": candidate[
                        "candidateModel"
                    ]["sha256"],
                    "candidateModelStateSha256": candidate["modelStateSha256"],
                    "executionContractSha256": self.execution_contract_sha256,
                    "promotionBundleSha256": bundle["promotionBundleSha256"],
                    "promotionBundleFileSha256": bundle_file_sha256,
                    "productRelease": False,
                    "bestByElo": False,
                },
                "promotion metadata",
            ),
        )


__all__ = [
    "PHASE4_PROMOTION_SCHEMA_VERSION",
    "PHASE4_CANDIDATE_CADENCE_GENERATIONS",
    "PHASE4_SCREEN_OPENING_PAIRS",
    "PHASE4_SCREEN_GAMES",
    "PHASE4_SCREEN_SIMULATIONS",
    "PHASE4_SCREEN_MINIMUM_WIN_RATE",
    "PHASE4_CONFIRMATION_OPENING_PAIRS",
    "PHASE4_CONFIRMATION_GAMES",
    "PHASE4_CONFIRMATION_SIMULATIONS",
    "PHASE4_CONFIRMATION_MINIMUM_WIN_RATE",
    "PHASE4_CONFIRMATION_MINIMUM_LOWER",
    "PHASE4_HISTORY_MINIMUM_WIN_RATE",
    "PHASE4_HISTORY_MINIMUM_LOWER",
    "PHASE4_HISTORY_RECENT",
    "PHASE4_ARENA_BOOTSTRAP_SEED",
    "PHASE4_ARENA_BOOTSTRAP_SAMPLES",
    "PHASE4_ARENA_CONFIDENCE",
    "PHASE4_SCREEN_SUITE_HASH",
    "PHASE4_CONFIRMATION_SUITE_HASH",
    "Phase4PromotionError",
    "Phase4PromotionEvidenceError",
    "PromotionStatus",
    "bootstrap_opponent_descriptor",
    "champion_opponent_descriptor",
    "phase4_arena_contract_sha256",
    "select_history_opponents",
    "Phase4PromotionController",
]
