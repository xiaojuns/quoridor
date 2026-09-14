"""Run the frozen F51 learner-occupancy and classic-root target probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import torch

from .calibration import GPU_LEDGER_NAME, GpuBudgetLedger
from .challenge_replay import arena_position_orbits, position_orbit_digest
from .classic_bridge import ClassicAIBridge, ClassicSearchBudget
from .codec import mirror_action_index, position_hash
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .game import legal_action_indices
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .network import NetworkConfig, PolicyValueNetwork
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4_checkpoint import canonical_model_state_sha256
from .phase4f29_fresh_screen import _mapping, _self_hashed, _verify_self_hash, _write_immutable
from .phase4f43_curriculum_pilot import _load_f42_evidence, load_curriculum_samples
from .phase4f45_curriculum_pilot_review import _f21_evidence
from .phase4f50_learner_teacher_preparation import verify_phase4f50
from .replay_store import ReplayGame, ReplayPly, ReplayStore
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    RunLock,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .selfplay import SelfPlayConfig, SelfPlayGame, play_self_games_scheduled
from .storage import directory_size


SCHEMA_VERSION = 1
CONTRACT = Path("training/configs/phase4f51-learner-occupancy-teacher-root-probe-frozen-v1.json")
EXPECTED_CONTRACT_SHA256 = "6d5419c56341a071c760d3334616f4408cb1fb63dfc13059aaade5d8406125fb"
ROOT = Path("training/runs/phase4/phase4f/f51-learner-teacher-root-probe-v1")
EXCLUSION = ROOT / "exclusion-inventory.json"
SELECTION = ROOT / "position-selection.json"
TEACHER_ROOT = ROOT / "teacher-root"
TARGET_RESULT = ROOT / "target-probe-result.json"
REPORT = ROOT / "probe-report.json"
GATE = ROOT / "probe-gate.json"
RUN_ID_PREFIX = "phase4f51-learner-teacher-root-probe"
MODEL_ID = "f21-stage-a-best-g00000001-s000000000192"
ROLLOUT_ACTORS = 64
CLASSIC_ACTORS = min(8, max(1, (os.cpu_count() or 2) // 2))
CLASSIC_CHUNK = 16
STABILITY_CHUNK = 8
STORE_BYTES = {"train": 384 * 1024**2, "development": 128 * 1024**2}
SPLIT_ORDER = ("train", "development")


class Phase4F51Error(RuntimeError):
    """Raised when the frozen F51 probe cannot be completed safely."""


def _progress(stage: str, status: str, **fields: object) -> None:
    print(canonical_json_bytes({
        "kind": "phase4f51-learner-teacher-probe-progress",
        "schemaVersion": SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        **fields,
    }).decode("ascii"), end="", flush=True)


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F51Error("workspace does not contain training/quoridor")
    return workspace


def _source(workspace: Path) -> dict[str, object]:
    return fingerprint_sources(
        workspace, paths=("training/quoridor/phase4f51_learner_teacher_probe.py",)
    ).to_payload()


def _contract(workspace: Path) -> dict[str, object]:
    verification = verify_phase4f50(workspace)
    contract = _mapping(workspace / CONTRACT, "F51 contract")
    digest = _verify_self_hash(contract, "contractSha256", "F51 contract")
    if (
        verification.get("executionContractSha256") != EXPECTED_CONTRACT_SHA256
        or digest != EXPECTED_CONTRACT_SHA256
        or contract.get("status") != "frozen"
        or contract.get("authorizations", {}).get("learnerOccupancyProbeGenerationAuthorized") is not True
        or contract.get("authorizations", {}).get("largeDatasetGenerationAuthorized") is not False
        or contract.get("authorizations", {}).get("cudaTrainingAuthorized") is not False
    ):
        raise Phase4F51Error("F50 does not authorize the frozen F51 probe")
    inputs = contract["inputs"]
    for field, relative in (
        ("f21ModelFileSha256", inputs["f21ModelPath"]),
        ("f21ConfigFileSha256", inputs["f21ConfigPath"]),
    ):
        if sha256_file(workspace / str(relative)) != inputs[field]:
            raise Phase4F51Error(f"frozen F51 input changed: {relative}")
    classic = inputs["classicImplementation"]
    for field, path_field in (
        ("aiSourceSha256", "aiSourcePath"),
        ("bridgeSha256", "bridgePath"),
        ("pythonAdapterSha256", "pythonAdapterPath"),
    ):
        if sha256_file(workspace / str(classic[path_field])) != classic[field]:
            raise Phase4F51Error(f"frozen classic implementation changed: {classic[path_field]}")
    return contract


def _resources(workspace: Path, contract: Mapping[str, object]) -> dict[str, int]:
    resources = contract["resources"]
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    output_bytes = directory_size(workspace / ROOT) if (workspace / ROOT).exists() else 0
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes >= int(resources["artifactLimitBytes"]):
        raise Phase4F51Error("F51 reached the 16 GiB artifact cap")
    if output_bytes >= int(resources["maximumOutputBytes"]):
        raise Phase4F51Error("F51 reached its 768 MiB output cap")
    if free_bytes < int(resources["minimumFreeDiskBytes"]):
        raise Phase4F51Error("F51 reached the 20 GiB free-disk stop line")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "f51OutputBytes": output_bytes,
        "f51OutputLimitBytes": int(resources["maximumOutputBytes"]),
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _arena_references(workspace: Path) -> list[dict[str, object]]:
    references: list[dict[str, object]] = []
    control = workspace / PHASE4_CONTROL_RELATIVE
    for path in sorted(control.glob("**/result.json")):
        relative = path.relative_to(workspace)
        lowered = relative.as_posix().lower()
        if "screen" not in lowered and "confirmation" not in lowered:
            continue
        try:
            payload = _mapping(path, "historical arena result")
        except Exception:
            continue
        if payload.get("status") != "completed" or not isinstance(payload.get("pairArtifacts"), list):
            continue
        if not all(
            isinstance(item, Mapping)
            and isinstance(item.get("file"), str)
            and (path.parent / str(item["file"])).is_file()
            for item in payload["pairArtifacts"]
        ):
            # Early arena wrappers copied result.json one directory above the
            # authoritative store. The store/result.json copy is scanned too.
            continue
        result_sha = payload.get("resultSha256")
        core = dict(payload)
        core.pop("resultSha256", None)
        if not isinstance(result_sha, str) or result_sha != sha256_bytes(canonical_json_bytes(core)):
            continue
        references.append({
            "path": relative.as_posix(),
            "fileSha256": sha256_file(path),
            "resultSha256": result_sha,
        })
    if not references:
        raise Phase4F51Error("no completed historical screen/confirmation arenas were found")
    return references


def _f42_orbits(workspace: Path) -> tuple[set[str], dict[str, object]]:
    gate, manifest = _load_f42_evidence(workspace)
    samples = load_curriculum_samples(workspace, manifest, "train")
    samples.extend(load_curriculum_samples(workspace, manifest, "development"))
    orbits = {str(sample["source"]["teacherOrbitSha256"]) for sample in samples}
    if len(orbits) != len(samples):
        raise Phase4F51Error("F42 exclusion inventory contains duplicate orbits")
    return orbits, {
        "gateSha256": gate["gateSha256"],
        "manifestSha256": manifest["manifestSha256"],
        "orbits": len(orbits),
    }


def _build_or_load_exclusion(workspace: Path, contract_sha: str, source_sha: str) -> tuple[set[str], dict[str, object]]:
    path = workspace / EXCLUSION
    if path.exists():
        payload = _mapping(path, "F51 exclusion inventory")
        _verify_self_hash(payload, "inventorySha256", "F51 exclusion inventory")
        if payload.get("contractSha256") != contract_sha:
            raise Phase4F51Error("F51 exclusion inventory belongs to a different contract")
        arena_orbits: set[str] = set()
        for reference in payload["arenaResults"]:
            arena_orbits.update(arena_position_orbits(workspace, reference))
        f42_orbits, _ = _f42_orbits(workspace)
        combined = arena_orbits | f42_orbits
        if (
            len(combined) != payload.get("combinedOrbits")
            or sha256_bytes(canonical_json_bytes(sorted(combined))) != payload.get("combinedOrbitSha256")
        ):
            raise Phase4F51Error("F51 exclusion orbit inventory changed")
        return combined, payload

    references = _arena_references(workspace)
    arena_orbits: set[str] = set()
    for index, reference in enumerate(references, 1):
        arena_orbits.update(arena_position_orbits(workspace, reference))
        if index % 8 == 0 or index == len(references):
            _progress("exclusion-audit", "running", arenaResults=index, arenaResultsTotal=len(references), arenaOrbits=len(arena_orbits))
    f42_orbits, f42 = _f42_orbits(workspace)
    combined = arena_orbits | f42_orbits
    payload = _self_hashed({
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f51-content-hashed-exclusion-orbit-inventory",
        "status": "completed",
        "contractSha256": contract_sha,
        "sourceSha256": source_sha,
        "algorithm": "sha256-sorted-horizontal-position-orbits-v1",
        "arenaResults": references,
        "arenaResultCount": len(references),
        "arenaOrbits": len(arena_orbits),
        "f42": f42,
        "combinedOrbits": len(combined),
        "combinedOrbitSha256": sha256_bytes(canonical_json_bytes(sorted(combined))),
    }, "inventorySha256")
    _write_immutable(path, payload, "F51 exclusion inventory")
    return combined, payload


def _network_config(workspace: Path, contract: Mapping[str, object]) -> NetworkConfig:
    config = load_experiment_config(workspace / str(contract["inputs"]["f21ConfigPath"]))
    return NetworkConfig(
        input_channels=config.network.input_channels,
        channels=config.network.channels,
        blocks=config.network.residual_blocks,
        se_reduction=config.network.se_reduction,
        policy_channels=config.network.policy_channels,
        value_channels=config.network.value_channels,
        value_hidden=config.network.value_hidden,
    )


def _load_f21_model(workspace: Path, contract: Mapping[str, object], device: torch.device) -> PolicyValueNetwork:
    _payload, state = _f21_evidence(workspace)
    if canonical_model_state_sha256(state) != contract["inputs"]["f21ModelStateSha256"]:
        raise Phase4F51Error("F21 learner state hash changed")
    model = PolicyValueNetwork(_network_config(workspace, contract))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _open_store(workspace: Path, split: str, games: int) -> ReplayStore:
    return ReplayStore(
        workspace / ROOT / "rollouts" / split,
        max_games=games,
        max_plies=games * 128,
        max_bytes=STORE_BYTES[split],
    )


def _replay_from_selfplay(game: SelfPlayGame, split: str, seed: int, game_index: int) -> ReplayGame:
    if len(game.raw_plies) != game.move_count or game.move_count > 256:
        raise Phase4F51Error("learner rollout trajectory is incomplete or exceeds the rule limit")
    return ReplayGame(
        game_id=f"f51-{split}-game-{game_index:06d}",
        initial_state=game.initial_state,
        plies=tuple(ReplayPly(
            action_index=ply.action_index,
            visit_counts=ply.visit_counts,
            root_value=ply.root_value,
            model_generation=1,
            model_id=MODEL_ID,
            metadata={"requestedSimulations": 800},
        ) for ply in game.raw_plies),
        metadata={
            "phase": "4F51",
            "split": split,
            "gameIndex": game_index,
            "schedulerSeed": seed,
            "modelStateSha256": "5cbff6245585e7452fec2869cf2e4c90f983744e5a0284d8059c263e6f98a527",
            "winner": game.winner,
            "terminationReason": game.termination_reason,
            "probeEligiblePlies": min(game.move_count, 128),
            "probeExcludedTailPlies": max(0, game.move_count - 128),
        },
    )


def _validate_store(store: ReplayStore, split: str, games: int, seed: int) -> dict[str, int]:
    verified_games = 0

    def validator(game: ReplayGame, states: Sequence[object]) -> None:
        nonlocal verified_games
        index = int(game.metadata.get("gameIndex", -1))
        eligible_plies = game.metadata.get(
            "probeEligiblePlies", min(len(states) - 1, 128)
        )
        excluded_tail = game.metadata.get(
            "probeExcludedTailPlies", max(0, len(states) - 1 - 128)
        )
        if (
            game.game_id != f"f51-{split}-game-{index:06d}"
            or game.metadata.get("split") != split
            or game.metadata.get("schedulerSeed") != seed
            or len(states) - 1 > 256
            or eligible_plies != min(len(states) - 1, 128)
            or excluded_tail != max(0, len(states) - 1 - 128)
            or any(ply.model_id != MODEL_ID or ply.metadata.get("requestedSimulations") != 800 for ply in game.plies)
        ):
            raise Phase4F51Error("F51 rollout store identity differs from the frozen invocation")
        verified_games += 1
        if verified_games % 32 == 0 or verified_games == store.game_count:
            _progress(
                "rollout-resume-validation",
                "running",
                split=split,
                games=verified_games,
                gamesTotal=store.game_count,
            )
    verification = store.verify(deep=True, game_validator=validator)
    if verification.game_count > games:
        raise Phase4F51Error("F51 rollout store exceeds its frozen game count")
    return {"games": verification.game_count, "plies": verification.ply_count, "bytes": verification.committed_bytes}


def _store_summary(store: ReplayStore) -> dict[str, int]:
    verification = store.verify(deep=False)
    return {
        "games": verification.game_count,
        "plies": verification.ply_count,
        "bytes": verification.committed_bytes,
    }


def _run_rollouts(
    workspace: Path,
    contract: Mapping[str, object],
    device: torch.device,
    ledger: GpuBudgetLedger,
    run_id: str,
) -> tuple[dict[str, ReplayStore], float]:
    rollout = contract["rollout"]
    stores = {split: _open_store(workspace, split, int(rollout["games"][split])) for split in SPLIT_ORDER}
    for split, store in stores.items():
        _validate_store(store, split, int(rollout["games"][split]), int(rollout["seeds"][split]))
    if all(stores[s].game_count == int(rollout["games"][s]) for s in SPLIT_ORDER):
        return stores, 0.0
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F51Error("F51 CPU preflight completed; learner-occupancy rollout requires visible CUDA")
    native = load_native_module()
    if str(contract["inputs"]["nativeBackendSha256"]) != __import__("quoridor.paired_arena", fromlist=["loaded_native_backend_hash"]).loaded_native_backend_hash():
        raise Phase4F51Error("native encoder/search backend differs from the F51 contract")
    model = _load_f21_model(workspace, contract, device)
    evaluator = TorchEvaluator(model, device=device, use_amp=True)
    evaluator.enable_native_encoder(native)
    maximum = float(contract["resources"]["maximumGlobalGpuHours"])
    if ledger.total_hours >= maximum:
        raise Phase4F51Error("F51 probe GPU budget is exhausted")
    ledger.begin(run_id, external_hours=0.0)
    started = time.monotonic()
    last_update = started
    try:
        for split in SPLIT_ORDER:
            store = stores[split]
            total = int(rollout["games"][split])
            seed = int(rollout["seeds"][split])
            missing = [index for index in range(total) if f"f51-{split}-game-{index:06d}" not in store.game_ids]
            if not missing:
                continue
            config = SelfPlayConfig(
                mcts=MCTSConfig(
                    simulations=800,
                    c_base=float(rollout["mcts"]["cBase"]),
                    c_init=float(rollout["mcts"]["cInit"]),
                    dirichlet_epsilon=0.25,
                    temperature=1.0,
                    add_root_noise=True,
                    seed=seed,
                ),
                temperature_moves=24,
                mirror_augmentation=False,
                seed=seed,
                input_channels=_network_config(workspace, contract).input_channels,
                collect_samples=False,
                collect_raw_records=True,
            )
            last_games = store.game_count

            def heartbeat(moves: int, _games: int) -> None:
                nonlocal last_update, last_games
                now = time.monotonic()
                if now - last_update >= 15:
                    current = ledger.update(run_id, elapsed_seconds=now - started, status="running")
                    _resources(workspace, contract)
                    if current >= maximum:
                        raise Phase4F51Error("F51 probe GPU budget is exhausted")
                    if store.game_count != last_games:
                        last_games = store.game_count
                        _progress("learner-rollout", "running", split=split, games=store.game_count, gamesTotal=total, moves=moves, globalGpuHours=current)
                    last_update = now

            def persist(local_index: int, game: SelfPlayGame) -> None:
                actual = missing[local_index]
                store.append(_replay_from_selfplay(game, split, seed, actual))

            play_self_games_scheduled(
                evaluator, config, len(missing),
                actors=min(ROLLOUT_ACTORS, len(missing)),
                search_factory=NativeMCTS,
                progress_callback=heartbeat,
                completed_game_callback=persist,
            )
            if store.game_count != total:
                raise Phase4F51Error("learner rollout did not commit every frozen game")
            _store_summary(store)
            _progress("learner-rollout", "split-completed", split=split, games=store.game_count, gamesTotal=total, plies=store.ply_count)
        elapsed = time.monotonic() - started
        total_hours = ledger.update(run_id, elapsed_seconds=elapsed, status="completed")
        if total_hours > maximum:
            raise Phase4F51Error("F51 completed after exceeding its frozen GPU budget")
        return stores, elapsed
    except BaseException:
        try:
            ledger.update(run_id, elapsed_seconds=time.monotonic() - started, status="interrupted")
        except Exception:
            pass
        raise


def ply_band(move_number: int) -> str:
    if move_number < 0:
        raise ValueError("move number must be non-negative")
    return "early" if move_number < 16 else "late" if move_number >= 40 else "middle"


def select_probe_positions(
    stores: Mapping[str, ReplayStore],
    excluded: set[str],
    quotas: Mapping[str, Mapping[str, int]],
    seeds: Mapping[str, int],
) -> tuple[list[tuple[object, dict[str, object]]], dict[str, object]]:
    selected: list[tuple[object, dict[str, object]]] = []
    used: set[str] = set()
    inventory_counts: Counter[tuple[str, str]] = Counter()
    excluded_hits: Counter[str] = Counter()
    overlength_games: Counter[str] = Counter()
    excluded_tail_plies: Counter[str] = Counter()
    for split in SPLIT_ORDER:
        candidates: dict[str, tuple[str, object, dict[str, object]]] = {}
        store = stores[split]
        for game_id in store.game_ids:
            game, states = store.load_game_with_states(game_id)
            source_plies = len(states) - 1
            if source_plies > 128:
                overlength_games[split] += 1
                excluded_tail_plies[split] += source_plies - 128
            # ReplayStore retains the complete rule-valid game so its winner
            # and termination can be deeply verified. The frozen probe only
            # admits the first 128 decisions from any source game.
            for ply_index, state in enumerate(states[:-1][:128]):
                band = ply_band(int(state.move_number))
                orbit = position_orbit_digest(state)
                inventory_counts[(split, band)] += 1
                if orbit in excluded:
                    excluded_hits[split] += 1
                    continue
                if orbit in used or orbit in candidates:
                    continue
                rank = sha256_bytes(f"{seeds[split]}\0{band}\0{game_id}\0{ply_index}\0{orbit}".encode("ascii"))
                metadata = {
                    "split": split,
                    "band": band,
                    "gameId": game_id,
                    "plyIndex": ply_index,
                    "positionOrbit": orbit,
                    "positionHash": position_hash(state),
                    "absoluteMoveNumber": int(state.move_number),
                    "currentPlayer": int(state.current_player),
                    "rank": rank,
                }
                candidates[orbit] = (rank, state, metadata)
        for band in ("early", "middle", "late"):
            count = int(quotas[split][band])
            band_rows = sorted((value for value in candidates.values() if value[2]["band"] == band), key=lambda item: (item[0], item[2]["positionOrbit"]))
            if len(band_rows) < count:
                raise Phase4F51Error(f"F51 {split}/{band} has {len(band_rows)} unique eligible orbits; needs {count}")
            for _rank, state, metadata in band_rows[:count]:
                metadata = {**metadata, "ordinal": len(selected)}
                selected.append((state, metadata))
                used.add(str(metadata["positionOrbit"]))
    payload = {
        "inventoryBySplitBand": {
            split: {band: inventory_counts[(split, band)] for band in ("early", "middle", "late")}
            for split in SPLIT_ORDER
        },
        "excludedHits": dict(excluded_hits),
        "sourceGamesOver128Plies": {
            split: overlength_games[split] for split in SPLIT_ORDER
        },
        "sourceTailPliesExcludedFromProbe": {
            split: excluded_tail_plies[split] for split in SPLIT_ORDER
        },
        "uniqueSelectedOrbits": len(used),
    }
    return selected, payload


def _selection_payload(contract_sha: str, source_sha: str, rows: Sequence[tuple[object, Mapping[str, object]]], audit: Mapping[str, object]) -> dict[str, object]:
    return _self_hashed({
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f51-learner-occupancy-position-selection",
        "status": "completed",
        "contractSha256": contract_sha,
        "sourceSha256": source_sha,
        "algorithm": "sha256-split-seed-band-game-ply-orbit-lowest-v1",
        "positions": len(rows),
        "audit": dict(audit),
        "rows": [dict(metadata) for _state, metadata in rows],
    }, "selectionSha256")


def select_stability_ordinals(rows: Sequence[Mapping[str, object]], count: int = 128) -> list[int]:
    development = [row for row in rows if row["split"] == "development"]
    desired = {"early": 64, "middle": 48, "late": 16}
    selected: list[int] = []
    for band, total in desired.items():
        by_side = {side: sorted((row for row in development if row["band"] == band and row["currentPlayer"] == side), key=lambda row: (row["rank"], row["positionOrbit"])) for side in (0, 1)}
        side_counts = [total // 2, total - total // 2]
        for side, side_count in enumerate(side_counts):
            if len(by_side[side]) < side_count:
                raise Phase4F51Error("development split cannot satisfy stability strata")
            selected.extend(int(row["ordinal"]) for row in by_side[side][:side_count])
    if len(selected) != count:
        raise Phase4F51Error("stability selection count differs from the contract")
    return sorted(selected)


def _classic_analysis(state: object, limit: int) -> dict[str, object]:
    with ClassicAIBridge(request_timeout=300.0) as bridge:
        result = bridge.analyze(state, budget=ClassicSearchBudget.nodes(limit), max_depth=5, timeout=300.0)
    scores = [{"actionIndex": item.action_index, "searched": item.searched, "score": item.score} for item in result.root_scores]
    return {
        "format": "legal-action-root-scores-v1",
        "budget": {"kind": "nodes", "limit": limit, "maxDepth": 5},
        "actionIndex": result.action_index,
        "score": result.score,
        "depth": result.depth,
        "nodes": result.nodes,
        "elapsedMs": result.elapsed_ms,
        "rootScoreKind": result.root_score_kind,
        "legalActions": len(scores),
        "searchedActions": sum(bool(item["searched"]) for item in scores),
        "rootScores": scores,
    }


def _teacher_chunk_path(kind: str, chunk_index: int) -> Path:
    return TEACHER_ROOT / kind / f"chunk-{chunk_index:04d}.json"


def _run_teacher_chunks(
    workspace: Path,
    selected: Sequence[tuple[object, Mapping[str, object]]],
    ordinals: Sequence[int],
    kind: str,
    limit: int,
    contract_sha: str,
    source_sha: str,
    selection_sha: str,
) -> list[dict[str, object]]:
    chunk_size = STABILITY_CHUNK if kind == "stability-16384" else CLASSIC_CHUNK
    chunks = [list(ordinals[start:start + chunk_size]) for start in range(0, len(ordinals), chunk_size)]
    references: list[dict[str, object] | None] = [None] * len(chunks)

    def build(chunk_index: int, chunk: Sequence[int]) -> tuple[int, dict[str, object]]:
        relative = _teacher_chunk_path(kind, chunk_index)
        path = workspace / relative
        if path.exists():
            payload = _mapping(path, "F51 teacher chunk")
            _verify_self_hash(payload, "chunkSha256", "F51 teacher chunk")
            if payload.get("contractSha256") != contract_sha or payload.get("sourceSha256") != source_sha or payload.get("selectionSha256") != selection_sha or payload.get("ordinals") != list(chunk):
                raise Phase4F51Error("F51 teacher chunk identity differs")
            return chunk_index, payload
        records = []
        for ordinal in chunk:
            state, metadata = selected[ordinal]
            records.append({
                "ordinal": ordinal,
                "positionHash": metadata["positionHash"],
                "analysis": _classic_analysis(state, limit),
            })
        payload = _self_hashed({
            "schemaVersion": SCHEMA_VERSION,
            "kind": f"phase4f51-{kind}-teacher-root-chunk",
            "status": "completed",
            "contractSha256": contract_sha,
            "sourceSha256": source_sha,
            "selectionSha256": selection_sha,
            "chunkIndex": chunk_index,
            "ordinals": list(chunk),
            "records": records,
        }, "chunkSha256")
        _write_immutable(path, payload, "F51 teacher chunk")
        return chunk_index, payload

    with ThreadPoolExecutor(max_workers=CLASSIC_ACTORS) as pool:
        futures = {pool.submit(build, index, chunk): index for index, chunk in enumerate(chunks)}
        completed = 0
        for future in as_completed(futures):
            index, payload = future.result()
            relative = _teacher_chunk_path(kind, index)
            references[index] = {"path": relative.as_posix(), "sha256": payload["chunkSha256"], "positions": len(payload["records"]), "firstOrdinal": payload["ordinals"][0], "lastOrdinal": payload["ordinals"][-1]}
            completed += len(payload["records"])
            _progress("classic-root", "running", analysis=kind, positions=completed, positionsTotal=len(ordinals))
            _resources(workspace, _mapping(workspace / CONTRACT, "F51 contract"))
    return [dict(reference) for reference in references if reference is not None]


def _load_analysis_records(workspace: Path, references: Sequence[Mapping[str, object]]) -> dict[int, Mapping[str, object]]:
    records: dict[int, Mapping[str, object]] = {}
    for reference in references:
        chunk = _mapping(workspace / str(reference["path"]), "F51 teacher chunk")
        if _verify_self_hash(chunk, "chunkSha256", "F51 teacher chunk") != reference["sha256"]:
            raise Phase4F51Error("F51 teacher chunk reference changed")
        for record in chunk["records"]:
            ordinal = int(record["ordinal"])
            if ordinal in records:
                raise Phase4F51Error("F51 teacher record ordinal is duplicated")
            records[ordinal] = record["analysis"]
    return records


def _ranked_scores(analysis: Mapping[str, object]) -> list[tuple[int, float]]:
    return sorted(
        ((int(row["actionIndex"]), float(row["score"])) for row in analysis["rootScores"] if row["searched"] and row["score"] is not None),
        key=lambda item: (-item[1], item[0]),
    )


def tie_preserving_rank_target(analysis: Mapping[str, object], temperature: float, *, top_k: int = 12, searched_mass: float = 0.97) -> dict[int, float]:
    if temperature <= 0 or not 0 < searched_mass <= 1:
        raise ValueError("target calibration parameters are invalid")
    legal = [int(row["actionIndex"]) for row in analysis["rootScores"]]
    ranked = _ranked_scores(analysis)
    if not ranked or set(legal) != set(int(row["actionIndex"]) for row in analysis["rootScores"]):
        raise Phase4F51Error("classic analysis has an invalid legal action inventory")
    cutoff_score = ranked[min(top_k, len(ranked)) - 1][1]
    included = [(action, score) for action, score in ranked if score >= cutoff_score]
    unique_scores = sorted({score for _action, score in included}, reverse=True)
    score_rank = {score: rank for rank, score in enumerate(unique_scores)}
    raw = {action: math.exp(-score_rank[score] / temperature) for action, score in included}
    included_mass = 1.0 if len(raw) == len(legal) else searched_mass
    scale = included_mass / sum(raw.values())
    policy = {action: value * scale for action, value in raw.items()}
    excluded = [action for action in legal if action not in raw]
    if excluded:
        smoothing = (1.0 - included_mass) / len(excluded)
        policy.update({action: smoothing for action in excluded})
    return policy


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    location = (len(ordered) - 1) * quantile
    lower, upper = math.floor(location), math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_targets(records: Sequence[Mapping[str, object]], temperature: float) -> dict[str, object]:
    entropies, top1, maximum_group, top5_mass, mass_errors, mirror_errors = ([] for _ in range(6))
    for analysis in records:
        policy = tie_preserving_rank_target(analysis, temperature)
        ranked = _ranked_scores(analysis)
        maximum_score = ranked[0][1]
        maximum_actions = {action for action, score in ranked if score == maximum_score}
        top5 = {action for action, _score in ranked[:5]}
        entropies.append(-sum(value * math.log(value) for value in policy.values() if value > 0))
        top1.append(max(policy.values()))
        maximum_group.append(sum(policy.get(action, 0.0) for action in maximum_actions))
        top5_mass.append(sum(policy.get(action, 0.0) for action in top5))
        mass_errors.append(abs(sum(policy.values()) - 1.0))
        mirrored = {mirror_action_index(action): probability for action, probability in policy.items()}
        restored = {mirror_action_index(action): probability for action, probability in mirrored.items()}
        mirror_errors.append(max(abs(policy.get(action, 0.0) - restored.get(action, 0.0)) for action in set(policy) | set(restored)))
    return {
        "temperature": temperature,
        "positions": len(records),
        "meanEntropy": mean(entropies),
        "top1ProbabilityP95": _percentile(top1, 0.95),
        "classicMaximumScoreGroupMassMean": mean(maximum_group),
        "classicTop5MassMean": mean(top5_mass),
        "policyMassMaximumAbsoluteError": max(mass_errors),
        "mirrorMaximumAbsoluteError": max(mirror_errors),
    }


def target_checks(summary: Mapping[str, object], acceptance: Mapping[str, object]) -> dict[str, bool]:
    return {
        "meanEntropyMinimum": float(summary["meanEntropy"]) >= float(acceptance["meanEntropyMinimum"]),
        "meanEntropyMaximum": float(summary["meanEntropy"]) <= float(acceptance["meanEntropyMaximum"]),
        "top1ProbabilityP95": float(summary["top1ProbabilityP95"]) <= float(acceptance["top1ProbabilityP95Maximum"]),
        "classicMaximumScoreGroupMass": float(summary["classicMaximumScoreGroupMassMean"]) >= float(acceptance["classicMaximumScoreGroupMassMinimum"]),
        "classicTop5Mass": float(summary["classicTop5MassMean"]) >= float(acceptance["classicTop5MassMinimum"]),
        "policyMass": float(summary["policyMassMaximumAbsoluteError"]) <= float(acceptance["policyMassMaximumAbsoluteError"]),
        "mirrorPolicy": float(summary["mirrorMaximumAbsoluteError"]) <= float(acceptance["mirrorMaximumAbsoluteError"]),
    }


def summarize_stability(primary: Mapping[int, Mapping[str, object]], high: Mapping[int, Mapping[str, object]]) -> dict[str, object]:
    ordinals = sorted(high)
    agreements, jaccards = [], []
    for ordinal in ordinals:
        low, comparison = primary[ordinal], high[ordinal]
        agreements.append(float(low["actionIndex"] == comparison["actionIndex"]))
        left = {action for action, _score in _ranked_scores(low)[:12]}
        right = {action for action, _score in _ranked_scores(comparison)[:12]}
        jaccards.append(len(left & right) / max(1, len(left | right)))
    return {
        "positions": len(ordinals),
        "actionAgreement": mean(agreements),
        "meanTop12Jaccard": mean(jaccards),
    }


def _teacher_manifest(
    workspace: Path,
    contract_sha: str,
    source_sha: str,
    selection_sha: str,
    primary_refs: Sequence[Mapping[str, object]],
    stability_refs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return _self_hashed({
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f51-classic-root-teacher-manifest",
        "status": "completed",
        "contractSha256": contract_sha,
        "sourceSha256": source_sha,
        "selectionSha256": selection_sha,
        "primary": {"budgetNodes": 4096, "positions": 4608, "chunks": list(primary_refs)},
        "stability": {"budgetNodes": 16384, "positions": 128, "chunks": list(stability_refs)},
    }, "manifestSha256")


def run_phase4f51(workspace_root: str | Path, device_name: str = "cuda:0") -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        return {**verify_phase4f51(workspace), "reused": True}
    contract = _contract(workspace)
    source = _source(workspace)
    resources = _resources(workspace, contract)
    run_id = f"{RUN_ID_PREFIX}-{time.time_ns()}"
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=run_id):
        excluded, exclusion = _build_or_load_exclusion(workspace, str(contract["contractSha256"]), str(source["sha256"]))
        _progress("preflight", "verified", contractSha256=contract["contractSha256"], excludedOrbits=len(excluded))
        ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
        stores, gpu_seconds = _run_rollouts(workspace, contract, torch.device(device_name), ledger, run_id)
        quotas = contract["probeDataset"]["absolutePlyBands"]
        selected, selection_audit = select_probe_positions(stores, excluded, {split: quotas[split] for split in SPLIT_ORDER}, contract["rollout"]["seeds"])
        if len(selected) != 4608 or any(meta["positionOrbit"] in excluded for _state, meta in selected):
            raise Phase4F51Error("F51 selected positions violate the exclusion contract")
        selection = _selection_payload(str(contract["contractSha256"]), str(source["sha256"]), selected, {**selection_audit, "excludedInventorySha256": exclusion["inventorySha256"], "excludedOrbitOverlap": 0})
        if (workspace / SELECTION).exists():
            existing = _mapping(workspace / SELECTION, "F51 selection")
            if existing != selection:
                raise Phase4F51Error("F51 immutable position selection changed")
        else:
            _write_immutable(workspace / SELECTION, selection, "F51 position selection")
        primary_refs = _run_teacher_chunks(workspace, selected, list(range(4608)), "primary-4096", 4096, str(contract["contractSha256"]), str(source["sha256"]), str(selection["selectionSha256"]))
        stability_ordinals = select_stability_ordinals([meta for _state, meta in selected])
        stability_refs = _run_teacher_chunks(workspace, selected, stability_ordinals, "stability-16384", 16384, str(contract["contractSha256"]), str(source["sha256"]), str(selection["selectionSha256"]))
        teacher = _teacher_manifest(workspace, str(contract["contractSha256"]), str(source["sha256"]), str(selection["selectionSha256"]), primary_refs, stability_refs)
        teacher_path = workspace / TEACHER_ROOT / "manifest.json"
        _write_immutable(teacher_path, teacher, "F51 teacher manifest")
        primary = _load_analysis_records(workspace, primary_refs)
        high = _load_analysis_records(workspace, stability_refs)
        stability = summarize_stability(primary, high)
        stability_checks = {
            "actionAgreement": float(stability["actionAgreement"]) >= float(contract["classicTeacher"]["stabilityAudit"]["minimumActionAgreement"]),
            "top12Jaccard": float(stability["meanTop12Jaccard"]) >= float(contract["classicTeacher"]["stabilityAudit"]["minimumTop12Jaccard"]),
        }
        development_records = [primary[int(meta["ordinal"])] for _state, meta in selected if meta["split"] == "development"]
        candidates = []
        selected_temperature = None
        acceptance = contract["policyTargetCalibration"]["acceptance"]
        for temperature in contract["policyTargetCalibration"]["rankTemperatureCandidates"]:
            summary = summarize_targets(development_records, float(temperature))
            checks = target_checks(summary, acceptance)
            candidates.append({**summary, "checks": checks, "passed": all(checks.values())})
            if selected_temperature is None and all(checks.values()):
                selected_temperature = float(temperature)
        target_passed = selected_temperature is not None
        result = _self_hashed({
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f51-tie-preserving-rank-target-probe",
            "status": "completed",
            "contractSha256": contract["contractSha256"],
            "selectionSha256": selection["selectionSha256"],
            "teacherManifestSha256": teacher["manifestSha256"],
            "calibrationSplit": "development",
            "positions": len(development_records),
            "candidates": candidates,
            "selectedRankTemperature": selected_temperature,
            "targetProbePassed": target_passed,
            "valueTarget": None,
        }, "resultSha256")
        _write_immutable(workspace / TARGET_RESULT, result, "F51 target probe result")
        probe_passed = target_passed and all(stability_checks.values())
        report = _self_hashed({
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f51-learner-occupancy-teacher-root-probe",
            "status": "completed",
            "source": source,
            "contractPath": CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "trainingStarted": False,
            "optimizerSteps": 0,
            "largeDatasetGenerationStarted": False,
            "screenTrajectoriesUsedForTrainingOrCalibration": False,
            "rollouts": {split: _store_summary(stores[split]) for split in SPLIT_ORDER},
            "exclusionInventoryPath": EXCLUSION.as_posix(),
            "exclusionInventorySha256": exclusion["inventorySha256"],
            "selectionPath": SELECTION.as_posix(),
            "selectionSha256": selection["selectionSha256"],
            "teacherManifestPath": (TEACHER_ROOT / "manifest.json").as_posix(),
            "teacherManifestSha256": teacher["manifestSha256"],
            "stabilityAudit": {**stability, "checks": stability_checks, "passed": all(stability_checks.values())},
            "targetProbePath": TARGET_RESULT.as_posix(),
            "targetProbeSha256": result["resultSha256"],
            "targetProbePassed": target_passed,
            "probePassed": probe_passed,
            "resources": resources,
            "execution": {"device": device_name, "rolloutActors": ROLLOUT_ACTORS, "classicActors": CLASSIC_ACTORS, "gpuSecondsThisInvocation": gpu_seconds, "globalGpuHours": ledger.total_hours},
            "authorizations": {"largeDatasetGenerationAuthorized": False, "cudaTrainingAuthorized": False, "formalTrainingAuthorized": False, "promotionAuthorized": False},
            "nextStage": {"name": "Phase 4F52 policy-head-only pilot preparation" if probe_passed else "Phase 4F52 F51 target/stability correction", "requiresSeparateUserConfirmation": True},
        }, "reportSha256")
        _write_immutable(workspace / REPORT, report, "F51 probe report")
        gate = _self_hashed({
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f51-learner-teacher-root-probe-gate",
            "status": "completed",
            "probePassed": probe_passed,
            "outcomeBand": "policy-head-pilot-preparation-candidate" if probe_passed else "probe-correction-required",
            "selectedRankTemperature": selected_temperature,
            "classicStabilityPassed": all(stability_checks.values()),
            "targetCalibrationPassed": target_passed,
            "reportPath": REPORT.as_posix(),
            "reportSha256": report["reportSha256"],
            "largeDatasetGenerationAuthorized": False,
            "cudaTrainingAuthorized": False,
            "formalTrainingAuthorized": False,
            "promotionAuthorized": False,
            "nextStage": report["nextStage"],
        }, "gateSha256")
        _write_immutable(workspace / GATE, gate, "F51 probe gate")
        return {**gate, "globalGpuHours": ledger.total_hours, "reused": False}


def verify_phase4f51(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    contract = _contract(workspace)
    report = _mapping(workspace / REPORT, "F51 report")
    gate = _mapping(workspace / GATE, "F51 gate")
    result = _mapping(workspace / TARGET_RESULT, "F51 target result")
    teacher = _mapping(workspace / TEACHER_ROOT / "manifest.json", "F51 teacher manifest")
    report_sha = _verify_self_hash(report, "reportSha256", "F51 report")
    gate_sha = _verify_self_hash(gate, "gateSha256", "F51 gate")
    result_sha = _verify_self_hash(result, "resultSha256", "F51 target result")
    teacher_sha = _verify_self_hash(teacher, "manifestSha256", "F51 teacher manifest")
    if (
        report.get("source") != _source(workspace)
        or report.get("contractSha256") != contract["contractSha256"]
        or report.get("targetProbeSha256") != result_sha
        or report.get("teacherManifestSha256") != teacher_sha
        or gate.get("reportSha256") != report_sha
        or gate.get("largeDatasetGenerationAuthorized") is not False
        or gate.get("cudaTrainingAuthorized") is not False
        or report.get("trainingStarted") is not False
        or report.get("optimizerSteps") != 0
    ):
        raise Phase4F51Error("F51 immutable evidence changed")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f51-learner-teacher-root-probe-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha,
        "reportPath": REPORT.as_posix(),
        "reportSha256": report_sha,
        "probePassed": gate["probePassed"],
        "outcomeBand": gate["outcomeBand"],
        "selectedRankTemperature": gate["selectedRankTemperature"],
        "formalTrainingAuthorized": False,
        "nextStage": gate["nextStage"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    payload = run_phase4f51(args.workspace_root, args.device) if args.command == "run" else verify_phase4f51(args.workspace_root)
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
