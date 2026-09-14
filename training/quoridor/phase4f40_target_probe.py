"""Phase 4F40 target-generation conformance probe.

The probe reanalyses a deterministic 4096/512 train/development sample with
the frozen F21 network.  It writes resumable immutable 64-position chunks and
never constructs an optimizer or changes model parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

import torch

from .calibration import GPU_HOUR_BUDGET, GPU_LEDGER_NAME, GpuBudgetLedger
from .codec import mirror_action_index, mirror_state, position_hash
from .config import load_experiment_config
from .evaluator import TorchEvaluator
from .game import (
    State,
    apply_action_index,
    get_winner,
    legal_action_indices,
    legal_pawn_moves,
    shortest_path_distance,
)
from .mcts import MCTSConfig
from .native_mcts import NativeMCTS, load_native_module
from .phase4_budget import PHASE4_CONTROL_RELATIVE
from .phase4f29_fresh_screen import (
    _mapping,
    _self_hashed,
    _verify_self_hash,
    _write_immutable,
)
from .phase4f39_early_curriculum import (
    CONTRACT as F39_CONTRACT,
    EXPECTED_F21_CANDIDATE_SHA256,
    F21_CONFIG,
    F21_RUN,
    GUIDANCE_ROOT,
    legal_action_bucket,
    ply_band,
    split_bucket,
)
from .replay_store import ReplayStore
from .root_cause_ablation import _load_model
from .run_manifest import (
    DEFAULT_MINIMUM_FREE_DISK_BYTES,
    EXTENDED_ARTIFACT_LIMIT_BYTES,
    RunLock,
    canonical_json_bytes,
    fingerprint_sources,
    sha256_bytes,
    sha256_file,
)
from .storage import directory_size


SCHEMA_VERSION = 1
EXPECTED_F39_CONTRACT_SHA256 = (
    "b4e9ce03cb8bc84253a834c05eea049c3518c8c4426709dc81161aca36372dc5"
)
ROOT = Path("training/runs/phase4/phase4f/f40-target-probe-v2")
SELECTION = ROOT / "selection.json"
CHUNKS = ROOT / "chunks"
CALIBRATION = ROOT / "target-calibration.json"
RESULT = ROOT / "probe-result.json"
GATE = ROOT / "probe-gate.json"

RUN_ID = "phase4f40-target-probe-v2"
SELECTION_SEED = 20260825
SEARCH_SEED = 2026082500
TRAIN_POSITIONS = 4096
DEVELOPMENT_POSITIONS = 512
CHUNK_POSITIONS = 64
SIMULATIONS = 800
TOP_K = 12
GPU_HOUR_LIMIT = 4.0
WRITE_RESERVE_BYTES = 512 * 1024**2
MIRROR_AUDIT_POSITIONS = 64
BAND_COUNTS = {
    "train": {"early": 2458, "middle": 819, "late": 819},
    "development": {"early": 308, "middle": 102, "late": 102},
}


class Phase4F40Error(RuntimeError):
    """Raised when the target probe violates its frozen inputs or gates."""


@dataclass(frozen=True, slots=True)
class ProbePosition:
    ordinal: int
    split: str
    game_id: str
    ply_index: int
    band: str
    state: State
    outcome: float
    teacher_action_index: int
    orbit_sha256: str


def _progress(stage: str, status: str, **fields: object) -> None:
    print(
        json.dumps(
            {
                "kind": "phase4f40-target-probe-progress",
                "schemaVersion": SCHEMA_VERSION,
                "stage": stage,
                "status": status,
                **fields,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _workspace(value: str | Path) -> Path:
    workspace = Path(value).resolve()
    if not (workspace / "training/quoridor").is_dir():
        raise Phase4F40Error("workspace does not contain training/quoridor")
    return workspace


def _contract(workspace: Path) -> dict[str, object]:
    payload = _mapping(workspace / F39_CONTRACT, "F39 curriculum contract")
    digest = _verify_self_hash(payload, "contractSha256", "F39 curriculum contract")
    authorizations = payload.get("authorizations")
    probe = payload.get("targetProbe")
    if (
        digest != EXPECTED_F39_CONTRACT_SHA256
        or payload.get("status") != "frozen"
        or not isinstance(authorizations, Mapping)
        or authorizations.get("targetProbeAuthorized") is not True
        or authorizations.get("cudaTrainingAuthorized") is not False
        or not isinstance(probe, Mapping)
        or probe.get("trainPositions") != TRAIN_POSITIONS
        or probe.get("developmentPositions") != DEVELOPMENT_POSITIONS
        or probe.get("finalPositions") != 0
        or probe.get("trainingSteps") != 0
    ):
        raise Phase4F40Error("F39 does not authorize the frozen F40 probe")
    return payload


def _implementation_sha256(workspace: Path) -> str:
    return fingerprint_sources(
        workspace,
        paths=(
            "training/quoridor/phase4f40_target_probe.py",
            "training/quoridor/phase4f39_early_curriculum.py",
            "training/quoridor/native_mcts.py",
            "training/quoridor/mcts.py",
            "training/quoridor/evaluator.py",
            "training/quoridor/game.py",
            "training/quoridor/codec.py",
            "native/src",
            "native/include",
        ),
    ).sha256


def _resources(workspace: Path) -> dict[str, int]:
    artifact_bytes = directory_size(workspace / PHASE4_CONTROL_RELATIVE)
    free_bytes = shutil.disk_usage(workspace).free
    if artifact_bytes + WRITE_RESERVE_BYTES >= EXTENDED_ARTIFACT_LIMIT_BYTES:
        raise Phase4F40Error("16 GiB artifact cap cannot reserve F40 outputs")
    if free_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise Phase4F40Error("less than 20 GiB of free disk remains")
    return {
        "artifactBytes": artifact_bytes,
        "artifactLimitBytes": EXTENDED_ARTIFACT_LIMIT_BYTES,
        "writeReserveBytes": WRITE_RESERVE_BYTES,
        "freeDiskBytes": free_bytes,
        "minimumFreeDiskBytes": DEFAULT_MINIMUM_FREE_DISK_BYTES,
    }


def _open_seed_store(workspace: Path) -> tuple[ReplayStore, dict[str, object]]:
    root = workspace / GUIDANCE_ROOT / "train"
    manifest = _mapping(root / "manifest.json", "guidance seed-train manifest")
    capacity = manifest.get("capacity")
    if not isinstance(capacity, Mapping):
        raise Phase4F40Error("guidance replay capacity is invalid")
    store = ReplayStore(
        root,
        max_games=int(capacity["games"]),
        max_plies=int(capacity["plies"]),
        max_bytes=int(capacity["bytes"]),
    )
    return store, manifest


def _stable_key(split: str, band: str, game_id: str, ply_index: int) -> str:
    return sha256_bytes(
        f"{SELECTION_SEED}:{split}:{band}:{game_id}:{ply_index}".encode("ascii")
    )


def select_references(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    """Select exact band counts without decoding replay shards."""

    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise Phase4F40Error("guidance manifest shards are invalid")
    candidates: dict[tuple[str, str], list[tuple[str, str, int]]] = defaultdict(list)
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise Phase4F40Error("guidance shard entry is invalid")
        game_id = shard.get("gameId")
        plies = shard.get("plies")
        if not isinstance(game_id, str) or type(plies) is not int or plies <= 0:
            raise Phase4F40Error("guidance shard identity is invalid")
        split = split_bucket(game_id, SELECTION_SEED)
        if split not in BAND_COUNTS:
            continue
        for ply_index in range(plies):
            band = ply_band(ply_index)
            candidates[(split, band)].append(
                (_stable_key(split, band, game_id, ply_index), game_id, ply_index)
            )
    selected: list[dict[str, object]] = []
    ordinal = 0
    for split in ("train", "development"):
        for band in ("early", "middle", "late"):
            count = BAND_COUNTS[split][band]
            ordered = sorted(candidates[(split, band)])
            if len(ordered) < count:
                raise Phase4F40Error(f"not enough {split}/{band} probe positions")
            for selection_key, game_id, ply_index in ordered[:count]:
                selected.append(
                    {
                        "ordinal": ordinal,
                        "split": split,
                        "band": band,
                        "gameId": game_id,
                        "plyIndex": ply_index,
                        "selectionKey": selection_key,
                    }
                )
                ordinal += 1
    return selected


def _materialize_selection(
    store: ReplayStore, references: Sequence[Mapping[str, object]]
) -> tuple[list[ProbePosition], list[dict[str, object]]]:
    by_game: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for reference in references:
        by_game[str(reference["gameId"])].append(reference)
    positions: dict[int, ProbePosition] = {}
    rows: dict[int, dict[str, object]] = {}
    seen_orbits: set[str] = set()
    for game_number, (game_id, game_references) in enumerate(
        sorted(by_game.items()), start=1
    ):
        game, states = store.load_game_with_states(game_id)
        winner = get_winner(states[-1])
        if winner not in (0, 1):
            raise Phase4F40Error(f"probe source game has no winner: {game_id}")
        for reference in game_references:
            ply_index = int(reference["plyIndex"])
            ply = game.plies[ply_index]
            state = states[ply_index]
            orbit = ply.metadata.get("teacherOrbitSha256")
            teacher = ply.metadata.get("teacherActionIndex")
            if not isinstance(orbit, str) or len(orbit) != 64:
                raise Phase4F40Error("probe position has no valid orbit provenance")
            if orbit in seen_orbits:
                raise Phase4F40Error("probe selection contains a duplicate orbit")
            if type(teacher) is not int or teacher not in legal_action_indices(state):
                raise Phase4F40Error("probe teacher action is invalid")
            seen_orbits.add(orbit)
            ordinal = int(reference["ordinal"])
            outcome = 1.0 if state.current_player == winner else -1.0
            position = ProbePosition(
                ordinal=ordinal,
                split=str(reference["split"]),
                game_id=game_id,
                ply_index=ply_index,
                band=str(reference["band"]),
                state=state,
                outcome=outcome,
                teacher_action_index=teacher,
                orbit_sha256=orbit,
            )
            positions[ordinal] = position
            rows[ordinal] = {
                **dict(reference),
                "currentPlayer": state.current_player,
                "outcome": int(outcome),
                "teacherActionIndex": teacher,
                "teacherOrbitSha256": orbit,
                "positionSha256": sha256_bytes(position_hash(state).encode("ascii")),
            }
        if game_number % 100 == 0 or game_number == len(by_game):
            _progress(
                "selection",
                "materializing",
                games=game_number,
                gamesTotal=len(by_game),
                positions=len(positions),
                positionsTotal=len(references),
            )
    return (
        [positions[index] for index in range(len(references))],
        [rows[index] for index in range(len(references))],
    )


def prepare_selection(
    workspace: Path,
    contract_sha256: str,
    implementation_sha256: str,
) -> tuple[dict[str, object], list[ProbePosition]]:
    store, manifest = _open_seed_store(workspace)
    references = select_references(manifest)
    positions, rows = _materialize_selection(store, references)
    counts = Counter((position.split, position.band) for position in positions)
    payload = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f40-target-probe-selection",
            "status": "completed",
            "contractSha256": contract_sha256,
            "implementationSha256": implementation_sha256,
            "algorithm": "whole-game-split-then-sha256-position-rank-v1",
            "selectionSeed": SELECTION_SEED,
            "sourceManifestSha256": sha256_file(
                workspace / GUIDANCE_ROOT / "train" / "manifest.json"
            ),
            "counts": {
                split: {
                    band: counts[(split, band)]
                    for band in ("early", "middle", "late")
                }
                for split in ("train", "development")
            },
            "positions": rows,
        },
        "selectionSha256",
    )
    _write_immutable(workspace / SELECTION, payload, "F40 selection")
    return payload, positions


def _load_selection_positions(
    workspace: Path,
    contract_sha256: str,
    implementation_sha256: str,
) -> tuple[dict[str, object], list[ProbePosition]]:
    if not (workspace / SELECTION).exists():
        return prepare_selection(workspace, contract_sha256, implementation_sha256)
    selection = _mapping(workspace / SELECTION, "F40 selection")
    _verify_self_hash(selection, "selectionSha256", "F40 selection")
    rows = selection.get("positions")
    if (
        selection.get("contractSha256") != contract_sha256
        or selection.get("implementationSha256") != implementation_sha256
        or not isinstance(rows, list)
        or len(rows) != TRAIN_POSITIONS + DEVELOPMENT_POSITIONS
    ):
        raise Phase4F40Error("existing F40 selection differs")
    store, _manifest = _open_seed_store(workspace)
    positions, materialized = _materialize_selection(store, rows)
    for expected, actual in zip(rows, materialized):
        for field in (
            "ordinal",
            "split",
            "band",
            "gameId",
            "plyIndex",
            "teacherActionIndex",
            "teacherOrbitSha256",
            "positionSha256",
        ):
            if expected.get(field) != actual.get(field):
                raise Phase4F40Error("F40 selection source changed")
    return selection, positions


def _classic_static_score(state: State, perspective: int, ply: int = 1) -> float:
    winner = get_winner(state)
    if winner is not None:
        return float(100_000 - ply if winner == perspective else -100_000 + ply)
    opponent = 1 - perspective
    own_distance = shortest_path_distance(state, perspective)
    opponent_distance = shortest_path_distance(state, opponent)
    wall_balance = state.walls_remaining[perspective] - state.walls_remaining[opponent]
    mobility = len(legal_pawn_moves(state, perspective)) - len(
        legal_pawn_moves(state, opponent)
    )
    return 120.0 * (opponent_distance - own_distance) + 7.0 * wall_balance + 3.0 * mobility


def _path_tempo_target(state: State) -> float:
    player = state.current_player
    opponent = 1 - player
    raw = (
        shortest_path_distance(state, opponent)
        - shortest_path_distance(state, player)
        + 0.35 * (state.walls_remaining[player] - state.walls_remaining[opponent])
    )
    return max(-0.75, min(0.75, math.tanh(raw / 4.0)))


def _canonical_mirror(state: State) -> tuple[State, bool]:
    mirrored = mirror_state(state)
    if position_hash(mirrored) < position_hash(state):
        return mirrored, True
    return state, False


def classic_candidates(state: State, teacher_action: int) -> list[dict[str, object]]:
    search_state, was_mirrored = _canonical_mirror(state)
    search_teacher = mirror_action_index(teacher_action) if was_mirrored else teacher_action
    player = search_state.current_player
    scored = [
        (action, _classic_static_score(apply_action_index(search_state, action), player))
        for action in legal_action_indices(search_state)
    ]
    ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
    if position_hash(mirror_state(search_state)) == position_hash(search_state):
        score_by_action = dict(scored)
        groups = sorted(
            {
                tuple(sorted({action, mirror_action_index(action)}))
                for action, _score in scored
            },
            key=lambda group: (-max(score_by_action[action] for action in group), group),
        )
        teacher_group = tuple(
            sorted({search_teacher, mirror_action_index(search_teacher)})
        )
        ordered_groups = [teacher_group] + [
            group for group in groups if group != teacher_group
        ]
        selected_actions: list[int] = []
        for group in ordered_groups:
            additions = [action for action in group if action not in selected_actions]
            if len(selected_actions) + len(additions) <= TOP_K:
                selected_actions.extend(additions)
            if len(selected_actions) == TOP_K:
                break
        if len(selected_actions) < TOP_K - 1:
            raise Phase4F40Error("cannot form a sufficiently large mirror-closed top-K set")
        chosen = sorted(
            ((action, score_by_action[action]) for action in selected_actions),
            key=lambda item: (-item[1], item[0]),
        )
    else:
        chosen = ranked[:TOP_K]
        if search_teacher not in {action for action, _score in chosen}:
            teacher_score = next(score for action, score in ranked if action == search_teacher)
            chosen[-1] = (search_teacher, teacher_score)
            chosen.sort(key=lambda item: (-item[1], item[0]))
    return [
        {
            "actionIndex": mirror_action_index(action) if was_mirrored else action,
            "score": score,
        }
        for action, score in chosen
    ]


def _sparse_visits(visits: Sequence[int], was_mirrored: bool) -> list[list[int]]:
    rows = [
        [mirror_action_index(index) if was_mirrored else index, int(count)]
        for index, count in enumerate(visits)
        if count > 0
    ]
    return sorted(rows)


def _run_mcts(
    search: NativeMCTS,
    positions: Sequence[ProbePosition],
) -> list[dict[str, object]]:
    representatives: list[State] = []
    transforms: list[bool] = []
    for position in positions:
        representative, was_mirrored = _canonical_mirror(position.state)
        representatives.append(representative)
        transforms.append(was_mirrored)
    results = search.run_batch(
        representatives,
        temperatures=[1.0] * len(representatives),
        add_root_noise=False,
    )
    return [
        {
            "visitCounts": _sparse_visits(result.visit_counts, was_mirrored),
            "rootValue": float(result.root_value),
            "searchStateWasMirrored": was_mirrored,
        }
        for result, was_mirrored in zip(results, transforms)
    ]


def _chunk_path(index: int) -> Path:
    return CHUNKS / f"chunk-{index:04d}.json"


def _load_chunk(
    workspace: Path,
    index: int,
    selection_sha256: str,
    contract_sha256: str,
    implementation_sha256: str,
) -> dict[str, object] | None:
    path = workspace / _chunk_path(index)
    if not path.exists():
        return None
    payload = _mapping(path, f"F40 chunk {index}")
    _verify_self_hash(payload, "chunkSha256", f"F40 chunk {index}")
    rows = payload.get("rows")
    if (
        payload.get("chunkIndex") != index
        or payload.get("selectionSha256") != selection_sha256
        or payload.get("contractSha256") != contract_sha256
        or payload.get("implementationSha256") != implementation_sha256
        or not isinstance(rows, list)
        or len(rows) != CHUNK_POSITIONS
    ):
        raise Phase4F40Error(f"existing F40 chunk {index} differs")
    return payload


def _build_search(
    evaluator: TorchEvaluator,
    native_module: object,
    seed: int,
    c_base: float,
    c_init: float,
) -> NativeMCTS:
    return NativeMCTS(
        evaluator,
        MCTSConfig(
            simulations=SIMULATIONS,
            c_base=c_base,
            c_init=c_init,
            dirichlet_epsilon=0.0,
            temperature=1.0,
            add_root_noise=False,
            seed=seed,
        ),
        module=native_module,
    )


def _probe_row(position: ProbePosition, mcts: Mapping[str, object]) -> dict[str, object]:
    legal_count = len(legal_action_indices(position.state))
    return {
        "ordinal": position.ordinal,
        "split": position.split,
        "gameId": position.game_id,
        "plyIndex": position.ply_index,
        "band": position.band,
        "currentPlayer": position.state.current_player,
        "outcome": position.outcome,
        "teacherActionIndex": position.teacher_action_index,
        "teacherOrbitSha256": position.orbit_sha256,
        "legalActionCount": legal_count,
        "legalActionBucket": legal_action_bucket(legal_count),
        "classicCandidates": classic_candidates(
            position.state, position.teacher_action_index
        ),
        "mcts": dict(mcts),
        "pathTempo": _path_tempo_target(position.state),
    }


def _mirror_position(position: ProbePosition) -> ProbePosition:
    return replace(
        position,
        state=mirror_state(position.state),
        teacher_action_index=mirror_action_index(position.teacher_action_index),
    )


def _generate_chunks(
    *,
    workspace: Path,
    positions: Sequence[ProbePosition],
    selection_sha256: str,
    contract_sha256: str,
    implementation_sha256: str,
    device: torch.device,
    ledger: GpuBudgetLedger,
) -> tuple[list[dict[str, object]], float]:
    chunk_total = len(positions) // CHUNK_POSITIONS
    chunks: list[dict[str, object]] = []
    missing = [
        index
        for index in range(chunk_total)
        if _load_chunk(
            workspace,
            index,
            selection_sha256,
            contract_sha256,
            implementation_sha256,
        )
        is None
    ]
    if not missing:
        return [
            _load_chunk(
                workspace,
                index,
                selection_sha256,
                contract_sha256,
                implementation_sha256,
            )
            for index in range(chunk_total)
        ], ledger.total_hours  # type: ignore[list-item]

    config = load_experiment_config(workspace / F21_CONFIG)
    model_path = (
        workspace
        / F21_RUN
        / "checkpoints/candidate-model-checkpoint-g00000001-s000000000192.pt"
    )
    if sha256_file(model_path) != EXPECTED_F21_CANDIDATE_SHA256:
        raise Phase4F40Error("F21 candidate model changed")
    model = _load_model(model_path, workspace / F21_CONFIG, device)
    evaluator = TorchEvaluator(model, device, use_amp=True)
    native_module = load_native_module()
    started = time.monotonic()
    previous_seconds = 0.0
    if RUN_ID in ledger.runs:
        previous_seconds = float(ledger.runs[RUN_ID]["accountedSeconds"])
        ledger.update(RUN_ID, elapsed_seconds=previous_seconds, status="running")
    else:
        ledger.begin(RUN_ID, external_hours=0.0)
    baseline = ledger.total_hours - previous_seconds / 3600.0
    if baseline + GPU_HOUR_LIMIT > GPU_HOUR_BUDGET:
        raise Phase4F40Error("global 120 GPU-hour budget cannot fit F40")

    try:
        for index in range(chunk_total):
            existing = _load_chunk(
                workspace,
                index,
                selection_sha256,
                contract_sha256,
                implementation_sha256,
            )
            if existing is not None:
                chunks.append(existing)
                continue
            batch = positions[
                index * CHUNK_POSITIONS : (index + 1) * CHUNK_POSITIONS
            ]
            search = _build_search(
                evaluator,
                native_module,
                SEARCH_SEED + index,
                config.mcts.c_base,
                config.mcts.c_init,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            chunk_started = time.monotonic()
            mcts_rows = _run_mcts(search, batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            rows = [
                _probe_row(position, mcts)
                for position, mcts in zip(batch, mcts_rows)
            ]
            if index == TRAIN_POSITIONS // CHUNK_POSITIONS:
                mirror_batch = [_mirror_position(position) for position in batch]
                mirror_search = _build_search(
                    evaluator,
                    native_module,
                    SEARCH_SEED + index,
                    config.mcts.c_base,
                    config.mcts.c_init,
                )
                mirror_mcts = _run_mcts(mirror_search, mirror_batch)
                for row, mirror_position, mirror_result in zip(
                    rows, mirror_batch, mirror_mcts
                ):
                    row["mirrorAudit"] = _probe_row(
                        mirror_position, mirror_result
                    )
            payload = _self_hashed(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "phase4f40-target-probe-chunk",
                    "status": "completed",
                    "chunkIndex": index,
                    "positions": len(rows),
                    "simulations": SIMULATIONS,
                    "searchSeed": SEARCH_SEED + index,
                    "selectionSha256": selection_sha256,
                    "contractSha256": contract_sha256,
                    "implementationSha256": implementation_sha256,
                    "rows": rows,
                },
                "chunkSha256",
            )
            _write_immutable(workspace / _chunk_path(index), payload, f"F40 chunk {index}")
            chunks.append(payload)
            elapsed = previous_seconds + time.monotonic() - started
            total = ledger.update(RUN_ID, elapsed_seconds=elapsed, status="running")
            if total >= baseline + GPU_HOUR_LIMIT:
                raise Phase4F40Error("F40 GPU budget was exhausted")
            _progress(
                "reanalysis",
                "progress",
                chunks=index + 1,
                chunksTotal=chunk_total,
                positions=(index + 1) * CHUNK_POSITIONS,
                positionsTotal=len(positions),
                chunkSeconds=time.monotonic() - chunk_started,
                globalGpuHours=total,
            )
        total = ledger.update(
            RUN_ID,
            elapsed_seconds=previous_seconds + time.monotonic() - started,
            status="completed",
        )
        return chunks, total
    except BaseException:
        try:
            ledger.update(
                RUN_ID,
                elapsed_seconds=previous_seconds + time.monotonic() - started,
                status="interrupted",
            )
        except Exception:
            pass
        raise


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    location = (len(ordered) - 1) * quantile
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _softmax_scores(candidates: Sequence[Mapping[str, object]], temperature: float) -> dict[int, float]:
    scores = [float(item["score"]) for item in candidates]
    maximum = max(scores)
    weights = [math.exp(max(-80.0, min(0.0, (score - maximum) / temperature))) for score in scores]
    total = sum(weights)
    return {int(item["actionIndex"]): weight / total for item, weight in zip(candidates, weights)}


def _entropy(policy: Mapping[int, float]) -> float:
    return -sum(value * math.log(value) for value in policy.values() if value > 0)


def fit_classic_temperature(rows: Sequence[Mapping[str, object]]) -> tuple[float, dict[str, float]]:
    train = [row for row in rows if row.get("split") == "train"]
    temperature = 1.0
    while temperature < 1_000_000.0:
        policies = [_softmax_scores(row["classicCandidates"], temperature) for row in train]
        entropies = [_entropy(policy) for policy in policies]
        top1 = [max(policy.values()) for policy in policies]
        if mean(entropies) >= 0.35 and _percentile(top1, 0.95) <= 0.85:
            return temperature, {
                "meanEntropy": mean(entropies),
                "top1ProbabilityP95": _percentile(top1, 0.95),
            }
        temperature *= 1.08
    raise Phase4F40Error("cannot fit a valid classic policy temperature")


def fit_isotonic(xs: Sequence[float], ys: Sequence[float]) -> list[dict[str, float]]:
    if len(xs) != len(ys) or not xs:
        raise ValueError("isotonic inputs must be non-empty and equally sized")
    grouped: dict[float, list[float]] = defaultdict(list)
    for x, y in zip(xs, ys):
        grouped[float(x)].append(float(y))
    ordered = [
        (x, mean(values), float(len(values)))
        for x, values in sorted(grouped.items())
    ]
    blocks: list[dict[str, float]] = []
    for x, y, observation_weight in ordered:
        blocks.append(
            {
                "minimum": x,
                "maximum": x,
                "weight": observation_weight,
                "value": y,
            }
        )
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left["weight"] + right["weight"]
            blocks.append(
                {
                    "minimum": left["minimum"],
                    "maximum": right["maximum"],
                    "weight": weight,
                    "value": (
                        left["value"] * left["weight"]
                        + right["value"] * right["weight"]
                    )
                    / weight,
                }
            )
    return blocks


def isotonic_predict(blocks: Sequence[Mapping[str, float]], value: float) -> float:
    for block in blocks:
        if value <= float(block["maximum"]):
            return float(block["value"])
    return float(blocks[-1]["value"])


def _mcts_policy(row: Mapping[str, object]) -> dict[int, float]:
    visits = row["mcts"]["visitCounts"]
    total = sum(int(item[1]) for item in visits)
    if total <= 0:
        raise Phase4F40Error("MCTS target has no visits")
    return {int(item[0]): int(item[1]) / total for item in visits}


def target_for_row(
    row: Mapping[str, object],
    classic_temperature: float,
    isotonic: Sequence[Mapping[str, float]],
) -> tuple[dict[int, float], float, dict[str, float]]:
    classic = _softmax_scores(row["classicCandidates"], classic_temperature)
    mcts = _mcts_policy(row)
    actions = set(classic).union(mcts)
    policy = {
        action: 0.5 * classic.get(action, 0.0) + 0.5 * mcts.get(action, 0.0)
        for action in actions
    }
    total = sum(policy.values())
    policy = {action: value / total for action, value in policy.items()}
    outcome = float(row["outcome"])
    search = isotonic_predict(isotonic, float(row["mcts"]["rootValue"]))
    path = float(row["pathTempo"])
    value = max(-1.0, min(1.0, 0.60 * outcome + 0.25 * search + 0.15 * path))
    return policy, value, {
        "terminalOutcome": outcome,
        "calibratedSearch": search,
        "pathTempo": path,
    }


def _mirror_policy_back(policy: Mapping[int, float]) -> dict[int, float]:
    return {mirror_action_index(action): value for action, value in policy.items()}


def _summarize_targets(
    rows: Sequence[Mapping[str, object]],
    classic_temperature: float,
    isotonic: Sequence[Mapping[str, float]],
) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for split in ("train", "development"):
        split_rows = [row for row in rows if row.get("split") == split]
        policies: list[dict[int, float]] = []
        values: list[float] = []
        components: list[dict[str, float]] = []
        teacher_ranks: list[int] = []
        for row in split_rows:
            policy, value, component = target_for_row(row, classic_temperature, isotonic)
            policies.append(policy)
            values.append(value)
            components.append(component)
            ranked = sorted(policy, key=lambda action: (-policy[action], action))
            teacher_ranks.append(ranked.index(int(row["teacherActionIndex"])) + 1)
        summaries[split] = {
            "positions": len(split_rows),
            "policy": {
                "meanEntropy": mean(_entropy(policy) for policy in policies),
                "top1ProbabilityP95": _percentile(
                    [max(policy.values()) for policy in policies], 0.95
                ),
                "classicActionTop1Coverage": mean(rank == 1 for rank in teacher_ranks),
                "classicActionTop5Coverage": mean(rank <= 5 for rank in teacher_ranks),
                "meanSupport": mean(len(policy) for policy in policies),
            },
            "value": {
                "mean": mean(values),
                "meanAbsolute": mean(abs(value) for value in values),
                "positiveRate": mean(value > 0 for value in values),
                "minimum": min(values),
                "maximum": max(values),
                "components": {
                    name: {
                        "mean": mean(component[name] for component in components),
                        "meanAbsolute": mean(abs(component[name]) for component in components),
                    }
                    for name in ("terminalOutcome", "calibratedSearch", "pathTempo")
                },
            },
            "byPlyBand": {
                band: sum(row.get("band") == band for row in split_rows)
                for band in ("early", "middle", "late")
            },
        }
    return summaries


def _mirror_audit(
    rows: Sequence[Mapping[str, object]],
    classic_temperature: float,
    isotonic: Sequence[Mapping[str, float]],
) -> dict[str, object]:
    audited = [row for row in rows if "mirrorAudit" in row]
    if len(audited) != MIRROR_AUDIT_POSITIONS:
        raise Phase4F40Error("mirror audit is incomplete")
    policy_errors: list[float] = []
    value_errors: list[float] = []
    for row in audited:
        original_policy, original_value, _ = target_for_row(
            row, classic_temperature, isotonic
        )
        mirror_row = row["mirrorAudit"]
        mirror_policy, mirror_value, _ = target_for_row(
            mirror_row, classic_temperature, isotonic
        )
        restored = _mirror_policy_back(mirror_policy)
        actions = set(original_policy).union(restored)
        policy_errors.append(
            max(abs(original_policy.get(action, 0.0) - restored.get(action, 0.0)) for action in actions)
        )
        value_errors.append(abs(original_value - mirror_value))
    return {
        "positions": len(audited),
        "policyMaximumAbsoluteError": max(policy_errors),
        "valueMaximumAbsoluteError": max(value_errors),
    }


def _all_rows(chunks: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for chunk in chunks:
        rows.extend(dict(row) for row in chunk["rows"])
    if len(rows) != TRAIN_POSITIONS + DEVELOPMENT_POSITIONS:
        raise Phase4F40Error("F40 chunks do not contain the full probe")
    if [int(row["ordinal"]) for row in rows] != list(range(len(rows))):
        raise Phase4F40Error("F40 chunk ordinals are not contiguous")
    return rows


def _finalize(
    workspace: Path,
    contract: Mapping[str, object],
    selection: Mapping[str, object],
    chunks: Sequence[Mapping[str, object]],
    implementation_sha256: str,
    global_gpu_hours: float,
) -> dict[str, object]:
    rows = _all_rows(chunks)
    temperature, temperature_fit = fit_classic_temperature(rows)
    train_rows = [row for row in rows if row.get("split") == "train"]
    isotonic = fit_isotonic(
        [float(row["mcts"]["rootValue"]) for row in train_rows],
        [float(row["outcome"]) for row in train_rows],
    )
    calibration = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f40-target-calibration",
            "status": "completed",
            "contractSha256": contract["contractSha256"],
            "selectionSha256": selection["selectionSha256"],
            "implementationSha256": implementation_sha256,
            "fitSplit": "train",
            "classicTemperature": temperature,
            "classicTemperatureFit": temperature_fit,
            "isotonicBlocks": isotonic,
        },
        "calibrationSha256",
    )
    _write_immutable(workspace / CALIBRATION, calibration, "F40 target calibration")
    summaries = _summarize_targets(rows, temperature, isotonic)
    mirror = _mirror_audit(rows, temperature, isotonic)
    chunk_refs = [
        {
            "path": _chunk_path(int(chunk["chunkIndex"])).as_posix(),
            "sha256": chunk["chunkSha256"],
            "positions": chunk["positions"],
        }
        for chunk in chunks
    ]
    result = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f40-target-generation-conformance-probe",
            "status": "completed",
            "trainingStarted": False,
            "optimizerSteps": 0,
            "contractPath": F39_CONTRACT.as_posix(),
            "contractSha256": contract["contractSha256"],
            "selectionPath": SELECTION.as_posix(),
            "selectionSha256": selection["selectionSha256"],
            "implementationSha256": implementation_sha256,
            "calibrationPath": CALIBRATION.as_posix(),
            "calibrationSha256": calibration["calibrationSha256"],
            "modelSha256": EXPECTED_F21_CANDIDATE_SHA256,
            "simulations": SIMULATIONS,
            "chunks": chunk_refs,
            "summaries": summaries,
            "mirrorAudit": mirror,
            "globalGpuHours": global_gpu_hours,
        },
        "resultSha256",
    )
    _write_immutable(workspace / RESULT, result, "F40 probe result")
    acceptance = contract["targetProbe"]["acceptance"]
    development = summaries["development"]
    checks = {
        "policyTargetMeanEntropy": development["policy"]["meanEntropy"]
        >= acceptance["policyTargetMeanEntropyMinimum"],
        "policyTargetTop1ProbabilityP95": development["policy"]["top1ProbabilityP95"]
        <= acceptance["policyTargetTop1ProbabilityP95Maximum"],
        "classicActionTop5Coverage": development["policy"]["classicActionTop5Coverage"]
        >= acceptance["classicActionTop5CoverageMinimum"],
        "valueTargetMeanAbsolute": development["value"]["meanAbsolute"]
        <= acceptance["valueTargetMeanAbsoluteMaximum"],
        "valueTargetPositiveRateMinimum": development["value"]["positiveRate"]
        >= acceptance["valueTargetPositiveRateMinimum"],
        "valueTargetPositiveRateMaximum": development["value"]["positiveRate"]
        <= acceptance["valueTargetPositiveRateMaximum"],
        "mirrorPolicy": mirror["policyMaximumAbsoluteError"]
        <= acceptance["mirrorPolicyMaximumAbsoluteError"],
        "mirrorValue": mirror["valueMaximumAbsoluteError"]
        <= acceptance["mirrorValueMaximumAbsoluteError"],
    }
    passed = all(checks.values())
    gate = _self_hashed(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "phase4f40-target-probe-gate",
            "status": "completed",
            "outcomeBand": "target-probe-passed" if passed else "target-probe-failed",
            "contractSha256": contract["contractSha256"],
            "resultPath": RESULT.as_posix(),
            "resultSha256": result["resultSha256"],
            "checks": checks,
            "developmentSummary": development,
            "mirrorAudit": mirror,
            "targetProbePassed": passed,
            "datasetGenerationAuthorized": passed,
            "cudaTrainingAuthorized": False,
            "formalTrainingAuthorized": False,
            "nextStage": {
                "name": (
                    "Phase 4F41 early-curriculum dataset generation"
                    if passed
                    else "Phase 4F41 target protocol correction"
                ),
                "requiresSeparateUserConfirmation": True,
            },
        },
        "gateSha256",
    )
    _write_immutable(workspace / GATE, gate, "F40 probe gate")
    return gate


def run_phase4f40(workspace_root: str | Path, device_name: str = "cuda") -> dict[str, object]:
    workspace = _workspace(workspace_root)
    if (workspace / GATE).exists():
        gate = _mapping(workspace / GATE, "F40 probe gate")
        _verify_self_hash(gate, "gateSha256", "F40 probe gate")
        return {**gate, "reused": True}
    contract = _contract(workspace)
    implementation_sha256 = _implementation_sha256(workspace)
    _resources(workspace)
    _progress("preflight", "verified", contractSha256=contract["contractSha256"])
    selection, positions = _load_selection_positions(
        workspace,
        str(contract["contractSha256"]),
        implementation_sha256,
    )
    _progress(
        "selection",
        "completed",
        positions=len(positions),
        selectionSha256=selection["selectionSha256"],
    )
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise Phase4F40Error("F40 reanalysis requires CUDA")
    ledger = GpuBudgetLedger(workspace / PHASE4_CONTROL_RELATIVE / GPU_LEDGER_NAME)
    with RunLock(workspace / PHASE4_CONTROL_RELATIVE / ".phase4.lock", run_id=RUN_ID):
        chunks, global_hours = _generate_chunks(
            workspace=workspace,
            positions=positions,
            selection_sha256=str(selection["selectionSha256"]),
            contract_sha256=str(contract["contractSha256"]),
            implementation_sha256=implementation_sha256,
            device=device,
            ledger=ledger,
        )
        gate = _finalize(
            workspace,
            contract,
            selection,
            chunks,
            implementation_sha256,
            global_hours,
        )
    return {**gate, "globalGpuHours": global_hours, "reused": False}


def verify_phase4f40(workspace_root: str | Path) -> dict[str, object]:
    workspace = _workspace(workspace_root)
    gate = _mapping(workspace / GATE, "F40 probe gate")
    gate_sha256 = _verify_self_hash(gate, "gateSha256", "F40 probe gate")
    result = _mapping(workspace / RESULT, "F40 probe result")
    result_sha256 = _verify_self_hash(result, "resultSha256", "F40 probe result")
    if (
        gate.get("status") != "completed"
        or gate.get("resultSha256") != result_sha256
        or gate.get("cudaTrainingAuthorized") is not False
        or gate.get("formalTrainingAuthorized") is not False
        or result.get("trainingStarted") is not False
        or result.get("optimizerSteps") != 0
    ):
        raise Phase4F40Error("F40 artifacts are inconsistent")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "phase4f40-target-probe-verification",
        "status": "verified",
        "gatePath": GATE.as_posix(),
        "gateSha256": gate_sha256,
        "resultPath": RESULT.as_posix(),
        "resultSha256": result_sha256,
        "outcomeBand": gate["outcomeBand"],
        "datasetGenerationAuthorized": gate["datasetGenerationAuthorized"],
        "formalTrainingAuthorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"), nargs="?", default="run")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = (
        run_phase4f40(args.workspace_root, args.device)
        if args.command == "run"
        else verify_phase4f40(args.workspace_root)
    )
    print(canonical_json_bytes(payload).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
