# Frozen new-rule arena

M5 provides `arena_v1`, separate from the legacy arena. Its development gate
uses the unchanged `configs/arena.yaml`: 200 openings, two side-swapped games
per opening, 256 PUCT simulations per move, CPU, no root noise, temperature zero,
seed 20261010, and a shared 3600-second budget. The fixed action tie-break is
largest visit count, then smallest absolute action ID. Both models have the
same architecture and search budget. No runtime search or scoring override is
accepted.

```sh
make arena CONFIG=configs/arena.yaml \
  CANDIDATE=path/to/candidate.pt CHAMPION=path/to/champion.pt \
  CANDIDATE_SHA256=exact_candidate_sha256 CHAMPION_SHA256=exact_champion_sha256
make check-m5
make check
```

The two path arguments are mandatory for `make arena`. Digests are optional
external expectations; every load hashes the bytes actually deserialized and
records that identity. `model_checkpoint` rejects legacy and training-resume
formats, nonfinite tensors, and incompatible rules/encoder/value metadata.
Models stay in eval mode and receive no optimizer updates. No `latest` champion
pointer is followed or updated. The output is a gate report, not an automatic
champion-store mutation.

The development opening bytes are checked again at consumption. Every prefix
is replayed from its declared initial player through M1, preserving all prior
position occurrences and counting opening plies toward the 256-ply game limit.
The legacy opening format is only an input container; legacy adjudication never
decides arena outcomes. Both legs start with the same full history, exchanging
model ownership. Goal wins and third-repetition draws are completed games;
runtime/ply/search truncation, interruption, illegal actions, numerical and
infrastructure failures never invent a result.

Runs contain the frozen YAML, code/lock provenance, model identities and
metadata, per-search actions/visits/history hashes, per-game outcomes, JSONL
events, and `arena-result.json`. Search must finish all 256 simulations before
its action can be committed. Incomplete comparisons retain diagnostic W/L/D,
scheduled/completed counts and error reasons, return nonzero, and cannot pass
the gate. The shared deadline includes model loading, opening validation and
result processing. Timing remains cooperative across indivisible operations.

Consumers must require both a completed `summary.json` and a completed,
eligible `arena-result.json` before accepting a promotion decision.
`m5_arena_summary` events are explicitly provisional. An interruption or
deadline overrun during finalization invalidates both promotion flags in the
result and appends `arena_finalization_invalidated`; the earlier event alone
cannot certify promotion.

`arena_statistics` scores win/loss/draw as 1/0/0.5. W/L/D includes every completed
leg, even when its mate is incomplete; such partial scores are diagnostic only.
Confidence resamples whole complete opening pairs 10,000 times using the fixed
seed. The central 95% percentile interval uses 2.5% and 97.5% quantiles with
linear interpolation. Promotion requires all 400 games/200 pairs completed,
no protocol/runtime errors, score >= 0.55 and lower bound strictly > 0.50.
Small schedules or altered bootstrap/threshold settings cannot certify promotion.

`arena_opponents` supplies fixed diagnostic random-legal, move-only shortest-path
and classic adapters, registered by `configs/opponents-v1.yaml` and its digest.
The classic adapter uses 587 nodes/depth 5 and translates the complete history
for the optional Node bridge. Its *internal search* still uses historical
repetition/ply-cap loss heuristics, explicitly recorded in the configuration;
only a new-engine-validated action may be consumed by an M1 referee. It is not
a migrated model champion. Missing Node or a bridge failure raises an error;
there is no fallback. Formal `make arena` remains neural candidate vs neural
champion; these diagnostic APIs do not replace either participant.

The existing dev suite has prior evaluation exposure; old `final.json` is also
exposed and is not an independent holdout. See `reserved-openings-v1.md` for the
fresh 200-opening suite frozen before M6 without model/score selection. Its
generation, exact-history validation and byte reproduction do not evaluate
playing strength. Using it formally requires a separately frozen final protocol,
not a runtime override of `arena-dev-v1`.

The historical direct CLI call `arena --config ...` without model arguments
retains its explicit M0 stub response to preserve old contract tests. Supplying
one/empty model argument returns an input error. `make arena` always supplies
the model flags; it cannot report success through the stub. `make experiment`
remains explicitly unimplemented.
