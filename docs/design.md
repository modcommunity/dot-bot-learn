# Design

## Why a separate repository

dot-bot-record records (the demonstration recorder) and already harvests rides on the motor. What it does not and should not hold is the learning: clustering, percentile folding, and later a small network. That code has heavy dependencies (numerical libraries, a GPU trainer), runs for minutes to hours, and is useless on a game server or in a browser build. Keeping it here keeps the runtime addons small and means a game's dependency list never grows because someone wants better bots.

Note: the recorder is the separate addon dot-bot-record (`addons/dot_bot_record/`, extracted from dot-bot). This repository is its offline consumer.

## Inputs

All come from dot-bot-record's recorder, through its stores ([demonstration-recorder](../../dot-bot-record/docs/demonstration-recorder.md)).

| Input | What |
| --- | --- |
| Episodes | `map_id`, `map_version`, `schema_id`, `kind`, `key`, `start` (plain values), `rate`, `ticks` (observation then action per kept tick, float32), `outcome`, `day` |
| `verified`, `score` | written by dot-bot's ride harvest after replaying on the motor; only `verified` rides are trusted as routes |
| Adaptive rate | episodes on a sparse key are kept at the tick rate, a well-covered key at the base rate; the learner reads `rate` per episode and must not assume a fixed step |
| Stores | the JSON store's files (`<path>/<map_id>@<version>/<kind>.json`), or a SQL dump of `dot_bot_demos`; the schema's field list (names, sizes, scales) travels in the export manifest |

A different `map_version` or `schema_id` is a different set and is never pooled. Bots never appear as subjects, so no input is bot-generated.

## Outputs

Exported files, versioned and keyed by map version and schema id. Every file has `format` (an integer), `schema_id`, `map_id`, `map_version`, `generated` (UTC date), `source` (counts only) and a `kind`.

Ride routes (`rides.json`), ranked per edge, in the shape dot-bot's ride book already merges:

```json
{
  "format": 1, "kind": "rides", "schema_id": "a1b2c3", "map_id": "surf_example", "map_version": "9f2e",
  "generated": "2026-10-22",
  "edges": {
    "17>9": [
      {"source": "recorded", "success": 0.83, "support": 41, "start": {}, "rate": 16, "commands": "<base64 float32>"}
    ]
  }
}
```

Nav hints (`nav_hints.json`): per-cell or per-waypoint weights and optional off-mesh link suggestions, consumed through `DotBotWorld.waypoints_fn`.

```json
{
  "format": 1, "kind": "nav_hints", "map_id": "surf_example", "map_version": "9f2e", "cell_size": 2.0,
  "places": [{"cell": [12, 3, 40], "weight": 0.07, "count": 63}],
  "links": [{"from": [0.0, 1.0, 2.0], "to": [4.0, 1.0, 2.0], "support": 18}]
}
```

Tuned params (`bot_params.json`): percentile-derived values for rows of `DotBotProfile.SPEC`, per game and map version.

```json
{
  "format": 1, "kind": "bot_params", "game": "example", "map_version": "9f2e",
  "percentiles": {"reaction": {"p20": 0.41, "p50": 0.29, "p80": 0.21}},
  "suggest": {"easy": {"reaction": 0.41}, "medium": {"reaction": 0.29}, "hard": {"reaction": 0.21}}
}
```

Rules: a number outside the spec's range or below a preset floor is clamped by the loader, not trusted; a suggestion is reviewed by a person and committed, never applied by a server by itself; later, network weights follow dot-bot's `.dotmlp` plan, still as plain files.

## The boundary

- **No training code in a runtime addon.** dot-bot and games contain loaders, validators and the forward pass of a tiny network at most. Statistics, clustering, trainers and evaluation harnesses live here.
- **Bots only load exported files.** They never read raw episodes, never call this repository, and never learn during play.
- **The file formats are the interface.** A loader checks `format` and `schema_id` and refuses a mismatch with a `DotResult` error, so the two sides can change independently.
- This repository needs no Godot at run time. Tools that need the motor (verification, evaluation on a simulator) call into a game's headless scene and are the one exception, listed per tool.

## Training pipeline options

In order of attempt; each stage must beat the one before on a measured metric before the next is built.

1. **Plain statistics.** Per map version: place counts on a grid with decay, reaction and aim percentiles, landing rates per ride. No model, fully explainable, runs in seconds.
2. **Route clustering.** Recorded rides on one edge are resampled to a fixed length, clustered (k-medoids or DBSCAN on trajectory distance), and each cluster's medoid is the representative route, with support (cluster size) and a verified landing rate. Outliers are dropped. This replaces "best of N by score" with a few distinct lines, so bots do not all take one.
3. **Small ML.** A small multilayer perceptron cloned from episodes, dataset aggregation where a simulator exists, trained on a GPU box through the family's `gpu.sh`. Gated by held-out evaluation against the fixed baseline, per map. Never a language model in a control loop. dot-bot's recorder doc sketches the dataset shards and `.dotmlp` format; this repository owns the trainer.

All stages read the same export manifest and write the same output files, so a stage can be swapped without a game noticing.

## Data hygiene

- **Consent and opt-out.** Recording is server-side and announced in the server's rules text; the owner turns it off with `bot_record 0`. A player who asks to be excluded is handled by the host's `include_fn` before an episode exists. Because episodes hold no identifier, an after-the-fact deletion of one person's data is impossible by design, which is why exclusion must happen at recording time and the rules text must say so.
- **No identifiers in exports.** No name, account id, peer number, address, team roster, chat, voice, or time finer than a day. Positions are movement, not identity; free-text fields do not exist.
- **Aggregation floors.** A place cell or statistic is exported only past a minimum count, so one person's habit never appears; the floor is a setting in the tool and recorded in the export.
- **Retention.** Raw episodes are pruned by dot-bot-record's store `prune` when a map version changes; exports are regenerated, not edited.
- **A check, not a hope.** A hygiene test scans every export for forbidden keys and for timestamps finer than a day, and fails the tool.

## Testing without data

The repository has no recordings and must still be tested.

- **Synthetic generator.** A seeded generator produces episodes against a toy motor (the same toy simulator style dot-bot's suite uses): known ride lines with noise, so the true clusters, the true percentiles and the true heat map are known.
- **Known-answer tests.** Statistics must recover the generator's parameters within tolerance; clustering must find the planted number of routes; a deliberately bad ride must be rejected by verification.
- **Round trip.** Generate, learn, export, then load the files in dot-bot's suite and ride one. This is the check that the formats really match.
- **Hygiene test.** The scan above, run on every export the tests make, including one generated with identifiers planted in the input to prove they are not copied.
- **Determinism.** Every tool takes a seed; the same input and seed give the same bytes.
- **Real data later.** The first run on a real server's recordings is reviewed by a person against the baseline before any export is committed.

## Open questions

- Decided: verification (the replay on the motor) stays in dot-bot, because bots need it at runtime; this repository only consumes the `verified` flags.
- Language for the trainers: plain Python with a numerical library, run outside Godot, is the working assumption.
- Where exports are published (committed beside a map, or delivered with a dot-cloud pack) is a game decision.
