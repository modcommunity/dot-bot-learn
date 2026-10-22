# dot-bot-learn

**Offline learning for bots.** Turns dot-bot's recorded episodes into exported files (ride routes, nav hints, tuned parameters) that bots load. Design only so far.

Read the family [CLAUDE.md](../../CLAUDE.md) first; this file is only what is specific to this repository. The reasoning is in [docs/design.md](docs/design.md).

**Depends on nothing at run time.** It reads dot-bot's episode files and writes export files; it names no addon class. dot-bot's recorder stays in dot-bot.

## Read before

| Read | Before |
| --- | --- |
| [docs/design.md](docs/design.md) | anything: inputs, outputs, the boundary, the pipeline, hygiene, testing |
| [dot-bot demonstration recorder](../dot-bot/docs/demonstration-recorder.md) | touching an input format |

## Rules that bite

- **No training code in a runtime addon.** Statistics, clustering and ML live here and run outside the game; dot-bot and every game only load the exported files. [design](docs/design.md#the-boundary)
- **An export carries no identifier**: no name, account, peer number, address, team roster, chat or time finer than a day. A tool that writes one is a bug. [design](docs/design.md#data-hygiene)
- **Every export names its schema id and a format version**, and a loader refuses a mismatch. [design](docs/design.md#outputs)
- **A tuned parameter is a suggestion a person reviews**, never rewritten by a server on its own. [design](docs/design.md#outputs)
- **Everything is testable without recordings**: a seeded synthetic generator makes the episodes. [design](docs/design.md#testing-without-data)
- Public repository: name no other game or engine in any file.
- Markdown is never hard-wrapped.

## Validating

Not built yet. When code exists: the tools' own unit tests on the synthetic generator, then a round trip (generate, learn, export, load in dot-bot's suite).
