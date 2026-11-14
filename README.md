# dot-bot-learn

Offline learning for [dot-bot](https://github.com/modcommunity/dot-bot). It will read the anonymous episodes that dot-bot's demonstration recorder stored and write plain files that bots load: verified ride routes, navigation hints and tuned bot parameters. It runs outside the game, on a developer machine or a GPU box. Part of the [dot-*](https://github.com/modcommunity) family.

**Status: design only.** Nothing is built yet. [docs/design.md](docs/design.md) is the whole content, and there is nothing to install.

## What it is and is not

| It is | It is not |
| --- | --- |
| A set of offline tools that turn recordings into export files. | A runtime addon. A game never depends on it; dot-bot only reads the files it writes. |
| Dependent on dot-bot's file formats (episode JSON, export manifest). | The recorder. The recorder, schema and stores live in [dot-bot-record](https://github.com/modcommunity/dot-bot-record) and dot-bot. |
| Free of player data. Exports carry no identifier of any person. | A place to keep player data. |

## Working on it

Read the family `CLAUDE.md` and the [CLAUDE.md](CLAUDE.md) in this repository first.

## Licence

MIT. Only the TMC back-end web infrastructure is not open source.
