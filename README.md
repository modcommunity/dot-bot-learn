# dot-bot-learn

**Offline learning for dot-bot.** Reads what dot-bot's demonstration recorder stored (anonymous movement and engagement episodes), and writes plain exported files that bots load: verified ride routes, navigation hints and tuned bot parameters. It runs outside the game, on a developer's machine or a GPU box, and ships no code that a game loads at run time.

This repository is **design only** so far: [docs/design.md](docs/design.md) is the whole content. Nothing here is built.

## What it is not

- Not a runtime addon. A game never depends on it; dot-bot only reads the files it writes.
- Not the recorder. The recorder, its schema and its stores live in dot-bot (`addons/dot_bot/demo/`, [docs](../dot-bot/docs/demonstration-recorder.md)).
- Not a place for player data. Exports carry no identifier of any person.

## Family

Depends on dot-bot's file formats only (the episode JSON and the export manifest), not on its code. Read the family [CLAUDE.md](../../CLAUDE.md) and [CLAUDE.md](CLAUDE.md) here before working in it.
