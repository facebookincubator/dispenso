# Dispenso development docs

Internal documentation: how the library works, what is planned, and what is
still an open question. None of it is part of the published API documentation.

User-facing docs live one level up in `docs/` — `getting_started.md`,
`building.md`, `faq.md`, and the migration guides. Those are linked from the
README and published by Doxygen, so their paths are effectively public API;
prefer adding to them over moving them.

## What lives where

| Directory | Contents | Tense |
|-----------|----------|-------|
| [`architecture/`](architecture/) | How a subsystem works **today** | Present |
| [`proposals/`](proposals/) | Designs for things **not built yet** | Conditional |
| [`roadmap/`](roadmap/) | What we intend to do, by area | Future |
| [`investigations/`](investigations/) | Measurements and open questions | Past |
| [`process/`](process/) | How we release | Imperative |

[`roadmap.md`](roadmap.md) is the index across the roadmap areas.

## Where does a new document go?

Ask what tense it is written in.

- Describing code that exists → `architecture/`. If it describes behaviour that
  changed, update the existing doc rather than adding a second one.
- Describing code that does not exist → `proposals/`. Include a "current state"
  section, and keep it honest: a proposal measured against a stale baseline
  overstates its own benefit.
- A measurement, a comparison, or a question you answered → `investigations/`.
  These are dated notes, not commitments; a result here may or may not become a
  roadmap item.
- Something we have decided to do → the relevant `roadmap/` area, not a new
  file.

## Conventions

- Each item belongs to exactly one place. If status appears in two tables, they
  will drift.
- Comments and docs describe the current state and its motivation, not the
  change that produced it — that belongs in the commit message.
- [`../../CHANGELOG.md`](../../CHANGELOG.md) is the record of shipped work. The
  roadmap's Completed table exists only so items carried there do not dangle.
