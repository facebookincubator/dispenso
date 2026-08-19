---
description: "Which dispenso BUCK target to depend on from a library versus an executable"
metadata:
  oncalls: ["dispenso_oncall"]
  apply_to_path: 'arvr/libraries/dispenso/.*'
---

# Depending on dispenso in BUCK

dispenso is split so that the choice of implementation is made once, by the
final binary, instead of by every library along the way.

| Target | Depend on it from |
|---|---|
| `//arvr/libraries/dispenso:dispenso_shim` | libraries |
| `//arvr/libraries/dispenso:dispenso_static_impl` | executables, tests, benchmarks |
| `//arvr/libraries/dispenso:dispenso_shared_impl` | executables that link dispenso dynamically |
| `//arvr/libraries/dispenso:dispenso` | shim + static impl, when the split is not needed |
| `//arvr/libraries/dispenso:dispenso_shared` | shim + shared impl |

* **Libraries take the shim.** It carries the public headers and include
  directories and no implementation, so the library compiles with dispenso's
  symbols unresolved. Set `fbcode_undefined_symbols = True` on the library if
  it is not already.
* **Executables take exactly one implementation.** A binary, test or benchmark
  adds `dispenso_static_impl` (or `dispenso_shared_impl`) itself. Static is the
  usual choice.
* **Never put an implementation on a library.** That is what the split exists
  to prevent.

## Why

dispenso owns process-global state, the default thread pool most of all. If
several libraries in one process each embedded the implementation, the process
would end up with several global thread pools -- oversubscribing the machine --
and with an ODR violation. Deferring the choice to the link root also lets a
dependency graph mix static and shared consumers, which is impossible if every
library hard-codes its own linkage.

## Symptoms of getting it wrong

A missing implementation is not a build failure in the library. It surfaces as
undefined `dispenso::` symbols when the final binary links. If a new executable
fails that way, it needs an implementation dep, not a change to the library it
pulled in.

Two implementations in one link is the opposite mistake, and is quieter: it can
link successfully and misbehave at runtime through duplicated global state.

## This split is fbsource-only

The open source CMake build produces a single `dispenso` library and has no
shim, so there is nothing here to mirror into `CMakeLists.txt`.
