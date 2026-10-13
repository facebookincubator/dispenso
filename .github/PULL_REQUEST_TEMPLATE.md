# PR Details

<!--- Provide a general summary of your changes in the Title above -->

## Description

<!--- What does this change do, and why is it needed? What problem does it -->
<!--- solve? Link any related issue here. -->

## Test Plan

<!--- How did you test this? Include your testing environment, and the tests -->
<!--- you ran to see how the change affects other areas of the code. -->

## Types of changes

<!--- Tick all that apply: -->

- [ ] Docs change
- [ ] Refactoring
- [ ] Dependency upgrade
- [ ] Bug fix
- [ ] New feature

## Stable surface

<!--- The stable surface is public API and ABI, default behaviour, and -->
<!--- documented concurrency guarantees; undocumented scheduling and internal -->
<!--- threading are not. An addition that could reasonably break existing -->
<!--- documented use counts as an alteration, not an addition. See -->
<!--- GOVERNANCE.md for what each answer implies. If unsure, say so in the -->
<!--- description — it is a question for review, not a trap. -->

<!--- Tick all that apply: -->

- [ ] Adds to the stable surface, without changing compatibility or behaviour for existing documented use
- [ ] Alters or removes something present in the most recent tagged release
- [ ] Alters or removes something added since that release, and not yet shipped

<!--- If either alteration box applies, which part: -->

- [ ] Public API or ABI
- [ ] Default behaviour
- [ ] A documented concurrency guarantee — ordering, forward progress, cancellation, lifetime or safety

## Checklist

<!--- Tick all that apply. If you're unsure about any of these, don't -->
<!--- hesitate to ask. We're here to help! -->

- [ ] My code follows the code style of this project.
- [ ] I have run clang-format.
- [ ] I have updated the documentation, if this change needed it.
- [ ] I have read the **CONTRIBUTING** document.
- [ ] I have added tests to cover my changes.
- [ ] All new and existing tests passed, including in ASAN and TSAN modes (if available on your platform).
- [ ] If this adds to or alters the stable surface, I have added a release-notes line.
