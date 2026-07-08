# Design decisions

OpenRAL records its architecture and design decisions as Architecture
Decision Records (ADRs). As of 2026-07-08 the ADR log itself lives in the
private [`OpenRAL/management`](https://github.com/OpenRAL/management) repo,
under `adr/` — not in this public repo.

ADR-XXXX identifiers still appear throughout this codebase's comments,
docstrings, and docs (`ADR-0012`, `ADR-0057`, `ADR-0083`, ...). They are
plain-text references to the private log, not links — contributors without
access to `OpenRAL/management` can ask about a specific decision by opening
an issue in this repo.

The ADR discipline is unchanged: adding, removing, renaming, or moving a
responsibility between the eight architecture layers (§3 of
[CLAUDE.md](https://github.com/OpenRAL/openral/blob/master/CLAUDE.md))
requires an ADR, written before the code that implements it.

## Licensing & commercial boundary

Two of those decisions define OpenRAL's public/private boundary and are
worth stating here in full, since the maintainer wants this posture visible
even though the ADR prose that established it is now private:

- **The public `OpenRAL/openral` repo is uniformly Apache-2.0.** Every
  package in this repo ships under a single permissive license — no
  source-available tier, no BSL, no per-package license drift. Copy-left
  dependencies are rejected without TSC review.
- **Commercial capabilities live in a separate, private monorepo**
  (`OpenRAL/openral-pro`), not in this repo. That includes the TensorRT/NVMM
  zero-copy runtime fast path, WAM (World Action Model) implementations,
  fleet/cloud dispatch, and future premium rSkills. This repo retains the
  protocols and extension seams those capabilities plug into; it does not
  ship the implementations themselves.
- Third-party model **weights** (e.g. NVIDIA GR00T checkpoints) keep their
  own upstream license, independent of OpenRAL's code license — this is
  compliance for models OpenRAL doesn't own, not a statement about OpenRAL's
  own code.

These two points were originally decided in ADR-0012 (uniform Apache-2.0)
and ADR-0083 (the OpenRAL Pro commercial tier, which supersedes ADR-0012's
earlier "no commercial tier, ever" commitment while retaining its
public-repo-is-Apache-2.0 posture). Full context, alternatives considered,
and consequences are in the private ADR log.
