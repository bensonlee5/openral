# ADR-0083: OpenRAL Pro — a private commercial tier, superseding ADR-0012's "no commercial tier, ever"

- Status: Accepted
- Date: 2026-07-08
- Related: supersedes [ADR-0012](0012-open-core-licensing.md) (commercial-tier
  commitment only — its "public repo is uniformly Apache-2.0" posture is
  retained); precedent [ADR-0004](0004-monorepo-over-polyrepo.md) §Context
  (deliberately-separated repos: `openral/cloud`, `openral/contrib-closed-shims`);
  relocates implementations from [ADR-0011](0011-nvmm-handoff.md) and
  [ADR-0082](0082-nvmm-vla-vision-zero-copy.md); touches the dispatcher surface
  anticipated in [ADR-0018](0018-ros2-reasoner-supervisor.md) §"Cloud-dispatch
  hook"; CLAUDE.md §1.9 (license lineage)

## Context

ADR-0012 (amended 2026-06-16) withdrew a previously-proposed two-tier
open-core model and committed OpenRAL to being, in its own words, "a fully
open-source project under a single permissive license" with "no Tier 2, no
source-available tier, and no BSL tier" — full stop, indefinitely. Its
"Revenue strategy is out of scope" consequence explicitly deferred any future
commercial offering to "its own ADR and TSC decision, not as a silent license
change here." This ADR is that decision.

The maintainer (Adrian, 2026-07-08) has decided the project needs a
commercial tier to be sustainable. Unlike the 2026-05-24 proposal ADR-0012
withdrew — which tried to gate core orchestration layers (`reasoner`, `wam`,
`dispatcher`, `skill_catalog`, `fleet`) behind a source-available license
inside the same repo — this decision does two things differently:

1. It draws the boundary around genuinely new or genuinely separable
   capability (a proprietary runtime fast path, unbuilt fleet/cloud
   dispatch, future premium weights), not around the reasoner or safety
   stack that make the open core useful on its own.
2. It ships the commercial code in a **separate private repository**, not a
   second license inside the public one — reusing the precedent ADR-0004
   already established for `openral/cloud` and
   `openral/contrib-closed-shims`: split out when licensing, audience, or
   release cadence differ, not to fragment the open core.

**Why this reverses a public commitment.** ADR-0012 said "no commercial tier,
ever," and this ADR walks that back. That is a real cost — it is a promise
made to the community — and the honest reason is project sustainability, not
new information about licensing law or a change of philosophy on openness.
The maintainer judged that an entirely-volunteer/self-funded open-source
robotics stack does not survive to the point of being useful to the
community it was built for. The mitigation is keeping the boundary narrow
and everything load-bearing for an independent deployment (reasoner, safety,
HAL, sim/benchmark, CLI, ONNX/PyTorch backends, all schemas and IDL) Apache-2.0
forever, so a self-hosted deployment on commodity hardware is never crippled
by the split.

## Decision

**Introduce OpenRAL Pro**: a private monorepo at `OpenRAL/openral-pro`,
packages named `openral-pro-*`, holding commercial-only capability. The
public `OpenRAL/openral` repo remains **uniformly Apache-2.0** — ADR-0012's
licensing-of-the-public-repo decision is retained verbatim; only its
"there will never be a commercial tier" commitment is superseded.

### What moves to OpenRAL Pro

Exactly four categories, each because it is either a genuine hardware-margin
runtime advantage, unbuilt capability with no open baseline to protect, or
a distribution channel that was already going to be gated (weights):

1. **The TensorRT engine runtime + NVMM/GStreamer zero-copy fast path.**
   Concretely: `openral_rskill.runtime_tensorrt` (ONNX→TensorRT engine
   build/run, `python/rskill/src/openral_rskill/runtime_tensorrt.py`), the
   NVMM zero-copy tee consumers under
   `openral_runner.backends.gstreamer.{trt_nvmm,nvmm_detector,nvmm_vision_encoder,act_nvmm}`
   (`python/runner/src/openral_runner/backends/gstreamer/`), and the
   `tensorrt` dependency group in the root `pyproject.toml`
   (`tensorrt-cu13` pin + build-on-load engine caching). This is the
   ADR-0011 / ADR-0082 NVMM-native, in-pipeline, zero-copy vision path — a
   real hardware-utilization advantage on NVIDIA GPUs, not a capability the
   open core needs to be a working robot-agent harness.
2. **WAM implementations.** `python/wam/` today holds no concrete
   implementation to move — it is the `WorldModel` Protocol, the `Rollout`
   contract, and a test-only `NullWorldModel` stub, with zero consumers
   repo-wide. That Protocol package stays open and unchanged, per
   CLAUDE.md's "types are the contract" principle: the interface a
   third-party world model plugs into must be public and stable. The
   roadmap's planned generative adapters (Cosmos Predict, UnifoLM-WMA-0,
   IRASim — v0.3+ per `python/wam/README.md`) are **built in
   `openral-pro`** as private `openral-pro-wam-*` packages depending on the
   public `openral-wam` Protocol — the same "born private" framing as
   fleet/cloud dispatch (item 3), not an extraction from existing code.
3. **Fleet/cloud dispatch.** Not yet built. ADR-0018 already anticipated a
   cloud-dispatch hook at the reasoner's LLM-endpoint selection boundary
   ("a cloud endpoint is just another endpoint... cloud-offloading heavy
   skills is a separate ADR") and, under the old ADR-0012 model, assumed it
   would be Apache-2.0. It is born in `openral-pro` instead — there is no
   open implementation to carve away from, so nothing regresses for
   existing users. Multi-robot fleet coordination and the dispatcher surface
   referenced by CLAUDE.md §3 ("The dispatcher (edge/cloud/split)... is
   Apache-2.0") move under this ADR; §3 is corrected by the CLAUDE.md
   update in the Consequences section below.
4. **Future premium rSkills**, distributed as private Hugging Face Hub
   repos rather than public ones. The rSkill **format, loader, and license
   checks** (`python/rskill/`, ADR-0006 packaging, the
   `rskill.unverified_provenance` warning, `OPENRAL_REQUIRE_SIGNED_SKILLS`)
   stay open and unchanged — a private rSkill is loaded by the same open
   loader with private Hub credentials, exactly as a commercial VLA
   checkpoint already would be. Only the *weights and manifest* for
   specific premium skills are private; nothing about how OpenRAL discovers,
   validates, or runs an rSkill becomes proprietary.

### What stays open (explicit, not by omission)

Reasoner (S2) and all replanning/tool-call logic; observability +
dashboard; every Pydantic schema in `openral_core` and every IDL in
`packages/openral_msgs`; the HAL (`packages/openral_hal_*/`); the safety
kernel (`packages/openral_safety/`, `cpp/openral_safety_kernel/`); the
rSkill format/loader/license-check machinery; sim and benchmark
(`python/sim/`); the `openral` CLI; ONNX export tooling; and the PyTorch /
plain-ONNX-Runtime inference backends (a TRT-less deployment keeps working,
just without the zero-copy fast path). This list is the same "safety beats
helpfulness" / "types are the contract" boundary CLAUDE.md already draws —
this ADR does not touch it.

### The extraction seam: a runtime-backend registry

To let `openral-pro-trt` (or a third party's own TRT/NVMM package) plug in
without the open `openral-runner` depending on private code, `openral-runner`
gains an entry-point-based runtime-backend registry:
`openral.runtime_backends`. A backend registers itself via a Python entry
point; `openral-runner` discovers installed backends at runtime and selects
by name/config, the same discovery pattern the codebase already uses for
other optional-dependency surfaces. Requesting a backend that is not
installed (e.g. `"tensorrt"` on a host without `openral-pro-trt`) raises a
typed `ROSConfigError` naming the pro package needed — explicit, no silent
fallback to a slower backend and no silent skip, per CLAUDE.md §1.4
("Explicit beats implicit") and §1.10.

### Monetization (context, not the core of this ADR)

Manual-first: Stripe payment links or invoices, hand-provisioned access to a
private PyPI-compatible wheel index and private Hugging Face Hub org/repos.
No billing automation, license-key server, or usage metering ships as part
of this decision; if one is built later it gets its own ADR.

## Consequences

- **ADR-0012 is superseded for its commercial-tier commitment only.** Its
  "public repo is uniformly Apache-2.0" decision, its Apache-2.0-for-all-of
  `python/`/`packages/`/`cpp/` scope, and its third-party-weight-licensing
  section are all retained unchanged — nothing in the *public* repo's license
  changes as a result of this ADR. ADR-0012 gets a dated append-only
  amendment (per ADR-0001) pointing here; its body is not rewritten.
- **ADR-0011 (`nvmm-handoff`) and ADR-0082 (NVMM VLA vision zero-copy)
  implementations relocate to `openral-pro`.** Their design documents stay
  in the public `docs/adr/` as historical record of the architecture (per
  ADR-0001, ADRs are never deleted), but the code they describe —
  `openral_rskill.runtime_tensorrt` and the `openral_runner.backends.gstreamer`
  NVMM modules listed above — moves to the private monorepo in a follow-up
  extraction PR. Those two ADRs get their own dated amendment noting the
  relocation once the extraction lands; this ADR does not itself move code.
- **CLAUDE.md §1.9 is updated in the same change as this ADR**: replace "no
  commercial / source-available / non-open tier" wording with "the public
  repo is uniformly Apache-2.0; commercial capabilities (TensorRT/NVMM
  runtime, WAM implementations, fleet/cloud dispatch, premium rSkills) live
  in the private OpenRAL Pro monorepo per ADR-0083" — the weight-license
  lineage paragraph (GR00T N1/N1.5/N1.6 non-commercial, N1.7+ Open Model
  License) is unchanged, since it was never about OpenRAL's own licensing.
- **CLAUDE.md §3's dispatcher line** ("The dispatcher (edge/cloud/split),
  like every OpenRAL package, is Apache-2.0") is now only true for the
  edge/local dispatch path that ships in the open reasoner; cloud/fleet
  dispatch is OpenRAL Pro. Corrected in the same CLAUDE.md edit.
- **No package under `python/`, `packages/`, or `cpp/` that exists today
  changes license or is deleted.** The extraction is additive at the seam
  (the new `openral.runtime_backends` registry) and subtractive only for the
  TRT/NVMM implementation files named above, in a follow-up PR — this ADR
  authorizes the split; it does not perform it.
- **A TRT-less deployment is not degraded functionally**, only in raw
  throughput/latency: the PyTorch and plain-ONNX-Runtime backends remain the
  open default and stay fully supported, per "stays open" above.
- **A third party could write a competing `openral.runtime_backends`
  plugin** instead of buying OpenRAL Pro — the registry is a public
  extension point, not a license gate. That is accepted as the cost of
  keeping the seam honest (CLAUDE.md §1.4); the commercial value is the
  maintained, pre-built, tested TensorRT/NVMM implementation and support,
  not exclusivity of the extension point.
- **Repo-state-map and docs/methods/ updates are deferred to the extraction
  PR** (Phase 1a/1b in the tracking plan), not this ADR — no code moves
  here, so there is nothing on the map to flip yet.

## Alternatives considered

- **Reopen ADR-0012's original two-tier model (source-available license
  inside the same repo).** Rejected again, for the same reason it failed
  the first time: gating `reasoner`/`wam` behind PolyForm inside the public
  repo means every clone ships license-ambiguous code, complicates OSI
  status for the whole tree, and — per ADR-0012's own retrospective —
  produced zero shipped lines of licensed code in its prior attempt. A
  separate private repo has none of these problems.
- **BSL/SSPL/Elastic-License-v2 on the extracted packages, kept in the
  public repo.** Rejected — ADR-0012 already surveyed and rejected this
  family (Datadog/Grafana/Sentry/Elastic-style commercial licenses) for the
  same OSI-purity and multi-license-per-repo reasons; nothing about this
  decision changes that analysis. A closed private repo is more honest than
  a source-visible-but-restricted one.
- **Keep everything Apache-2.0 and monetize services only (support
  contracts, hosted deployment, consulting) with zero private code.**
  Considered, but rejected as insufficient on its own: it does not capture
  the TensorRT/NVMM engineering investment as a differentiator, and it does
  not give a place for premium rSkill weights that must not be redistributed
  under an open weights license. Nothing in this ADR forecloses *also*
  selling services on top of the open core.
- **Fold OpenRAL Pro into the existing `openral/cloud` repo instead of a new
  `openral/openral-pro`.** Rejected — `openral/cloud` is scoped to the
  hosted observability/fleet control plane (ADR-0004); the TRT/NVMM runtime
  and premium rSkills are not cloud/hosted concerns, they are on-prem
  commercial code. A dedicated repo keeps `openral/cloud`'s scope legible
  and lets the two evolve on independent release cadences.
