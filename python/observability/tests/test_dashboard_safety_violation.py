"""Safety-violation surfacing in the dashboard store.

A ``safety.check`` span whose ``safety.severity == "violation"`` (the C++
kernel dropping an action — self-collision / envelope breach) must reach the
operator on two durable surfaces, because the raw per-span event is severity
``info`` (the kernel span's own status is OK — dropping the action IS the
kernel working) and the 30 Hz ``hal.read_state`` stream evicts it from the
200-slot event ring within seconds:

1. ``topics.safety.last_violation`` — a persistent slot only the next
   violation overwrites (the per-check ledger row resets on the next OK check).
2. a dedicated ``safety.violation`` error-severity event + the
   ``openral.event.safety_violation`` counter the UI's Safety tile reads.

Real protobuf spans, no mocks (CLAUDE.md §1.11).
"""

from __future__ import annotations

import time

from openral_observability.dashboard import TelemetryStore
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import (
    ResourceSpans,
    ScopeSpans,
    Span,
    Status,
)


def _av(value: object) -> AnyValue:
    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    if isinstance(value, float):
        return AnyValue(double_value=value)
    return AnyValue(string_value=str(value))


def _attrs(d: dict[str, object]) -> list[KeyValue]:
    return [KeyValue(key=k, value=_av(v)) for k, v in d.items()]


def _safety_span(attrs: dict[str, object]) -> Span:
    start = time.time_ns()
    return Span(
        trace_id=b"\x09" * 16,
        span_id=b"\x09" * 8,
        name="safety.check",
        start_time_unix_nano=start,
        end_time_unix_nano=start + 300_000,  # 0.3 ms — matches the kernel span
        attributes=_attrs(attrs),
        status=Status(code=0),  # kernel span is OK even on a violation
    )


def _wrap(*spans: Span) -> list[ResourceSpans]:
    return [
        ResourceSpans(
            resource=Resource(attributes=_attrs({"service.name": "openral_safety_kernel"})),
            scope_spans=[ScopeSpans(spans=list(spans))],
        )
    ]


_VIOLATION_ATTRS = {
    "safety.check_name": "envelope",
    "safety.kernel": "cpp",
    "safety.severity": "violation",
    "safety.drop_reason": "collision",
    "safety.collision_mode": 0,
    "safety.violation_value": -0.0841,
    "rskill.id": "OpenRAL/rskill-smolvla-so101-pen",
}


def test_violation_populates_persistent_slot() -> None:
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))
    lv = store.snapshot()["topics"]["safety"]["last_violation"]
    assert lv["drop_reason"] == "collision"
    assert lv["violation_value"] == -0.0841
    assert lv["rskill_id"] == "OpenRAL/rskill-smolvla-so101-pen"


def test_violation_emits_error_event_and_counter() -> None:
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))
    snap = store.snapshot()

    violations = [e for e in snap["events"] if e["kind"] == "safety.violation"]
    assert len(violations) == 1
    assert violations[0]["severity"] == "error"
    assert "collision" in violations[0]["title"]
    # The counter the UI's Safety tile reads (cnt-safety).
    assert snap["counters"]["openral.event.safety_violation"] == 1


def test_persistent_slot_survives_a_high_rate_span_flood() -> None:
    """The last_violation slot outlives the event-ring eviction the flood causes."""
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))

    # Flood the 200-slot event ring with unrelated high-rate spans — exactly
    # what evicts the raw per-span event on a live deploy.
    start = time.time_ns()
    flood = [
        Span(
            trace_id=b"\x01" * 16,
            span_id=bytes([i % 256]) * 8,
            name="hal.read_state",
            start_time_unix_nano=start,
            end_time_unix_nano=start + 1_000_000,
            attributes=_attrs({"openral.hal.adapter": "so100"}),
            status=Status(code=0),
        )
        for i in range(300)
    ]
    store.ingest_spans(_wrap(*flood))

    snap = store.snapshot()
    # The safety.violation event now SURVIVES the flood via the protected error
    # lane (it used to be evicted from the shared 200-slot ring within seconds,
    # leaving no trace) …
    assert [e for e in snap["events"] if e["kind"] == "safety.violation"]
    # … and the persistent slot + counter still carry the violation too.
    assert snap["topics"]["safety"]["last_violation"]["drop_reason"] == "collision"
    assert snap["counters"]["openral.event.safety_violation"] == 1


def test_error_events_survive_high_rate_flood_via_protected_lane() -> None:
    """Any error event (skill_failure, estop, ...) outlives the main-ring flood.

    The shared 200-slot event ring cycles in ~seconds under a 30 Hz stream; the
    protected error lane keeps the last N error/fatal events so the operator can
    still find WHY the robot stopped. Generic — not tied to safety.violation.
    """
    store = TelemetryStore()
    # An error-status span → a synthesised error-severity event (this is how a
    # reasoner skill-failure / a HAL estop surface a red row on the dashboard).
    es = time.time_ns()
    err_span = Span(
        trace_id=b"\x07" * 16,
        span_id=b"\x07" * 8,
        name="reasoner.skill_failure",
        start_time_unix_nano=es,
        end_time_unix_nano=es + 1_000,
        status=Status(code=2),  # ERROR
    )
    store.ingest_spans(_wrap(err_span))

    # Flood the 200-slot main ring well past capacity with info spans.
    fs = time.time_ns()
    flood = [
        Span(
            trace_id=b"\x01" * 16,
            span_id=bytes([i % 256]) * 8,
            name="hal.read_state",
            start_time_unix_nano=fs,
            end_time_unix_nano=fs + 1_000,
            status=Status(code=0),
        )
        for i in range(400)
    ]
    store.ingest_spans(_wrap(*flood))

    snap = store.snapshot()
    error_kinds = [e["kind"] for e in snap["events"] if e["severity"] in ("error", "fatal")]
    assert "reasoner.skill_failure" in error_kinds


def test_ok_check_does_not_overwrite_last_violation() -> None:
    """A subsequent passing check resets its ledger pill but not last_violation."""
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))
    store.ingest_spans(
        _wrap(
            _safety_span(
                {
                    "safety.check_name": "envelope",
                    "safety.kernel": "cpp",
                    "safety.severity": "ok",
                    "safety.clamped": False,
                }
            )
        )
    )
    safety = store.snapshot()["topics"]["safety"]
    assert safety["checks"]["envelope"]["severity"] == "ok"  # pill reset
    assert safety["last_violation"]["drop_reason"] == "collision"  # slot survives


def _ok_span() -> Span:
    return _safety_span(
        {"safety.check_name": "envelope", "safety.kernel": "cpp", "safety.severity": "ok"}
    )


def test_estopped_flag_defaults_false() -> None:
    """Before any safety activity the dashboard shows the robot as runnable."""
    assert TelemetryStore().snapshot()["topics"]["safety"]["estopped"] is False


def test_estopped_flag_latches_on_violation_and_clears_on_ok() -> None:
    """The separate Reset control's visibility follows the kernel latch.

    A violation (self-collision, envelope, or an /openral/estop drop) latches
    the kernel → ``estopped`` True → the UI reveals the Reset e-stop button. A
    subsequent passing check means the kernel is running clean again →
    ``estopped`` False → Reset is hidden. The red E-STOP is always present and
    always a stop; only Reset is state-driven. Self-corrects after a reset with
    no rclpy node.
    """
    store = TelemetryStore()
    store.ingest_spans(_wrap(_safety_span(_VIOLATION_ATTRS)))
    assert store.snapshot()["topics"]["safety"]["estopped"] is True
    store.ingest_spans(_wrap(_ok_span()))
    assert store.snapshot()["topics"]["safety"]["estopped"] is False
