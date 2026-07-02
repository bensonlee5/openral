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
    # The raw event is gone from the ring…
    assert not [e for e in snap["events"] if e["kind"] == "safety.violation"]
    # …but the persistent slot + counter still carry the violation.
    assert snap["topics"]["safety"]["last_violation"]["drop_reason"] == "collision"
    assert snap["counters"]["openral.event.safety_violation"] == 1


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
