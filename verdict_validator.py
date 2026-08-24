#!/usr/bin/env python3
"""Deterministic post-verdict validator for the ta-shadow stock lane.

Spec 2026-08-24 §10: the LLM verdict is an advisory proposal. This module
validates the parsed verdict + trade params against the immutable snapshot
and fails CLOSED to HOLD/NO_ACTION with machine-readable rejection reasons.

No LLM is involved. No override path exists — a validation failure is final.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

# --- Configuration (spec §10 gates) --------------------------------------
MIN_RISK_REWARD = 1.0       # TP-SL distance must be >= 1R
MIN_ATR_DISTANCE = 0.15     # entry-to-stop must be >= 0.15 * ATR14 (tight-stop guard)
MAX_LEVEL_DEVIATION = 0.05  # invented-level tolerance vs snapshot fields (5%)
REQUIRED_SNAPSHOT_FIELDS = (
    "close", "sma20", "sma50", "high20", "low20", "atr14",
    "pivot_p", "r1", "r2", "s1", "s2",
)


def snapshot_identity(snapshot: dict) -> dict:
    """Immutable snapshot identity (spec §8): id, version, timestamp, hash.

    The hash covers the full snapshot so any drift (re-fetch, stale data,
    mutated levels) is caught by the validator.
    """
    canonical = json.dumps(snapshot, sort_keys=True, default=str)
    return {
        "snapshot_id": hashlib.sha256(canonical.encode()).hexdigest()[:16],
        "snapshot_version": "1.0",
        "snapshot_timestamp": snapshot.get("as_of"),
        "snapshot_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "dmc_version": "htf-daily-body-levels-v1",
        "instrument": snapshot.get("symbol"),
        "timeframe_set": "HTF daily + LTF session (5m/1m when present)",
        "data_freshness_status": "fresh" if snapshot.get("as_of") else "unknown",
    }


def validate_verdict(
    verdict: str,
    params: dict,
    snapshot: dict,
    role_llms: dict | None = None,
) -> dict:
    """Deterministic post-verdict validation. Fail-closed to HOLD.

    Returns {verdict, params, ok, reasons[], snapshot_id, validated_at}.
    ``verdict`` becomes HOLD and trade params are nulled on any failure.
    """
    reasons: list[str] = []
    levels = snapshot.get("levels") or {}
    identity = snapshot_identity(snapshot)
    ok = True

    # --- snapshot completeness (spec §10: missing/stale/malformed) ----
    missing = [f for f in REQUIRED_SNAPSHOT_FIELDS if f not in levels]
    if missing:
        ok = False
        reasons.append(f"snapshot missing fields: {missing}")

    # --- schema completeness (spec §9/§10) ----------------------------
    if verdict not in ("BUY", "SELL", "HOLD", "NO_ACTION"):
        ok = False
        reasons.append(f"invalid verdict label: {verdict!r}")

    # --- execution type (spec §9: market | limit | none) ---------------
    execution = (params.get("execution") or "limit").lower()
    if execution not in ("market", "limit", "none"):
        ok = False
        reasons.append(f"invalid execution type: {execution!r}")
    elif execution == "none" and verdict in ("BUY", "SELL"):
        ok = False
        reasons.append("execution 'none' with a directional verdict")

    # --- provider/model identity (spec §10) ----------------------------
    if not role_llms:
        ok = False
        reasons.append("role/model identity missing")

    # --- numeric-level provenance (spec §10) ----------------------------
    # Every TP/SL/entry must be derivable from a snapshot field (close,
    # pivot, R/S, SMA, 20d high/low, DMC body levels) within tolerance.
    if verdict in ("BUY", "SELL") and levels:
        allowed = [
            levels.get("close"), levels.get("prior_close"),
            levels.get("pivot_p"), levels.get("r1"), levels.get("r2"),
            levels.get("s1"), levels.get("s2"),
            levels.get("sma20"), levels.get("sma50"),
            levels.get("high20"), levels.get("low20"),
        ]
        # DMC body levels appear in snapshot["dmc"] text; parse numbers.
        dmc_text = str(snapshot.get("dmc") or "")
        import re as _re
        allowed += [float(x) for x in _re.findall(r"\d+\.?\d*", dmc_text)
                    if _is_sane_price(x)]
        allowed = [a for a in allowed if a is not None]

        for field in ("entry", "take_profit", "stop_loss"):
            val = params.get(field)
            if val is None:
                continue
            # market entry: fill price unknown until the open — the stated
            # entry is only a reference, so skip provenance for it.
            if field == "entry" and execution == "market":
                continue
            if not any(abs(val - a) / max(a, 1e-9) <= MAX_LEVEL_DEVIATION
                       for a in allowed):
                ok = False
                reasons.append(
                    f"{field}={val} not derivable from snapshot levels")

    # --- direction consistency (spec §10) ------------------------------
    entry = params.get("entry")
    tp = params.get("take_profit")
    sl = params.get("stop_loss")
    if verdict == "BUY" and (tp is not None or sl is not None):
        if tp is not None and entry is not None and tp <= entry:
            ok = False
            reasons.append(f"BUY take_profit {tp} not above entry {entry}")
        if sl is not None and entry is not None and sl >= entry:
            ok = False
            reasons.append(f"BUY stop_loss {sl} not below entry {entry}")
    if verdict == "SELL" and (tp is not None or sl is not None):
        if tp is not None and entry is not None and tp >= entry:
            ok = False
            reasons.append(f"SELL take_profit {tp} not below entry {entry}")
        if sl is not None and entry is not None and sl <= entry:
            ok = False
            reasons.append(f"SELL stop_loss {sl} not above entry {entry}")

    # --- ATR distance + min risk/reward (spec §10) ----------------------
    if verdict in ("BUY", "SELL") and entry and tp and sl:
        atr = levels.get("atr14")
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if atr and risk < MIN_ATR_DISTANCE * atr:
            ok = False
            reasons.append(f"stop distance {risk:.2f} < 0.15*ATR14 ({0.15*atr:.2f})")
        if reward / max(risk, 1e-9) < MIN_RISK_REWARD:
            ok = False
            reasons.append(f"risk/reward {reward/max(risk,1e-9):.2f} < {MIN_RISK_REWARD}")

    # --- fail closed -----------------------------------------------------
    if not ok:
        return {
            "verdict": "HOLD",
            "confidence": None,
            "params": {"entry": None, "take_profit": None,
                       "stop_loss": None, "risk_reward": None},
            "ok": False,
            "reasons": reasons,
            **identity,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "verdict": verdict,
        "confidence": None,  # confidence is advisory; caller reattaches from parse
        "params": params,
        "ok": True,
        "reasons": [],
        **identity,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }


def _is_sane_price(x) -> bool:
    try:
        return 0.01 < float(x) < 1_000_000
    except (TypeError, ValueError):
        return False
