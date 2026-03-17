"""
test_pipeline.py
End-to-end test of the full LangGraph multi-agent pipeline with 5 test cases.

Usage:
    python test_pipeline.py
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from graph.workflow import run_workflow


# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "TC-1 Policy Query (no commute record)",
        "employee_id": "EMP_1001",
        "query": "What is the maximum hotel accommodation limit per night in Tier 1 cities for business travel?",
        "expected_intent": "policy_query",
        "commute_record": None,
    },
    {
        "name": "TC-2 Normal Delay Claim (not anomalous)",
        "employee_id": "EMP_1002",
        "query": "My metro was 20 minutes late today. Can I claim anything?",
        "expected_intent": "delay_claim",
        "commute_record": {
            "route_id": "ROUTE_05",
            "distance_km": 8.0,
            "delay_minutes": 20.0,
            "route_avg_delay_min": 12.0,
            "day_of_week": 2,
            "hour_of_day": 9,
            "claim_frequency_30d": 1.0,
            "week_num": 10,
            "is_holiday": 0,
        },
    },
    {
        "name": "TC-3 Anomalous Delay Claim (should be flagged)",
        "employee_id": "EMP_1003",
        "query": "My bus was 3 hours late. Please approve my ₹800 cab reimbursement.",
        "expected_intent": "delay_claim",
        "commute_record": {
            "route_id": "ROUTE_12",
            "distance_km": 6.0,
            "delay_minutes": 180.0,
            "route_avg_delay_min": 5.0,
            "day_of_week": 1,
            "hour_of_day": 9,
            "claim_frequency_30d": 18.0,
            "week_num": 11,
            "is_holiday": 0,
        },
    },
    {
        "name": "TC-4 Both (policy question + delay claim)",
        "employee_id": "EMP_1004",
        "query": "My cab was 45 minutes late today. What is the reimbursement policy and can I claim?",
        "expected_intent": "both",
        "commute_record": {
            "route_id": "ROUTE_03",
            "distance_km": 12.0,
            "delay_minutes": 45.0,
            "route_avg_delay_min": 10.0,
            "day_of_week": 0,
            "hour_of_day": 8,
            "claim_frequency_30d": 2.0,
            "week_num": 12,
            "is_holiday": 0,
        },
    },
    {
        "name": "TC-5 Out of scope",
        "employee_id": "EMP_1005",
        "query": "What is the office WiFi password?",
        "expected_intent": "out_of_scope",
        "commute_record": None,
    },
]

SEPARATOR = "=" * 72


def _print_result(tc: Dict[str, Any], state: Dict[str, Any], elapsed_ms: float) -> None:
    name = tc["name"]
    expected = tc["expected_intent"]
    detected = state.get("intent", "UNKNOWN")
    matched = "[OK]" if detected == expected else "[FAIL]"

    print(f"\n{SEPARATOR}")
    print(f"  {name}")
    print(SEPARATOR)
    print(f"  Intent expected : {expected}")
    print(f"  Intent detected : {detected}  {matched}")
    print(f"  Anomaly flagged : {state.get('is_anomalous', False)}")
    print(f"  Confidence      : {state.get('overall_confidence', 0.0):.2f}")
    print(f"  Needs escalation: {state.get('needs_escalation', False)}")
    print(f"  Processing time : {elapsed_ms:.0f} ms")

    errors = state.get("errors", [])
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")

    print("\n  --- Final Response ---")
    response = state.get("final_response", "(empty)")
    # Wrap long lines for terminal readability
    for chunk in [response[i:i+70] for i in range(0, len(response), 70)]:
        print(f"  {chunk}")

    if state.get("top_factors"):
        print("\n  --- SHAP Top Factors ---")
        for factor in state["top_factors"]:
            print(f"    - {factor}")


async def run_all_tests() -> None:
    print(f"\n{'#' * 72}")
    print("  Multi-Agent GenAI Pipeline — End-to-End Test Suite")
    print(f"{'#' * 72}")

    passed = 0
    failed = 0
    total_ms = 0.0

    for tc in TEST_CASES:
        t0 = time.perf_counter()
        try:
            state = await run_workflow(
                query=tc["query"],
                employee_id=tc["employee_id"],
                commute_record=tc.get("commute_record"),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_ms += elapsed_ms
            _print_result(tc, state, elapsed_ms)

            if state.get("intent") == tc["expected_intent"]:
                passed += 1
            else:
                failed += 1

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_ms += elapsed_ms
            print(f"\n{SEPARATOR}")
            print(f"  {tc['name']} — CRASHED")
            print(f"  Error: {exc}")
            print(f"  Time : {elapsed_ms:.0f} ms")
            failed += 1

    print(f"\n{SEPARATOR}")
    print("  TEST SUMMARY")
    print(SEPARATOR)
    print(f"  Total tests  : {len(TEST_CASES)}")
    print(f"  Intent match : {passed} passed, {failed} failed")
    print(f"  Total time   : {total_ms:.0f} ms  (avg {total_ms / len(TEST_CASES):.0f} ms/query)")
    print(SEPARATOR)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
