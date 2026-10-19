"""Golden token-budget and reference-limit table for the v1.1.0 documentation.

The totals come from the measured/derived layout table in STATUS.md. This test
pins the published limits so a docs change cannot silently widen them. When P1
lands, add a cross-check against ``paiton_video.refpack.build_packed_layout``.
"""

from __future__ import annotations

import pytest

MAX_TOKENS = 49152
LIMITS = {"images": 9, "videos": 3, "audios": 3}

# (case, total packed rows S, fits under 49,152)
# 53,065 is the exact 1344x768 nine-ref `match` layout that fails closed in
# tests/test_h3_refpack.py::RefNodeTest::test_preflight_rejects_53k_before_engine_node.
# 44,625 is the exact 768p @512 preflight: 37,296 video + 414 audio +
# 3,360 ref rows + 3,555 text/vision.
CASES = [
    ("864x480 5s, 0 refs", 14985, True),
    ("864x480 5s, 1 ref", 17100, True),
    ("864x480 5s, 3 refs", 19000, True),
    ("864x480 5s, 9 refs", 26700, True),
    ("1344x768 5s, 0 refs", 37910, True),
    ("1344x768 5s, 9 refs @512", 44625, True),
    ("1344x768 5s, 9 refs @match", 53065, False),
    ("1344x768 5s, 9 refs @max", 55952, False),
    ("864x480 15s, 9 refs @512", 52877, False),
    ("cap boundary: 49,151", 49151, True),
    ("cap boundary: 49,152", 49152, True),
    ("cap boundary: 49,153", 49153, False),
]


@pytest.mark.parametrize("case,total,fits", CASES)
def test_budget_decision(case: str, total: int, fits: bool):
    assert (total <= MAX_TOKENS) is fits, case


def test_reference_limits_match_manifest_contract():
    assert LIMITS == {"images": 9, "videos": 3, "audios": 3}


def test_no_published_case_exceeds_cap():
    published = [(case, total) for case, total, fits in CASES if fits]
    assert published
    assert all(total <= MAX_TOKENS for _, total in published)
    assert any(total > MAX_TOKENS for case, total, fits in CASES if not fits)


def test_15s_768p_is_explicitly_unsupported():
    pathlib = pytest.importorskip("pathlib")
    for candidate in (
        pathlib.Path(__file__).resolve().parents[1] / "PLAN.md",
        pathlib.Path(__file__).resolve().parents[2] / "PLAN.md",
    ):
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
            assert "1344×768" in text and "not supported" in text, (
                "the 15 s 1344x768 rejection must stay documented in PLAN.md"
            )
            return
    pytest.skip("PLAN.md is not part of the packaged release tree")
