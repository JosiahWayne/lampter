"""TRES string parsing, which is how GPU counts are recovered."""

from __future__ import annotations

import pytest

from lampter.tres import gpu_count, gpu_type, memory, parse_tres


def test_parse_tres():
    assert parse_tres("cpu=8,mem=128G,node=1,billing=8,gres/gpu=1") == {
        "cpu": "8",
        "mem": "128G",
        "node": "1",
        "billing": "8",
        "gres/gpu": "1",
    }


@pytest.mark.parametrize("value", [None, "", "   ", "garbage", "novalue"])
def test_parse_tres_tolerates_junk(value):
    assert parse_tres(value) == {}


def test_parse_tres_skips_empty_fragments():
    assert parse_tres("cpu=4,,mem=64G,") == {"cpu": "4", "mem": "64G"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cpu=8,mem=128G,gres/gpu=1", 1),
        ("cpu=16,mem=128G,gres/gpu=2", 2),
        ("gres/gpu:h100=4", 4),
        ("gres/gpu:h200=2,gres/gpu=1", 3),
        # No GPU entry at all: the caller needs to tell this apart from "0 GPUs".
        ("cpu=4,mem=64G,node=1,billing=4", None),
        (None, None),
        ("", None),
    ],
)
def test_gpu_count(value, expected):
    assert gpu_count(value) == expected


def test_gpu_count_reports_zero_when_explicitly_zero():
    assert gpu_count("gres/gpu=0") == 0


def test_gpu_type():
    assert gpu_type("gres/gpu:h100=4") == "h100"
    assert gpu_type("gres/gpu=4") is None
    assert gpu_type(None) is None


def test_memory():
    assert memory("cpu=8,mem=128G") == "128G"
    assert memory("cpu=8") is None
