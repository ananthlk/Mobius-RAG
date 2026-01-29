"""Unit tests for app.services.utils."""
import pytest

from app.services.utils import parse_json_response


def test_parse_json_response_plain():
    s = '{"a": 1, "b": "x"}'
    assert parse_json_response(s) == {"a": 1, "b": "x"}


def test_parse_json_response_markdown_json():
    s = '```json\n{"a": 1}\n```'
    assert parse_json_response(s) == {"a": 1}


def test_parse_json_response_markdown_no_lang():
    s = '```\n{"a": 1}\n```'
    assert parse_json_response(s) == {"a": 1}


def test_parse_json_response_preamble():
    s = 'Here is the analysis:\n\n{"summary": "ok", "facts": []}'
    assert parse_json_response(s) == {"summary": "ok", "facts": []}


def test_parse_json_response_trailing_extra_data():
    s = '{"a": 1} and some extra text'
    assert parse_json_response(s) == {"a": 1}


def test_parse_json_response_trailing_extra_data_newlines():
    s = '{"a": 1}\n\nAdditional explanation here.'
    assert parse_json_response(s) == {"a": 1}


def test_parse_json_response_invalid_raises():
    with pytest.raises(Exception):
        parse_json_response("not json at all")
