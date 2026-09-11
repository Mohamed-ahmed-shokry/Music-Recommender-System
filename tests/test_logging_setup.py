import logging

import pytest

from music_recommender.logging_setup import configure_logging


def _stream_handler_count() -> int:
    return sum(
        1
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.StreamHandler)
    )


def test_configure_logging_is_idempotent() -> None:
    before = _stream_handler_count()

    configure_logging(logging.INFO)
    after_first = _stream_handler_count()

    configure_logging(logging.WARNING)
    configure_logging(logging.DEBUG)
    after_repeat = _stream_handler_count()

    assert after_first in (before, before + 1)
    assert after_repeat == after_first
    assert logging.getLogger().level in (logging.DEBUG, logging.WARNING)


def test_configure_logging_emits_records_to_root(
    caplog: pytest.LogCaptureFixture,
) -> None:
    emitter = logging.getLogger("music_recommender.logging_setup.tester")

    with caplog.at_level(logging.INFO):
        configure_logging(logging.INFO)
        emitter.info("hello from logger")

    assert any(record.getMessage() == "hello from logger" for record in caplog.records)
