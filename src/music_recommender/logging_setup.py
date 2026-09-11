"""Shared logging configuration for the package.

The package defers logging configuration to its entry points: the CLI
configures logging in its app callback and the API configures it at import
time. Library modules obtain named loggers with ``logging.getLogger(__name__)``
so messages live in a predictable hierarchy regardless of the front end.
"""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Ensure the root logger emits formatted records at ``level``.

    The call is idempotent: it attaches a single ``StreamHandler`` to the root
    logger when none exists yet, so repeated calls (for example separate CLI
    invocations in one interpreter) do not stack handlers.
    """
    root = logging.getLogger()
    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    root.setLevel(level)
