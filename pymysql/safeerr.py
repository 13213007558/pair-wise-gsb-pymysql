"""Opt-in safe projection of MySQL errors.

Responsibility boundary
-----------------------
By default (``safe_errors=False``) PyMySQL keeps its historical,
MySQLdb-compatible behavior: the message text sent by the server is
included verbatim in raised exceptions, and it is the *caller's*
responsibility to avoid forwarding it to untrusted parties.  Server
messages can embed business data (e.g. the conflicting key of a
duplicate-entry error), so applications may instead opt in to
``safe_errors=True``.

When enabled, any :class:`~pymysql.err.DatabaseError` raised while
connecting, executing a query, or streaming rows of an unbuffered
result is replaced by a *projected* exception that:

* keeps the same DB-API exception class, ``errno`` (``args[0]``) and
  ``sqlstate``, so error-code handling keeps working;
* exposes a ``retryable`` flag classifying transient failures;
* carries a generic, data-free message plus a correlation
  ``error_token`` instead of the server-supplied text.

The original error - including the server message - is only emitted to
the ``pymysql.safeerr`` logger (internal diagnostics), correlated by
the same token.  If the projection itself fails, a fully generic
``OperationalError`` is produced so the original text can never leak.
"""

import logging
import uuid
from contextlib import suppress

from . import err
from .constants import CR, ER

logger = logging.getLogger(__name__)

#: Error numbers that indicate a transient failure worth retrying.
RETRYABLE_ERRNOS = frozenset(
    {
        CR.CR_CONNECTION_ERROR,
        CR.CR_CONN_HOST_ERROR,
        CR.CR_SERVER_GONE_ERROR,
        CR.CR_SERVER_HANDSHAKE_ERR,
        CR.CR_SERVER_LOST,
        CR.CR_SERVER_LOST_EXTENDED,
        ER.CON_COUNT_ERROR,
        ER.TOO_MANY_USER_CONNECTIONS,
        ER.LOCK_WAIT_TIMEOUT,
        ER.LOCK_DEADLOCK,
        ER.SERVER_SHUTDOWN,
        ER.QUERY_TIMEOUT,
        ER.STATEMENT_TIMEOUT,
    }
)

#: Message used when even the projection itself fails.
GENERIC_MESSAGE = "MySQL error suppressed (safe_errors is enabled)"


def is_retryable_errno(errno):
    """Return True if *errno* denotes a transient, retryable failure."""
    return errno in RETRYABLE_ERRNOS


def is_projected(exc):
    """Return True if *exc* already went through safe projection."""
    return getattr(exc, "_pymysql_safe_projected", False)


def project_error(exc, phase):
    """Project *exc* into a caller-safe exception.

    Non-database exceptions (which never carry server text) and
    already-projected exceptions are returned unchanged, so nested
    phases (e.g. queries issued during connect) are not re-wrapped.
    """
    if not isinstance(exc, err.DatabaseError) or is_projected(exc):
        return exc
    try:
        return _project(exc, phase)
    except Exception:
        # The projection itself must never leak the original message.
        with suppress(Exception):
            logger.exception("safe error projection failed during phase=%s", phase)
        fallback = err.OperationalError(CR.CR_UNKNOWN_ERROR, GENERIC_MESSAGE)
        fallback.sqlstate = None
        fallback.errno = CR.CR_UNKNOWN_ERROR
        fallback.retryable = False
        fallback.phase = phase
        fallback.error_token = None
        fallback._pymysql_safe_projected = True
        return fallback


def _project(exc, phase):
    errno = exc.args[0] if exc.args and isinstance(exc.args[0], int) else 0
    sqlstate = getattr(exc, "sqlstate", None)
    token = uuid.uuid4().hex[:12]
    retryable = is_retryable_errno(errno)
    # Internal diagnostics: the original message only goes to the log,
    # correlated with the public exception via the token.
    logger.warning(
        "MySQL error suppressed (token=%s phase=%s errno=%s sqlstate=%s "
        "retryable=%s): %r",
        token,
        phase,
        errno,
        sqlstate,
        retryable,
        exc,
    )
    message = (
        f"MySQL error {errno} during {phase} "
        f"(sqlstate={sqlstate}, retryable={retryable}, token={token})"
    )
    projected = exc.__class__(errno, message, sqlstate=sqlstate)
    projected.errno = errno
    projected.retryable = retryable
    projected.phase = phase
    projected.error_token = token
    projected._pymysql_safe_projected = True
    return projected
