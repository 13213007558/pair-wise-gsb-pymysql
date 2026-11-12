"""Tests for the opt-in safe error projection (pymysql.safeerr).

Server errors are injected as fake wire packets carrying a placeholder
sensitive string.  Tests verify the whole public surface of the raised
exception (class, errno, sqlstate, retryable flag, formatted output),
the internal diagnostic log, and the fallback when projection fails -
not just message string replacement.
"""

import io
import logging
import struct
import traceback

import pytest

from pymysql import err, safeerr
from pymysql.connections import Connection, MySQLResult
from pymysql.constants import CR, ER

SENSITIVE = "ACCT-9001:alice@example.com"


def error_payload(errno, message, sqlstate=None):
    payload = b"\xff" + struct.pack("<H", errno)
    if sqlstate is not None:
        payload += b"#" + sqlstate.encode()
    return payload + message.encode()


def frame(payload, seq):
    return len(payload).to_bytes(3, "little") + bytes([seq]) + payload


class FakeSocket:
    """Minimal socket stand-in serving pre-built wire packets."""

    def __init__(self, *frames):
        self._file = io.BytesIO(b"".join(frames))
        self.sent = b""

    def settimeout(self, timeout):
        pass

    def setsockopt(self, *args):
        pass

    def makefile(self, mode):
        return self._file

    def sendall(self, data):
        self.sent += data

    def close(self):
        pass


def make_connection(*frames, safe_errors=True):
    """A deferred Connection wired to a FakeSocket, ready to read packets."""
    conn = Connection(defer_connect=True, safe_errors=safe_errors)
    conn._sock = FakeSocket(*frames)
    conn._rfile = conn._sock.makefile("rb")
    conn._next_seq_id = 0
    conn._current_timeout = None
    return conn


def assert_no_leak(exc):
    haystack = (
        str(exc),
        repr(exc),
        "".join(traceback.format_exception(exc)),
        repr(vars(exc)),
    )
    for text in haystack:
        assert SENSITIVE not in text


# ---------------------------------------------------------------------------
# Unit-level projection behavior
# ---------------------------------------------------------------------------


def test_project_preserves_class_errno_sqlstate_and_retryability():
    original = err.IntegrityError(
        ER.DUP_ENTRY, f"Duplicate entry '{SENSITIVE}'", sqlstate="23000"
    )
    projected = safeerr.project_error(original, "execute")

    assert type(projected) is err.IntegrityError
    assert projected.args[0] == ER.DUP_ENTRY
    assert projected.sqlstate == "23000"
    assert projected.errno == ER.DUP_ENTRY
    assert projected.retryable is False
    assert projected.phase == "execute"
    assert projected.error_token
    assert_no_leak(projected)


@pytest.mark.parametrize(
    "errno, retryable",
    [
        (ER.LOCK_DEADLOCK, True),
        (ER.LOCK_WAIT_TIMEOUT, True),
        (ER.CON_COUNT_ERROR, True),
        (CR.CR_SERVER_GONE_ERROR, True),
        (CR.CR_SERVER_LOST, True),
        (CR.CR_CONN_HOST_ERROR, True),
        (ER.DUP_ENTRY, False),
        (ER.NO_SUCH_TABLE, False),
        (ER.SYNTAX_ERROR, False),
    ],
)
def test_retryable_classification(errno, retryable):
    exc = err.OperationalError(errno, f"boom {SENSITIVE}")
    projected = safeerr.project_error(exc, "execute")
    assert projected.args[0] == errno
    assert projected.retryable is retryable
    assert safeerr.is_retryable_errno(errno) is retryable


def test_non_database_errors_pass_through_unchanged():
    interface_error = err.InterfaceError(0, "")
    assert safeerr.project_error(interface_error, "execute") is interface_error
    value_error = ValueError(f"client bug {SENSITIVE}")
    assert safeerr.project_error(value_error, "execute") is value_error


def test_projection_is_idempotent():
    original = err.OperationalError(ER.LOCK_DEADLOCK, f"deadlock {SENSITIVE}")
    once = safeerr.project_error(original, "execute")
    twice = safeerr.project_error(once, "connect")
    assert twice is once


def test_internal_log_keeps_original_cause_with_token(caplog):
    original = err.IntegrityError(
        ER.DUP_ENTRY, f"Duplicate entry '{SENSITIVE}'", sqlstate="23000"
    )
    with caplog.at_level(logging.WARNING, logger="pymysql.safeerr"):
        projected = safeerr.project_error(original, "execute")

    records = [r for r in caplog.records if SENSITIVE in r.getMessage()]
    assert len(records) == 1
    assert projected.error_token in records[0].getMessage()
    assert str(ER.DUP_ENTRY) in records[0].getMessage()


def test_projection_failure_falls_back_without_leak(monkeypatch, caplog):
    def broken_project(exc, phase):
        raise RuntimeError("projection exploded")

    monkeypatch.setattr(safeerr, "_project", broken_project)
    original = err.IntegrityError(ER.DUP_ENTRY, f"Duplicate entry '{SENSITIVE}'")
    with caplog.at_level(logging.WARNING, logger="pymysql.safeerr"):
        projected = safeerr.project_error(original, "execute")

    assert type(projected) is err.OperationalError
    assert projected.args == (CR.CR_UNKNOWN_ERROR, safeerr.GENERIC_MESSAGE)
    assert projected.error_token is None
    assert projected.retryable is False
    assert_no_leak(projected)


# ---------------------------------------------------------------------------
# Phase coverage via fake packet injection
# ---------------------------------------------------------------------------


def test_connect_phase_error_packet_is_projected(caplog):
    payload = error_payload(
        ER.ACCESS_DENIED_ERROR,
        f"Access denied for user '{SENSITIVE}'",
        sqlstate="28000",
    )
    conn = Connection(defer_connect=True, safe_errors=True)
    sock = FakeSocket(frame(payload, 0))

    with caplog.at_level(logging.WARNING, logger="pymysql.safeerr"):
        with pytest.raises(err.OperationalError) as excinfo:
            conn.connect(sock=sock)

    exc = excinfo.value
    assert exc.args[0] == ER.ACCESS_DENIED_ERROR
    assert exc.sqlstate == "28000"
    assert exc.phase == "connect"
    assert exc.retryable is False
    assert exc.__cause__ is None
    assert exc.__suppress_context__ is True
    assert_no_leak(exc)
    # Internal diagnostics still record the original server message.
    assert any(SENSITIVE in r.getMessage() for r in caplog.records)


def test_execute_phase_error_packet_is_projected(caplog):
    payload = error_payload(
        ER.DUP_ENTRY, f"Duplicate entry '{SENSITIVE}'", sqlstate="23000"
    )
    conn = make_connection(frame(payload, 1))

    with caplog.at_level(logging.WARNING, logger="pymysql.safeerr"):
        with pytest.raises(err.IntegrityError) as excinfo:
            conn.query("INSERT INTO users (email) VALUES ('x')")

    exc = excinfo.value
    assert exc.args[0] == ER.DUP_ENTRY
    assert exc.sqlstate == "23000"
    assert exc.phase == "execute"
    assert exc.retryable is False
    assert_no_leak(exc)
    assert any(SENSITIVE in r.getMessage() for r in caplog.records)


def test_streaming_fetch_phase_error_packet_is_projected(caplog):
    payload = error_payload(
        ER.LOCK_DEADLOCK, f"Deadlock on row '{SENSITIVE}'", sqlstate="40001"
    )
    conn = make_connection(frame(payload, 0))
    result = MySQLResult(conn)
    result.unbuffered_active = True

    with caplog.at_level(logging.WARNING, logger="pymysql.safeerr"):
        with pytest.raises(err.OperationalError) as excinfo:
            result._read_rowdata_packet_unbuffered()
    result.unbuffered_active = False

    exc = excinfo.value
    assert exc.args[0] == ER.LOCK_DEADLOCK
    assert exc.sqlstate == "40001"
    assert exc.phase == "fetch"
    assert exc.retryable is True
    assert_no_leak(exc)
    assert any(SENSITIVE in r.getMessage() for r in caplog.records)


def test_finish_unbuffered_query_error_is_projected():
    payload = error_payload(ER.DUP_ENTRY, f"late error '{SENSITIVE}'")
    conn = make_connection(frame(payload, 0))
    result = MySQLResult(conn)
    result.unbuffered_active = True

    with pytest.raises(err.IntegrityError) as excinfo:
        result._finish_unbuffered_query()
    result.unbuffered_active = False

    assert excinfo.value.phase == "fetch"
    assert_no_leak(excinfo.value)


# ---------------------------------------------------------------------------
# Responsibility boundary: default behavior stays MySQLdb-compatible
# ---------------------------------------------------------------------------


def test_default_connection_keeps_verbatim_server_message():
    message = f"Duplicate entry '{SENSITIVE}' for key 'users.email'"
    payload = error_payload(ER.DUP_ENTRY, message, sqlstate="23000")
    conn = make_connection(frame(payload, 1), safe_errors=False)

    with pytest.raises(err.IntegrityError) as excinfo:
        conn.query("INSERT INTO users (email) VALUES ('x')")

    exc = excinfo.value
    assert exc.args == (ER.DUP_ENTRY, message)
    assert SENSITIVE in str(exc)
    assert not safeerr.is_projected(exc)


def test_default_connect_keeps_verbatim_server_message():
    message = f"Access denied for user '{SENSITIVE}'"
    payload = error_payload(ER.ACCESS_DENIED_ERROR, message, sqlstate="28000")
    conn = Connection(defer_connect=True)  # safe_errors defaults to False

    with pytest.raises(err.OperationalError) as excinfo:
        conn.connect(sock=FakeSocket(frame(payload, 0)))

    assert SENSITIVE in str(excinfo.value)
