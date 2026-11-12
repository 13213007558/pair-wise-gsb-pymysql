"""Offline protocol tests: parameter escaping must track connection state.

These tests never touch a real server. A scripted fake server feeds
handshake/OK/EOF packets to the driver and records the exact bytes the
driver writes back, so each test can assert two things at once:

1. the final SQL bytes emitted for a parameterized query, and
2. where the state behind those bytes came from (the handshake status
   flags, the OK packet of a ``SET sql_mode`` statement, the terminating
   OK packet of each result in a multi-result response, or the new
   handshake after a reconnect).

Only plain ``str`` values are used as query data.

Guarantee boundary: the driver can only keep escaping semantics correct
for values passed through ``cursor.execute(sql, args)`` (i.e. through
``Connection.escape``). SQL text concatenated by the application itself
is sent verbatim -- the driver never sees the value and cannot escape
it. Note that ``Connection.escape`` is documented as "Non-standard, for
internal use; do not use this in your applications", so applications
building SQL by hand get no escaping guarantee at all.
"""

import struct
from collections import deque
from unittest import mock

import pytest

from pymysql.connections import Connection, MySQLResult
from pymysql.constants import CLIENT, COMMAND, FIELD_TYPE, SERVER_STATUS
from pymysql.protocol import MysqlPacket

AUTOCOMMIT = SERVER_STATUS.SERVER_STATUS_AUTOCOMMIT
NO_BACKSLASH = SERVER_STATUS.SERVER_STATUS_NO_BACKSLASH_ESCAPES
MORE_RESULTS = SERVER_STATUS.SERVER_MORE_RESULTS_EXISTS

# One string exercising every character the two escaping modes treat
# differently: quote, backslash, newline, double quote, carriage return,
# NUL and Ctrl-Z.
VALUE = 'it\'s a \\ test\nwith "quotes"\r\x00and\x1actrl'
# converters.escape_string(): backslash-escapes every special character.
ESCAPED_BACKSLASH_MODE = 'it\\\'s a \\\\ test\\nwith \\"quotes\\"\\r\\0and\\Zctrl'
# NO_BACKSLASH_ESCAPES: only the single quote is escaped, by doubling.
ESCAPED_NO_BACKSLASH_MODE = "it''s a \\ test\nwith \"quotes\"\r\x00and\x1actrl"


def ok_payload(status=AUTOCOMMIT, more_results=False):
    if more_results:
        status |= MORE_RESULTS
    return b"\x00\x00\x00" + struct.pack("<HH", status, 0)


def eof_payload(status=AUTOCOMMIT):
    return b"\xfe" + struct.pack("<HH", 0, status)


def _lenenc_str(value):
    return bytes([len(value)]) + value


def field_descriptor_payload(name=b"v"):
    strings = b"".join(_lenenc_str(s) for s in (b"def", b"", b"", b"", name, b""))
    fixed = struct.pack("<HIBHB", 45, 255, FIELD_TYPE.VAR_STRING, 0, 0)
    return strings + b"\x0c" + fixed + b"\x00\x00"


def resultset_payloads(rows, status=AUTOCOMMIT):
    payloads = [b"\x01", field_descriptor_payload(), eof_payload(status)]
    payloads += [_lenenc_str(value.encode("utf-8")) for (value,) in rows]
    payloads.append(eof_payload(status))
    return payloads


def handshake_payload(status):
    caps = CLIENT.PROTOCOL_41 | CLIENT.SECURE_CONNECTION | CLIENT.PLUGIN_AUTH
    salt = b"0123456789abcdef0123"
    return (
        b"\x0a"  # protocol version 10
        b"8.0.36-offline\x00"
        + struct.pack("<I", 1)  # thread id
        + salt[:8]
        + b"\x00"
        + struct.pack("<H", caps & 0xFFFF)
        + b"\x2d"  # server charset id 45 (utf8mb4_general_ci)
        + struct.pack("<H", status)
        + struct.pack("<H", caps >> 16)
        + b"\x15"  # auth_plugin_data_len
        + b"\x00" * 10
        + salt[8:]
        + b"\x00"
        + b"mysql_native_password\x00"
    )


class ScriptedServer:
    """Offline server double: queues response packets, captures writes."""

    def __init__(self, conn):
        self.conn = conn
        self.responses = deque()
        self.written = []

    def attach(self):
        self.conn._sock = mock.Mock()
        self.conn._read_packet = self._read_packet
        self.conn._write_bytes = self._write_bytes
        return self

    def queue(self, *payloads):
        self.responses.extend(payloads)

    def _read_packet(self, packet_type=MysqlPacket):
        packet = packet_type(self.responses.popleft(), self.conn.encoding)
        if packet.is_error_packet():
            packet.raise_for_error()
        return packet

    def _write_bytes(self, data):
        self.written.append(bytes(data))

    def sent_queries(self):
        """SQL bytes of every COM_QUERY command sent so far."""
        return [data[5:] for data in self.written if data[4] == COMMAND.COM_QUERY]

    def last_query(self):
        return self.sent_queries()[-1]


def make_connection(connection_class=Connection, server_status=AUTOCOMMIT, **kwargs):
    kwargs.setdefault("charset", "utf8mb4")
    conn = connection_class(
        user="offline-user",
        password="offline-password",
        ssl_disabled=True,
        defer_connect=True,
        **kwargs,
    )
    # Stands in for the status flags learned from the handshake packet;
    # the reconnect test below goes through a real (scripted) handshake.
    conn.server_status = server_status
    server = ScriptedServer(conn).attach()
    return conn, server


def expected_insert(escaped_value):
    return b"INSERT INTO t VALUES ('" + escaped_value.encode("ascii") + b"')"


def test_escape_uses_backslash_mode_from_handshake_status():
    conn, server = make_connection(server_status=AUTOCOMMIT)
    server.queue(ok_payload(AUTOCOMMIT))

    conn.cursor().execute("INSERT INTO t VALUES (%s)", (VALUE,))

    assert server.last_query() == expected_insert(ESCAPED_BACKSLASH_MODE)
    # State source: the handshake status flags, nothing else has run.
    assert conn.server_status == AUTOCOMMIT
    assert not conn.server_status & NO_BACKSLASH


def test_sql_mode_switch_to_no_backslash_escapes():
    conn, server = make_connection(server_status=AUTOCOMMIT)
    cur = conn.cursor()

    server.queue(ok_payload(AUTOCOMMIT | NO_BACKSLASH))
    cur.execute("SET sql_mode='NO_BACKSLASH_ESCAPES'")
    # State source: the OK packet terminating the SET statement.
    assert conn.server_status & NO_BACKSLASH

    server.queue(ok_payload(AUTOCOMMIT | NO_BACKSLASH))
    cur.execute("INSERT INTO t VALUES (%s)", (VALUE,))

    assert server.last_query() == expected_insert(ESCAPED_NO_BACKSLASH_MODE)


def test_sql_mode_switch_back_to_backslash_escapes():
    conn, server = make_connection(server_status=AUTOCOMMIT | NO_BACKSLASH)
    cur = conn.cursor()

    server.queue(ok_payload(AUTOCOMMIT | NO_BACKSLASH))
    cur.execute("INSERT INTO t VALUES (%s)", (VALUE,))
    assert server.last_query() == expected_insert(ESCAPED_NO_BACKSLASH_MODE)

    server.queue(ok_payload(AUTOCOMMIT))
    cur.execute("SET sql_mode=DEFAULT")
    # State source: the OK packet of the second SET statement.
    assert not conn.server_status & NO_BACKSLASH

    server.queue(ok_payload(AUTOCOMMIT))
    cur.execute("INSERT INTO t VALUES (%s)", (VALUE,))
    assert server.last_query() == expected_insert(ESCAPED_BACKSLASH_MODE)


def test_multi_result_status_chain_updates_escaping_per_result():
    conn, server = make_connection(server_status=AUTOCOMMIT)
    cur = conn.cursor()

    server.queue(
        ok_payload(AUTOCOMMIT | NO_BACKSLASH, more_results=True),
        ok_payload(AUTOCOMMIT),
    )
    cur.execute("SET sql_mode='NO_BACKSLASH_ESCAPES'; SET sql_mode=DEFAULT")
    # State source: OK packet of the first result in the chain.
    assert conn.server_status & NO_BACKSLASH

    assert cur.nextset()
    # State source: OK packet terminating the second result.
    assert not conn.server_status & NO_BACKSLASH
    assert cur.nextset() is None

    server.queue(ok_payload(AUTOCOMMIT))
    cur.execute("INSERT INTO t VALUES (%s)", (VALUE,))
    assert server.last_query() == expected_insert(ESCAPED_BACKSLASH_MODE)


def test_select_result_eof_status_is_not_a_state_source():
    """EOF packets of result sets do not update ``Connection.server_status``.

    ``MySQLResult._check_packet_is_eof`` reads the EOF status flags only
    for ``has_next``; the escaping-relevant flags keep coming from the
    last handshake/OK packet. This pins that behavior so a future change
    of state source is a conscious decision.
    """
    conn, server = make_connection(server_status=AUTOCOMMIT | NO_BACKSLASH)
    cur = conn.cursor()

    # The EOF terminating this result set claims the flag is gone...
    server.queue(*resultset_payloads([("row-value",)], status=AUTOCOMMIT))
    cur.execute("SELECT v FROM t")
    assert cur.fetchall() == (("row-value",),)
    # ...but the driver still derives escaping from the last OK packet.
    assert conn.server_status & NO_BACKSLASH

    server.queue(ok_payload(AUTOCOMMIT | NO_BACKSLASH))
    cur.execute("INSERT INTO t VALUES (%s)", (VALUE,))
    assert server.last_query() == expected_insert(ESCAPED_NO_BACKSLASH_MODE)


def test_reconnect_resets_escaping_state_from_new_handshake():
    # Session 1: the server session is in NO_BACKSLASH_ESCAPES mode.
    conn, server = make_connection(
        server_status=AUTOCOMMIT | NO_BACKSLASH, autocommit=True
    )
    server.queue(ok_payload(AUTOCOMMIT | NO_BACKSLASH))
    conn.cursor().execute("INSERT INTO t VALUES (%s)", (VALUE,))
    assert server.last_query() == expected_insert(ESCAPED_NO_BACKSLASH_MODE)

    # Reconnect: fresh handshake without the flag, auth OK, SET NAMES OK.
    server.queue(
        handshake_payload(AUTOCOMMIT),
        ok_payload(AUTOCOMMIT),
        ok_payload(AUTOCOMMIT),
    )
    conn.connect(sock=mock.Mock())

    # State source: the new handshake status flags. The session sql_mode
    # did not survive the reconnect, and the driver re-negotiates the
    # charset with SET NAMES.
    assert conn.server_status == AUTOCOMMIT
    assert not conn.server_status & NO_BACKSLASH
    assert b"SET NAMES utf8mb4" in server.sent_queries()

    server.queue(ok_payload(AUTOCOMMIT))
    conn.cursor().execute("INSERT INTO t VALUES (%s)", (VALUE,))
    assert server.last_query() == expected_insert(ESCAPED_BACKSLASH_MODE)


@pytest.mark.parametrize(
    ("charset", "expected_bytes"),
    [
        ("utf8mb4", "héllo".encode("utf-8")),
        ("latin1", "héllo".encode("cp1252")),
    ],
)
def test_charset_determines_emitted_query_bytes(charset, expected_bytes):
    conn, server = make_connection(charset=charset)
    server.queue(ok_payload(AUTOCOMMIT))

    conn.cursor().execute("SELECT %s", ("héllo",))

    assert server.last_query() == b"SELECT '" + expected_bytes + b"'"
    # State source: the connection charset negotiated for this session.
    assert "héllo".encode(conn.encoding) == expected_bytes


def test_driver_cannot_escape_app_concatenated_sql():
    """Values interpolated by the application bypass parameter escaping.

    The driver sends whatever SQL text it is given; the escaping
    guarantee only covers values passed as query arguments.
    """
    conn, server = make_connection()
    server.queue(ok_payload(AUTOCOMMIT))

    conn.cursor().execute("INSERT INTO t VALUES ('%s')" % VALUE)

    assert server.last_query() == ("INSERT INTO t VALUES ('" + VALUE + "')").encode(
        "ascii"
    )


class StatusDroppingConnection(Connection):
    """Mutant driver: omits the server_status update after each query."""

    def _read_query_result(self, unbuffered=False):
        self._result = None
        result = MySQLResult(self)
        result.read()
        self._result = result
        # Deliberately missing:
        #   if result.server_status is not None:
        #       self.server_status = result.server_status
        return result.affected_rows


def _mode_switch_then_insert(conn, server):
    cur = conn.cursor()
    server.queue(ok_payload(AUTOCOMMIT | NO_BACKSLASH))
    cur.execute("SET sql_mode='NO_BACKSLASH_ESCAPES'")
    server.queue(ok_payload(AUTOCOMMIT | NO_BACKSLASH))
    cur.execute("INSERT INTO t VALUES (%s)", (VALUE,))
    return server.last_query()


def test_missing_status_update_is_detected():
    """A driver variant that drops one status update emits wrong SQL.

    The healthy connection escapes for NO_BACKSLASH_ESCAPES after the
    SET; the mutant keeps stale state and emits backslash escapes the
    server would not interpret. The divergence is what the tests above
    catch.
    """
    healthy_conn, healthy_server = make_connection()
    mutant_conn, mutant_server = make_connection(
        connection_class=StatusDroppingConnection
    )

    healthy_sql = _mode_switch_then_insert(healthy_conn, healthy_server)
    mutant_sql = _mode_switch_then_insert(mutant_conn, mutant_server)

    assert healthy_sql == expected_insert(ESCAPED_NO_BACKSLASH_MODE)
    assert mutant_sql == expected_insert(ESCAPED_BACKSLASH_MODE)
    assert mutant_sql != healthy_sql
    # The mutant never learned the mode change from the OK packet.
    assert not mutant_conn.server_status & NO_BACKSLASH
