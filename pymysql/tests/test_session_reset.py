"""Fake-protocol tests for :meth:`Connection.reset_session`.

These tests never connect to a real MySQL server.  Every packet the driver
reads after the handshake comes from a pre-built byte stream, so failure paths
(unread unbuffered rows, a server without COM_RESET_CONNECTION, a broken
baseline initialization) are deterministic.
"""

import struct

import pytest

import pymysql
from pymysql.constants import COMMAND, ER
from pymysql.protocol import MysqlPacket

OK_PACKET = b"\x00\x00\x00\x02\x00\x00\x00"
OK_NO_AUTOCOMMIT = b"\x00\x00\x00\x00\x00\x00\x00"
EOF_PACKET = b"\xfe\x00\x00\x02\x00"
ROW_PACKET = b"\x05hello"


def ok_packet(data=OK_PACKET):
    return MysqlPacket(data, "utf8")


def default_baseline_packets():
    """Packet reads performed by _restore_session_baseline for a default conn.

    A default connection re-sends SET NAMES and, because COM_RESET_CONNECTION
    leaves autocommit on, SET AUTOCOMMIT=0.
    """
    return [ok_packet(), ok_packet(OK_NO_AUTOCOMMIT)]


def eof_packet():
    return MysqlPacket(EOF_PACKET, "utf8")


def error_packet(errno, message=b"unsupported"):
    return MysqlPacket(b"\xff" + struct.pack("<H", errno) + b"#HY000" + message, "utf8")


def field_packet(name=b"c1"):
    return MysqlPacket(
        b"\x03def"
        + b"\x03db\x00"
        + b"\x03tbl\x00"
        + b"\x03org\x00"
        + name
        + b"\x00"
        + b"\x0c"
        + struct.pack("<xHIBHBxx", 33, 10, 3, 0, 0),
        "utf8",
    )


def frame(payload, seq):
    return len(payload).to_bytes(3, "little") + bytes([seq]) + payload


class FramedReader:
    """Serve real wire-framed packets so sequence numbers are checked.

    Connection._read_packet validates the packet sequence number and closes
    the connection on a mismatch.  Using framed data therefore proves that
    draining left the stream at the exact point the reset command expects.
    """

    def __init__(self, frames):
        self.buffer = bytearray(b"".join(frames))
        self.closed = False

    def read(self, size):
        if len(self.buffer) < size:
            raise AssertionError("framed stream exhausted")
        data = bytes(self.buffer[:size])
        del self.buffer[:size]
        return data

    def close(self):
        self.closed = True


def framed_socket(frames):
    sock = FakeSocket()
    sock._reader = FramedReader(frames)
    sock.makefile = lambda mode: sock._reader
    return sock


def make_framed_connection(frames):
    """Connection whose post-handshake stream comes from framed packets."""
    conn = pymysql.connect(
        user="pool-user", password="fixture", ssl_disabled=True, defer_connect=True
    )
    sock = framed_socket(frames)
    conn._sock = sock
    conn._rfile = sock._reader
    conn._closed = False
    conn._current_timeout = None
    conn.server_status = 0
    conn._next_seq_id = 1
    return conn


def make_connection(**kwargs):
    """Build a Connection whose handshake and initialization are faked."""
    kwargs.setdefault("user", "pool-user")
    kwargs.setdefault("password", "fixture-password")
    conn = pymysql.connect(ssl_disabled=True, defer_connect=True, **kwargs)
    conn._sock = FakeSocket()
    conn._rfile = conn._sock.makefile("rb")
    conn._closed = False
    conn._current_timeout = None
    conn.server_status = 0
    return conn


class FakeSocket:
    """Socket stand-in that only needs to record writes and close."""

    def __init__(self):
        self.sent = []
        self.closed = False

    def settimeout(self, timeout):
        pass

    def makefile(self, mode):
        return self

    def sendall(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


class FakeStream:
    """A sequence of packets served through Connection._read_packet."""

    def __init__(self, packets):
        self.packets = list(packets)
        self.reads = 0

    def read(self, packet_type=MysqlPacket):
        self.reads += 1
        if not self.packets:
            raise AssertionError("fake stream exhausted")
        packet = self.packets.pop(0)
        if packet.is_error_packet():
            packet.raise_for_error()
        if packet_type is not MysqlPacket:
            packet = packet_type(packet.get_all_data(), "utf8")
        return packet


@pytest.fixture
def reset_recording(monkeypatch):
    sent = []
    original = pymysql.connections.Connection._execute_command

    def recording(self, command, sql):
        sent.append((command, sql))
        return original(self, command, sql)

    monkeypatch.setattr(pymysql.connections.Connection, "_execute_command", recording)
    return sent


def test_reset_drains_incomplete_unbuffered_stream_before_reset(
    reset_recording, monkeypatch
):
    conn = make_connection()
    cursor = conn.cursor(pymysql.cursors.SSCursor)
    monkeypatch.setattr(
        conn,
        "_read_packet",
        FakeStream(
            [
                MysqlPacket(b"\x01", "utf8"),  # 1 column
                field_packet(),
                eof_packet(),  # end of metadata
                MysqlPacket(ROW_PACKET, "utf8"),  # first row
            ]
        ).read,
    )
    cursor.execute("SELECT c1 FROM t")
    assert cursor.fetchone() == ("hello",)
    assert conn._result.unbuffered_active is True

    # The application left the rest of the stream unread. reset_session must
    # consume the remaining rows and the EOF before sending the reset command.
    packets = [
        MysqlPacket(b"\x03two", "utf8"),
        eof_packet(),
        ok_packet(),  # COM_RESET_CONNECTION acknowledged
        *default_baseline_packets(),
    ]
    stream = FakeStream(packets)
    monkeypatch.setattr(conn, "_read_packet", stream.read)

    result = conn.reset_session()

    assert result.reusable is True
    assert result.drained_results is True
    assert result.reset_supported is True
    assert stream.reads == 5
    commands = [command for command, _ in reset_recording]
    assert COMMAND.COM_RESET_CONNECTION in commands
    # The reset must come after the stream was fully drained, never before.
    reset_index = commands.index(COMMAND.COM_RESET_CONNECTION)
    assert reset_index == 1
    assert reset_recording[0][0] == COMMAND.COM_QUERY  # the SELECT


def test_reset_drains_all_following_result_sets(monkeypatch):
    conn = make_connection()
    monkeypatch.setattr(
        conn,
        "_read_packet",
        FakeStream(
            [
                MysqlPacket(b"\x01", "utf8"),
                field_packet(),
                eof_packet(),
                MysqlPacket(ROW_PACKET, "utf8"),
            ]
        ).read,
    )
    cursor = conn.cursor(pymysql.cursors.SSCursor)
    cursor.execute("SELECT 1; SELECT 2")
    assert cursor.fetchone() == ("hello",)

    # EOF of the first set flags SERVER_MORE_RESULTS_EXISTS, then the second
    # unbuffered result set: field count, metadata, EOF, one row, EOF, then the
    # OK packet for the reset command itself.
    packets = [
        MysqlPacket(b"\xfe\x00\x00\x0a\x00", "utf8"),
        MysqlPacket(b"\x01", "utf8"),
        field_packet(),
        eof_packet(),
        MysqlPacket(ROW_PACKET, "utf8"),
        eof_packet(),
        ok_packet(),  # COM_RESET_CONNECTION acknowledged
        *default_baseline_packets(),
    ]
    stream = FakeStream(packets)
    monkeypatch.setattr(conn, "_read_packet", stream.read)

    result = conn.reset_session()
    assert result.reusable is True
    assert result.drained_results is True
    # Every trailing result set was consumed, not just the first EOF.
    assert not stream.packets


def test_reset_success_restores_declared_baseline(reset_recording, monkeypatch):
    conn = make_connection(
        database="testdb",
        charset="utf8mb4",
        sql_mode="STRICT_TRANS_TABLES",
        init_command="SET @pool_ready = 1",
        autocommit=True,
    )
    # COM_RESET_CONNECTION and then the baseline: INIT_DB, SET NAMES,
    # SET sql_mode and init_command. The reset OK packet reports autocommit on,
    # matching the declared baseline, so no extra SET AUTOCOMMIT is needed.
    monkeypatch.setattr(
        conn,
        "_read_packet",
        FakeStream(
            [
                ok_packet(),  # COM_RESET_CONNECTION
                ok_packet(),  # select_db
                ok_packet(),  # SET NAMES
                ok_packet(),  # SET sql_mode
                ok_packet(),  # init_command
            ]
        ).read,
    )

    result = conn.reset_session()
    assert result.reusable is True
    assert result.baseline == {
        "database": "testdb",
        "charset": "utf8mb4",
        "collation": None,
        "sql_mode": "STRICT_TRANS_TABLES",
        "init_command": "SET @pool_ready = 1",
        "autocommit": True,
    }
    commands = [command for command, _ in reset_recording]
    assert commands.count(COMMAND.COM_INIT_DB) == 1
    assert commands.count(COMMAND.COM_RESET_CONNECTION) == 1
    sent_sql = [
        sql.decode("utf8") if isinstance(sql, bytes) else sql
        for command, sql in reset_recording
        if command == COMMAND.COM_QUERY
    ]
    assert "SET NAMES utf8mb4" in sent_sql
    assert "SET @pool_ready = 1" in sent_sql
    assert "SET AUTOCOMMIT = 1" not in sent_sql


def test_reset_unsupported_server_marks_connection_unusable(
    reset_recording, monkeypatch
):
    conn = make_connection()
    stream = FakeStream([error_packet(ER.UNKNOWN_COM_ERROR, b"Unknown command")])
    monkeypatch.setattr(conn, "_read_packet", stream.read)

    result = conn.reset_session()

    assert result.reusable is False
    assert result.reset_supported is False
    assert isinstance(result.error, pymysql.err.OperationalError)
    assert result.error.args[0] == ER.UNKNOWN_COM_ERROR
    # A server without COM_RESET_CONNECTION may not be pooled: the session
    # still contains the previous task's transaction, variables and settings.
    assert conn._sock is None
    assert conn._rfile is None
    assert conn.open is False
    with pytest.raises(pymysql.err.InterfaceError):
        conn.query("SELECT 1")


def test_reset_baseline_initialization_failure_marks_connection_unusable(
    reset_recording, monkeypatch
):
    conn = make_connection(init_command="SET @pool_ready = 1")
    # Reset is acknowledged, SET NAMES succeeds, then init_command fails
    # because the server rejects the statement.
    stream = FakeStream(
        [
            ok_packet(),  # COM_RESET_CONNECTION
            ok_packet(OK_NO_AUTOCOMMIT),  # SET NAMES
            error_packet(ER.PARSE_ERROR, b"init failed"),
        ]
    )
    monkeypatch.setattr(conn, "_read_packet", stream.read)

    result = conn.reset_session()

    assert result.reusable is False
    # The server did support the reset; the failure happened afterwards.
    assert result.reset_supported is True
    assert isinstance(result.error, pymysql.err.ProgrammingError)
    assert conn._sock is None
    assert conn.open is False


def test_unfinishable_stream_marks_connection_unusable(monkeypatch):
    conn = make_connection()
    monkeypatch.setattr(
        conn,
        "_read_packet",
        FakeStream(
            [
                MysqlPacket(b"\x01", "utf8"),
                field_packet(),
                eof_packet(),
                MysqlPacket(ROW_PACKET, "utf8"),
            ]
        ).read,
    )
    cursor = conn.cursor(pymysql.cursors.SSCursor)
    cursor.execute("SELECT c1 FROM t")
    assert cursor.fetchone() == ("hello",)

    # The server never sends the rest of the rows.  Draining must fail and
    # leave the connection unusable instead of sending a reset command whose
    # OK packet would be confused with result data.
    def broken_read(packet_type=MysqlPacket):
        raise pymysql.err.OperationalError(2013, "Lost connection")

    monkeypatch.setattr(conn, "_read_packet", broken_read)

    result = conn.reset_session()

    assert result.reusable is False
    assert result.drained_results is False
    assert isinstance(result.error, pymysql.err.OperationalError)
    assert conn._sock is None
    assert conn.open is False


def test_reset_on_closed_connection_never_reports_reusable():
    conn = make_connection()
    conn._force_close()

    result = conn.reset_session()

    assert result.reusable is False
    assert isinstance(result.error, pymysql.err.InterfaceError)


def test_reset_does_not_issue_rollback_and_relies_on_server_reset(
    reset_recording, monkeypatch
):
    conn = make_connection()
    monkeypatch.setattr(
        conn,
        "_read_packet",
        FakeStream([ok_packet(), *default_baseline_packets()]).read,
    )
    # A rollback-only implementation would never send this command.
    result = conn.reset_session()
    assert result.reusable is True
    commands = [command for command, _ in reset_recording]
    assert commands[0] == COMMAND.COM_RESET_CONNECTION
    assert all(command == COMMAND.COM_QUERY for command in commands[1:])
    sent_sql = [sql for command, sql in reset_recording]
    assert not any("ROLLBACK" in sql for sql in sent_sql if isinstance(sql, str))


def test_framed_stream_sequence_stays_aligned_across_drain_and_reset(monkeypatch):
    """A real wire stream would close the connection on a stale sequence."""
    conn = make_framed_connection(
        [
            frame(b"\x01", 1),  # one-column result set header
            frame(field_packet().get_all_data(), 2),
            frame(EOF_PACKET, 3),
            frame(ROW_PACKET, 4),  # only the first row is read by the app
            frame(b"\x03two", 5),  # drained row
            frame(EOF_PACKET, 6),  # drained EOF
            frame(OK_PACKET, 1),  # COM_RESET_CONNECTION OK
            frame(OK_PACKET, 1),  # SET NAMES
            frame(OK_NO_AUTOCOMMIT, 1),  # SET AUTOCOMMIT = 0
        ]
    )
    cursor = conn.cursor(pymysql.cursors.SSCursor)
    cursor.execute("SELECT c1 FROM t")
    assert cursor.fetchone() == ("hello",)
    assert conn._result.unbuffered_active is True

    result = conn.reset_session()

    assert result.reusable is True
    assert result.drained_results is True
    assert conn.open is True
    assert conn._next_seq_id == 2  # no packet was skipped or double-read


def test_framed_stream_unread_result_is_not_mistaken_for_reset_ok(monkeypatch):
    """Without draining, the row packet would be read as the reset response."""
    conn = make_framed_connection(
        [
            frame(b"\x01", 1),
            frame(field_packet().get_all_data(), 2),
            frame(EOF_PACKET, 3),
            frame(ROW_PACKET, 4),
            # No drain and no reset packets: the fake stream ends here.
        ]
    )
    cursor = conn.cursor(pymysql.cursors.SSCursor)
    cursor.execute("SELECT c1 FROM t")
    assert cursor.fetchone() == ("hello",)

    result = conn.reset_session()

    assert result.reusable is False
    assert result.error is not None
    assert conn.open is False
