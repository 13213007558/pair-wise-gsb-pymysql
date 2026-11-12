"""Contract tests for custom authentication plugin return values.

The scripted socket is a fake server: it replays exact protocol packets and
records the packets the driver sends.  Server packet sequence numbers are
validated by ``Connection._read_packet`` itself, so a passed test also proves
the post-authentication command/response stream remained in sync.
"""

import struct

import pytest

import pymysql
from pymysql import err
from pymysql.constants import CLIENT
from pymysql.protocol import MysqlPacket

SALT = b"01234567890123456789"
CAPABILITIES = CLIENT.SECURE_CONNECTION | CLIENT.PLUGIN_AUTH | CLIENT.CONNECT_ATTRS
OK_PAYLOAD = b"\0\0\0\x02\0\0\0"
AUTH_SWITCH = b"\xfefake\0challenge\0"


def packet(payload, seq):
    return struct.pack("<I", len(payload))[:3] + bytes([seq]) + payload


def ok_packet(seq):
    return packet(OK_PAYLOAD, seq)


def handshake_packet():
    salt_len = len(SALT) + 1
    payload = (
        b"\x0a"
        + b"8.0.0\0"
        + struct.pack("<I", 42)
        + SALT[:8]
        + b"\0"
        + struct.pack("<H", CAPABILITIES & 0xFFFF)
        + bytes([33])
        + struct.pack("<H", 2)
        + struct.pack("<H", CAPABILITIES >> 16)
        + bytes([salt_len])
        + b"\0" * 10
        + SALT[8:]
        + b"\0"
        + b"fake\0"
    )
    return packet(payload, 0)


class ScriptedSocket:
    """Minimal socket double that replays packets and records writes."""

    def __init__(self, packets):
        self._data = bytearray(b"".join(packets))
        self.writes = []
        self.closed = False

    def settimeout(self, timeout):
        pass

    def makefile(self, mode):
        return self

    def read(self, size):
        if len(self._data) < size:
            raise AssertionError(
                f"server script exhausted: wanted {size}, have {len(self._data)}"
            )
        result = bytes(self._data[:size])
        del self._data[:size]
        return result

    def sendall(self, data):
        length = int.from_bytes(data[:3], "little")
        self.writes.append((data[3], bytes(data[4 : 4 + length])))

    def close(self):
        self.closed = True

    @property
    def remaining(self):
        return bytes(self._data)


class ReadOKThenReturn:
    def __init__(self, conn):
        self.conn = conn

    def authenticate(self, packet):
        self.conn.write_packet(b"response")
        return self.conn._read_packet()


class ReadOKThenReturnNone:
    def __init__(self, conn):
        self.conn = conn

    def authenticate(self, packet):
        self.conn.write_packet(b"response")
        self.conn._read_packet()
        return None


class ReturnNoneImmediately:
    def __init__(self, conn):
        pass

    def authenticate(self, packet):
        return None


class ReturnInputPacket:
    def __init__(self, conn):
        pass

    def authenticate(self, packet):
        return packet


class ReturnUnreadOK:
    def __init__(self, conn):
        pass

    def authenticate(self, packet):
        return MysqlPacket(OK_PAYLOAD, "utf8")


class ReturnStaleTransitionPacket:
    def __init__(self, conn):
        self.conn = conn
        self.saved = None

    def authenticate(self, packet):
        if packet.is_extra_auth_data():
            self.conn.write_packet(b"response2")
            self.conn._read_packet()  # consumes the terminal OK
            return self.saved
        self.conn.write_packet(b"response1")
        self.saved = packet
        return self.conn._read_packet()


def make_connection(handler):
    return pymysql.connect(
        user="plugin-user",
        password="plugin-password",
        ssl_disabled=True,
        autocommit=None,
        defer_connect=True,
        auth_plugin_map={b"fake": handler},
    )


def connect_with(packets, handler):
    sock = ScriptedSocket(packets)
    conn = make_connection(handler)
    conn.connect(sock=sock)
    return conn, sock


def assert_query_still_in_sync(conn, sock):
    """Exercise a query and verify protocol ordering, not just no exception."""
    affected = conn.query("DO 1")
    assert affected == 0
    assert sock.remaining == b""
    assert [seq for seq, _ in sock.writes] == [1, 3, 0, 0]
    assert sock.writes[1][1] == b"response"
    assert sock.writes[2][1] == b"\x03SET NAMES utf8mb4"
    assert sock.writes[3][1] == b"\x03DO 1"
    assert conn.open
    assert conn.server_status == 2
    assert conn._next_seq_id == 2


def test_plugin_returns_final_ok_then_query_is_in_sync():
    packets = [
        handshake_packet(),
        packet(AUTH_SWITCH, 2),
        ok_packet(4),
        ok_packet(1),
        ok_packet(1),
    ]
    conn, sock = connect_with(packets, ReadOKThenReturn)
    assert [seq for seq, _ in sock.writes] == [1, 3, 0]
    assert sock.writes[1][1] == b"response"
    assert sock.writes[2][1] == b"\x03SET NAMES utf8mb4"
    assert_query_still_in_sync(conn, sock)


def test_plugin_reads_ok_and_returns_none_then_query_is_in_sync():
    packets = [
        handshake_packet(),
        packet(AUTH_SWITCH, 2),
        ok_packet(4),
        ok_packet(1),
        ok_packet(1),
    ]
    conn, sock = connect_with(packets, ReadOKThenReturnNone)
    assert [seq for seq, _ in sock.writes] == [1, 3, 0]
    assert sock.writes[1][1] == b"response"
    assert sock.writes[2][1] == b"\x03SET NAMES utf8mb4"
    assert_query_still_in_sync(conn, sock)


def test_none_without_reading_terminal_ok_is_rejected():
    sock = ScriptedSocket([handshake_packet(), packet(AUTH_SWITCH, 2), ok_packet(4)])
    conn = make_connection(ReturnNoneImmediately)

    with pytest.raises(err.OperationalError, match="returned None"):
        conn.connect(sock=sock)

    assert not conn.open
    assert sock.closed
    assert sock.remaining == ok_packet(4)
    assert [seq for seq, _ in sock.writes] == [1]


def test_returning_the_input_packet_is_rejected():
    sock = ScriptedSocket([handshake_packet(), packet(AUTH_SWITCH, 2), ok_packet(4)])
    conn = make_connection(ReturnInputPacket)

    with pytest.raises(err.OperationalError, match="returned the packet it was given"):
        conn.connect(sock=sock)

    assert not conn.open
    assert sock.closed
    assert sock.remaining == ok_packet(4)
    assert [seq for seq, _ in sock.writes] == [1]


def test_returning_an_ok_that_was_not_read_is_rejected():
    sock = ScriptedSocket([handshake_packet(), packet(AUTH_SWITCH, 2), ok_packet(4)])
    conn = make_connection(ReturnUnreadOK)

    with pytest.raises(err.OperationalError, match="not the most recent packet"):
        conn.connect(sock=sock)

    assert not conn.open
    assert sock.closed
    assert sock.remaining == ok_packet(4)
    assert [seq for seq, _ in sock.writes] == [1]


def test_stale_transition_packet_after_ok_is_rejected_without_rereading():
    packets = [
        handshake_packet(),
        packet(AUTH_SWITCH, 2),
        packet(b"\x01follow-up", 4),
        ok_packet(6),
    ]
    sock = ScriptedSocket(packets)
    conn = make_connection(ReturnStaleTransitionPacket)

    with pytest.raises(err.OperationalError, match="not the most recent packet"):
        conn.connect(sock=sock)

    assert not conn.open
    assert sock.closed
    assert sock.remaining == b""
    assert [seq for seq, _ in sock.writes] == [1, 3, 5]
    assert sock.writes[1][1] == b"response1"
    assert sock.writes[2][1] == b"response2"
