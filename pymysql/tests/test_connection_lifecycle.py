"""Lifecycle/state-machine tests for :class:`pymysql.connections.Connection`.

These tests run without a MySQL server. The TCP layer is replaced by an
in-memory fake socket and every handshake/initialization packet is mocked, so
the failure paths that are impossible to trigger deterministically against a
real server (TLS wrap failure, authentication plugin error and post-auth
initialization failure) can be asserted precisely.
"""

import ssl
import struct
from unittest import mock

import pytest

import pymysql
from pymysql.connections import ConnectionState
from pymysql.constants import CLIENT
from pymysql.protocol import MysqlPacket

OK_PACKET = b"\x00\x00\x00\x02\x00\x00\x00"
SALT = b"01234567890123456789"
CAPABILITIES = (
    CLIENT.PLUGIN_AUTH
    | CLIENT.SECURE_CONNECTION
    | CLIENT.PLUGIN_AUTH_LENENC_CLIENT_DATA
)


def ok_packet():
    return MysqlPacket(OK_PACKET, "utf8")


class FakeSocket:
    """Minimal socket stand-in that records close()/FD release."""

    def __init__(self):
        self._closed = False
        self.options = []
        self.timeout = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def setsockopt(self, level, opt, value):
        self.options.append((level, opt, value))

    def makefile(self, mode="rb"):
        return mock.MagicMock(name="rfile")

    def sendall(self, data):
        if self._closed:
            raise OSError("socket is closed")

    def close(self):
        self._closed = True

    def fileno(self):
        return -1 if self._closed else 7


def handshake_packet(
    thread_id=1234, plugin=b"mysql_native_password", capabilities=CAPABILITIES
):
    salt_part_1 = SALT[:8]
    salt_part_2 = SALT[8:]
    data = bytes([10])
    data += b"8.0.36\x00"
    data += struct.pack("<I", thread_id)
    data += salt_part_1
    data += b"\x00"
    data += struct.pack("<H", capabilities & 0xFFFF)
    data += bytes([45])
    data += struct.pack("<H", 2)
    data += struct.pack("<H", capabilities >> 16)
    data += bytes([21])
    data += b"\x00" * 10
    data += salt_part_2 + b"\x00"
    data += plugin + b"\x00"
    return MysqlPacket(data, "utf8")


def make_connection(**kwargs):
    kwargs.setdefault("user", "router-user")
    kwargs.setdefault("password", "router-password")
    kwargs.setdefault("ssl_disabled", True)
    kwargs.setdefault("defer_connect", True)
    return pymysql.connect(**kwargs)


def patch_transport(conn, sock, read_packets=()):
    """Replace socket creation and packet I/O for a deferred connection."""
    packets = list(read_packets)

    def _read_packet(packet_type=MysqlPacket):
        if not packets:
            raise AssertionError("no more mocked packets available")
        return packets.pop(0)

    return (
        mock.patch.object(
            pymysql.connections.socket,
            "create_connection",
            return_value=sock,
        ),
        mock.patch.object(conn, "_read_packet", side_effect=_read_packet),
        mock.patch.object(conn, "write_packet"),
    )


def test_deferred_connection_is_disconnected_and_unusable():
    conn = make_connection()

    assert conn._state is ConnectionState.DISCONNECTED
    assert conn.open is False
    assert conn._sock is None

    with pytest.raises(pymysql.err.InterfaceError):
        conn.query("SELECT 1")
    with pytest.raises(pymysql.err.Error):
        conn.ping(False)


def test_successful_connect_reaches_connected_state_and_serves_queries():
    conn = make_connection()
    sock = FakeSocket()
    create_conn, read_packet, write_packet = patch_transport(
        conn, sock, [handshake_packet(), ok_packet()]
    )

    with create_conn, read_packet, write_packet, mock.patch.object(
        conn, "set_character_set"
    ), mock.patch.object(conn, "autocommit"):
        conn.connect()

    try:
        assert conn._state is ConnectionState.CONNECTED
        assert conn.open is True
        assert conn._sock is sock
        assert conn.server_thread_id == (1234,)
        assert conn._auth_plugin_name == "mysql_native_password"

        # A query goes through the command path only while CONNECTED.
        with mock.patch.object(conn, "_execute_command") as execute, mock.patch.object(
            conn, "_read_query_result", return_value=1
        ):
            assert conn.query("SELECT 1") == 1
            execute.assert_called_once()
    finally:
        conn.close()

    assert conn._state is ConnectionState.CLOSED
    assert sock._closed is True


def test_tls_wrap_failure_releases_socket_and_marks_connection_failed():
    conn = make_connection()
    conn.ssl = True
    conn._ssl_required = True
    conn.ctx = mock.Mock(name="ssl-context")
    conn.client_flag |= CLIENT.SSL
    assert conn._ssl_required is True
    sock = FakeSocket()
    rfile = mock.MagicMock(name="plain-rfile")

    create_conn, read_packet, write_packet = patch_transport(
        conn, sock, [handshake_packet(capabilities=CAPABILITIES | CLIENT.SSL)]
    )

    with create_conn, read_packet, write_packet, mock.patch.object(
        sock, "makefile", return_value=rfile
    ), mock.patch.object(
        conn.ctx,
        "wrap_socket",
        side_effect=ssl.SSLError("handshake failure"),
    ) as wrap_socket:
        # SSLError is an OSError, so connect() wraps it like any other
        # transport-level failure while still releasing the resources.
        with pytest.raises(pymysql.err.OperationalError):
            conn.connect()
        wrap_socket.assert_called_once()
        assert conn._state is ConnectionState.FAILED

    assert conn._state is ConnectionState.FAILED
    assert conn.open is False
    assert conn._sock is None
    assert conn._rfile is None
    assert conn._secure is False
    assert rfile.close.called
    assert sock._closed is True

    # The half-authenticated object must not become usable through ping.
    with pytest.raises(pymysql.err.Error):
        conn.ping(False)
    with pytest.raises(pymysql.err.InterfaceError):
        conn.query("SELECT 1")


def test_tls_wrap_success_replaces_plain_reader_and_completes_handshake():
    conn = make_connection()
    conn.ssl = True
    conn._ssl_required = True
    conn.ctx = mock.Mock(name="ssl-context")
    conn.client_flag |= CLIENT.SSL

    sock = FakeSocket()
    plain_rfile = mock.MagicMock(name="plain-rfile")
    tls_sock = FakeSocket()
    tls_rfile = mock.MagicMock(name="tls-rfile")
    conn.ctx.wrap_socket.return_value = tls_sock
    sock.makefile = mock.Mock(return_value=plain_rfile)
    tls_sock.makefile = mock.Mock(return_value=tls_rfile)

    create_conn, read_packet, write_packet = patch_transport(
        conn,
        sock,
        [handshake_packet(capabilities=CAPABILITIES | CLIENT.SSL), ok_packet()],
    )

    with create_conn, read_packet, write_packet, mock.patch.object(
        conn, "set_character_set"
    ), mock.patch.object(conn, "autocommit"):
        conn.connect()

    try:
        assert conn._state is ConnectionState.CONNECTED
        assert conn._secure is True
        assert conn._sock is tls_sock
        assert conn._rfile is tls_rfile
        # The plaintext reader created before the upgrade must be released.
        assert plain_rfile.close.called
        conn.ctx.wrap_socket.assert_called_once()
    finally:
        conn.close()

    assert tls_sock._closed is True


def test_authentication_plugin_error_releases_socket_and_marks_failed():
    conn = make_connection()
    sock = FakeSocket()
    plugin_error = pymysql.err.OperationalError(
        2059, "Authentication plugin cannot be loaded"
    )

    create_conn, read_packet, write_packet = patch_transport(
        conn, sock, [handshake_packet()]
    )

    with create_conn, read_packet, write_packet, mock.patch.object(
        conn, "_request_authentication", side_effect=plugin_error
    ):
        with pytest.raises(pymysql.err.OperationalError, match="cannot be loaded"):
            conn.connect()

    assert conn._state is ConnectionState.FAILED
    assert conn.open is False
    assert conn._sock is None
    assert conn._rfile is None
    assert sock._closed is True

    with pytest.raises(pymysql.err.Error):
        conn.ping(False)


def test_initialization_sql_failure_after_auth_marks_connection_failed():
    conn = make_connection()
    sock = FakeSocket()

    create_conn, read_packet, write_packet = patch_transport(
        conn, sock, [handshake_packet(), ok_packet()]
    )
    init_error = pymysql.err.OperationalError(1064, "syntax error in SET NAMES")

    with create_conn, read_packet, write_packet, mock.patch.object(
        conn, "_request_authentication"
    ), mock.patch.object(conn, "set_character_set", side_effect=init_error):
        with pytest.raises(pymysql.err.OperationalError, match="syntax error"):
            conn.connect()

    # Authentication succeeded but initialization did not: never CONNECTED.
    assert conn._state is ConnectionState.FAILED
    assert conn.open is False
    assert conn._sock is None
    assert conn._rfile is None
    assert sock._closed is True

    # A single ping must not be able to mask the unfinished initialization.
    with pytest.raises(pymysql.err.Error):
        conn.ping(False)
    with pytest.raises(pymysql.err.InterfaceError):
        conn.query("SELECT 1")


def test_ping_reconnect_rebuilds_full_identity_state():
    conn = make_connection()
    first_sock = FakeSocket()
    first_packets = [
        handshake_packet(thread_id=111),
        ok_packet(),
    ]
    create_conn, read_packet, write_packet = patch_transport(
        conn, first_sock, first_packets
    )

    with create_conn, read_packet, write_packet, mock.patch.object(
        conn, "set_character_set"
    ), mock.patch.object(conn, "autocommit"):
        conn.connect()

    assert conn._state is ConnectionState.CONNECTED
    assert conn.server_thread_id == (111,)
    assert conn.salt == SALT
    assert conn._auth_plugin_name == "mysql_native_password"

    # Simulate the server going away: socket-level failure invalidates it.
    conn._force_close()
    assert conn._state is ConnectionState.FAILED
    assert conn.open is False
    assert first_sock._closed is True

    second_sock = FakeSocket()
    second_packets = [
        handshake_packet(thread_id=222, plugin=b"caching_sha2_password"),
        ok_packet(),
        ok_packet(),
    ]

    def create_second(*args, **kwargs):
        # Runs after _reset_session_state() but before the new handshake, so
        # the old identity (thread 111) must already have been discarded.
        assert conn.server_thread_id == (0,)
        assert conn.salt == b""
        assert conn._auth_plugin_name == ""
        assert conn._secure is False
        assert conn._state is ConnectionState.CONNECTING
        return second_sock

    with mock.patch.object(
        pymysql.connections.socket,
        "create_connection",
        side_effect=create_second,
    ), mock.patch.object(
        conn, "_read_packet", side_effect=list(second_packets)
    ), mock.patch.object(conn, "write_packet"), mock.patch.object(
        conn, "set_character_set"
    ), mock.patch.object(conn, "autocommit"):
        with pytest.warns(DeprecationWarning):
            conn.ping(reconnect=True)

    assert conn._state is ConnectionState.CONNECTED
    assert conn.open is True
    assert conn._sock is second_sock
    # The new handshake identity fully replaces the stale one.
    assert conn.server_thread_id == (222,)
    assert conn._auth_plugin_name == "caching_sha2_password"
    assert conn.salt == SALT

    conn.close()
    assert second_sock._closed is True


def test_reconnect_after_failure_resets_negotiated_identity():
    conn = make_connection()
    sock = FakeSocket()

    create_conn, read_packet, write_packet = patch_transport(
        conn, sock, [handshake_packet(thread_id=333)]
    )
    with create_conn, read_packet, write_packet, mock.patch.object(
        conn,
        "_request_authentication",
        side_effect=pymysql.err.OperationalError(1045, "Access denied"),
    ):
        with pytest.raises(pymysql.err.OperationalError):
            conn.connect()

    assert conn._state is ConnectionState.FAILED
    # The failed handshake leaves no negotiated identity behind.
    assert conn.server_thread_id == (0,)
    assert conn.salt == b""
    assert conn._auth_plugin_name == ""
    assert conn._secure is False

    # A fresh connect goes through the whole handshake again from DISCONNECTED.
    fresh_sock = FakeSocket()
    fresh_packets = [
        handshake_packet(thread_id=444),
        ok_packet(),
    ]
    with mock.patch.object(
        pymysql.connections.socket,
        "create_connection",
        return_value=fresh_sock,
    ), mock.patch.object(
        conn, "_read_packet", side_effect=fresh_packets
    ), mock.patch.object(conn, "write_packet"), mock.patch.object(
        conn, "set_character_set"
    ), mock.patch.object(conn, "autocommit"):
        conn.connect()

    assert conn._state is ConnectionState.CONNECTED
    assert conn.server_thread_id == (444,)
    assert conn._sock is fresh_sock
    conn.close()


def test_connect_is_not_reentrant_while_handshake_in_progress():
    conn = make_connection()
    conn._state = ConnectionState.AUTHENTICATING
    with pytest.raises(pymysql.err.Error, match="already being established"):
        conn.connect()


def test_close_on_failed_connection_keeps_failed_state_and_is_idempotent():
    conn = make_connection()
    sock = FakeSocket()
    create_conn, read_packet, write_packet = patch_transport(
        conn, sock, [handshake_packet()]
    )
    with create_conn, read_packet, write_packet, mock.patch.object(
        conn,
        "_request_authentication",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError):
            conn.connect()

    assert conn._state is ConnectionState.FAILED
    conn.close()
    assert conn._state is ConnectionState.FAILED
    assert conn._closed is True
    with pytest.raises(pymysql.err.Error, match="Already closed"):
        conn.close()
