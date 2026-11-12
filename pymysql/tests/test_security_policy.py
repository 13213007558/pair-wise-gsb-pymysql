"""Unit tests for the credential-free connection security result and the
strict security policy, using a fake socket and scripted auth packets."""

import io
import ssl
import struct

import pytest

import pymysql
from pymysql import err
from pymysql.connections import SecurityInfo
from pymysql.constants import CLIENT

SALT = b"12345678901234567890"
OK_PAYLOAD = b"\x00\x00\x00\x00\x00\x00\x00"


def frame(seq, payload):
    return struct.pack("<I", len(payload))[:3] + bytes([seq]) + payload


def handshake_packet(capabilities, plugin=b"caching_sha2_password"):
    return (
        b"\x0a"  # protocol version 10
        b"8.0.36\x00"
        + struct.pack("<I", 1234)  # thread id
        + SALT[:8]
        + b"\x00"
        + struct.pack("<H", capabilities & 0xFFFF)
        + b"\x21"  # charset 33 (utf8_general_ci)
        + struct.pack("<H", 0)  # server status
        + struct.pack("<H", (capabilities >> 16) & 0xFFFF)
        + bytes([len(SALT) + 1])
        + b"\x00" * 10  # reserved
        + SALT[8:]
        + b"\x00"
        + plugin
        + b"\x00"
    )


PLAIN_CAPS = CLIENT.PLUGIN_AUTH | CLIENT.SECURE_CONNECTION
TLS_CAPS = PLAIN_CAPS | CLIENT.SSL


def auth_switch_packet(plugin):
    return b"\xfe" + plugin + b"\x00" + SALT + b"\x00"


class FakeSocket:
    """Minimal socket stand-in fed with scripted server packets."""

    def __init__(self, server_bytes=b"", sent=None):
        self._reader = io.BytesIO(server_bytes)
        self.sent = sent if sent is not None else []
        self.closed = False

    def settimeout(self, timeout):
        pass

    def setsockopt(self, *args):
        pass

    def makefile(self, mode):
        return self._reader

    def sendall(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


class FakeTLSSocket(FakeSocket):
    def version(self):
        return "TLSv1.3"

    def cipher(self):
        return ("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256)


class FakeSSLContext:
    def __init__(self, wrapped, check_hostname, verify_mode):
        self._wrapped = wrapped
        self.check_hostname = check_hostname
        self.verify_mode = verify_mode

    def wrap_socket(self, sock, server_hostname=None):
        return self._wrapped


def make_connection(server_script, **kwargs):
    """Build a deferred connection plus the fake socket it will use."""
    kwargs.setdefault("user", "scott")
    kwargs.setdefault("password", "tiger")
    kwargs["defer_connect"] = True
    conn = pymysql.connect(**kwargs)
    sent = []
    sock = FakeSocket(server_script, sent)
    return conn, sock, sent


def connect_plain(*, strict=False, switch=True):
    """Plain TCP handshake; server asks for an auth plugin switch."""
    script = frame(0, handshake_packet(PLAIN_CAPS))
    if switch:
        script += frame(2, auth_switch_packet(b"mysql_native_password"))
        script += frame(4, OK_PAYLOAD)
    else:
        script += frame(2, OK_PAYLOAD)
    script += frame(1, OK_PAYLOAD)  # SET NAMES
    conn, sock, sent = make_connection(
        script, ssl_disabled=True, strict_security=strict
    )
    conn.connect(sock=sock)
    return conn, sent


class TestSecurityInfo:
    def test_fields_are_credential_free(self):
        assert SecurityInfo._fields == (
            "transport",
            "encrypted",
            "server_identity_verified",
            "auth_plugin",
            "tls_version",
            "cipher_suite",
        )

    def test_plain_handshake_reports_unencrypted(self):
        conn, _ = connect_plain()
        info = conn.get_security_info()
        assert info.transport == "tcp"
        assert info.encrypted is False
        assert info.server_identity_verified is False
        assert info.tls_version is None
        assert info.cipher_suite is None
        conn.close()

    def test_auth_plugin_reflects_switch_not_handshake(self):
        # Server advertised caching_sha2_password but switched to
        # mysql_native_password: the result must show the final plugin.
        conn, _ = connect_plain(switch=True)
        assert conn.get_security_info().auth_plugin == "mysql_native_password"
        conn.close()

    def test_auth_plugin_without_switch(self):
        conn, _ = connect_plain(switch=False)
        assert conn.get_security_info().auth_plugin == "caching_sha2_password"
        conn.close()

    def test_compat_mode_allows_unencrypted_query(self):
        # Default (non-strict) mode: security state is reported but never
        # enforced, business queries flow over plaintext.
        script = frame(0, handshake_packet(PLAIN_CAPS)) + frame(2, OK_PAYLOAD)
        script += frame(1, OK_PAYLOAD)  # SET NAMES
        script += frame(1, OK_PAYLOAD)  # SELECT 1 result
        conn, sock, sent = make_connection(script, ssl_disabled=True)
        conn.connect(sock=sock)
        assert conn.get_security_info().encrypted is False
        conn.query("SELECT 1")
        assert any(b"SELECT 1" in chunk for chunk in sent)
        conn.close()

    def test_unix_socket_counts_as_secure_transport(self):
        script = frame(0, handshake_packet(PLAIN_CAPS)) + frame(2, OK_PAYLOAD)
        script += frame(1, OK_PAYLOAD)  # SET NAMES
        conn, sock, _ = make_connection(
            script,
            unix_socket="/tmp/mysql.sock",
            ssl_disabled=True,
            strict_security=True,
        )
        conn.connect(sock=sock)
        info = conn.get_security_info()
        assert info.transport == "unix"
        assert info.encrypted is False
        conn.close()


class TestStrictSecurityPolicy:
    def test_plain_tcp_is_rejected_and_queries_blocked(self):
        script = frame(0, handshake_packet(PLAIN_CAPS)) + frame(2, OK_PAYLOAD)
        conn, sock, sent = make_connection(
            script, ssl_disabled=True, strict_security=True
        )
        with pytest.raises(err.OperationalError, match="not encrypted"):
            conn.connect(sock=sock)

        # The failed policy check must not leave a security result behind.
        assert conn.get_security_info() is None
        # Business queries cannot be sent on the aborted connection.
        with pytest.raises(err.InterfaceError):
            conn.query("SELECT 1")
        assert not any(b"SELECT 1" in chunk for chunk in sent)
        assert not any(b"SET NAMES" in chunk for chunk in sent)

    def test_tls_without_identity_verification_is_rejected(self):
        conn, sock, sent = make_connection(
            frame(0, handshake_packet(TLS_CAPS)), strict_security=True
        )
        tls_sock = FakeTLSSocket(frame(3, OK_PAYLOAD), sent)
        conn.ctx = FakeSSLContext(tls_sock, False, ssl.CERT_NONE)
        with pytest.raises(err.OperationalError, match="identity was not verified"):
            conn.connect(sock=sock)
        assert conn.get_security_info() is None
        with pytest.raises(err.InterfaceError):
            conn.query("SELECT 1")

    def test_tls_with_verified_identity_passes_and_queries_flow(self):
        conn, sock, sent = make_connection(
            frame(0, handshake_packet(TLS_CAPS)), strict_security=True
        )
        tls_script = frame(3, OK_PAYLOAD)  # auth OK
        tls_script += frame(1, OK_PAYLOAD)  # SET NAMES
        tls_script += frame(1, OK_PAYLOAD)  # SELECT 1 result
        tls_sock = FakeTLSSocket(tls_script, sent)
        conn.ctx = FakeSSLContext(tls_sock, True, ssl.CERT_REQUIRED)
        conn.connect(sock=sock)

        info = conn.get_security_info()
        assert info.transport == "tcp+tls"
        assert info.encrypted is True
        assert info.server_identity_verified is True
        assert info.auth_plugin == "caching_sha2_password"
        assert info.tls_version == "TLSv1.3"
        assert info.cipher_suite == "TLS_AES_256_GCM_SHA384"

        conn.query("SELECT 1")
        assert any(b"SELECT 1" in chunk for chunk in sent)
        conn.close()

    def test_unverified_tls_is_allowed_in_compat_mode(self):
        conn, sock, sent = make_connection(frame(0, handshake_packet(TLS_CAPS)))
        tls_script = frame(3, OK_PAYLOAD)  # auth OK
        tls_script += frame(1, OK_PAYLOAD)  # SET NAMES
        tls_sock = FakeTLSSocket(tls_script, sent)
        conn.ctx = FakeSSLContext(tls_sock, False, ssl.CERT_NONE)
        conn.connect(sock=sock)

        info = conn.get_security_info()
        assert info.encrypted is True
        assert info.server_identity_verified is False
        conn.close()


class TestReconnect:
    def test_failed_reconnect_clears_previous_security_result(self):
        conn, _ = connect_plain()
        assert conn.get_security_info() is not None

        # Reconnect: server sends the handshake then drops during auth.
        script = frame(0, handshake_packet(PLAIN_CAPS))
        with pytest.raises(err.OperationalError):
            conn.connect(sock=FakeSocket(script))

        assert conn.get_security_info() is None
        assert conn._secure is False
        with pytest.raises(err.InterfaceError):
            conn.query("SELECT 1")

    def test_failed_handshake_read_clears_previous_security_result(self):
        conn, _ = connect_plain()
        assert conn.get_security_info() is not None

        # Reconnect: server drops before sending anything.
        with pytest.raises(err.OperationalError):
            conn.connect(sock=FakeSocket(b""))

        assert conn.get_security_info() is None
        with pytest.raises(err.InterfaceError):
            conn.query("SELECT 1")

    def test_successful_reconnect_replaces_security_result(self):
        conn, _ = connect_plain(switch=True)
        assert conn.get_security_info().auth_plugin == "mysql_native_password"

        script = frame(0, handshake_packet(PLAIN_CAPS)) + frame(2, OK_PAYLOAD)
        script += frame(1, OK_PAYLOAD)  # SET NAMES
        conn.connect(sock=FakeSocket(script))
        info = conn.get_security_info()
        assert info is not None
        assert info.auth_plugin == "caching_sha2_password"
        conn.close()
