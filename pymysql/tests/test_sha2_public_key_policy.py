"""Unit tests for the SHA-2 server public key source policy.

All key material is generated at test time and used only by these tests.
Fake auth packets drive the authentication state machine without a server.
"""

import struct
from unittest import mock

import pytest

import pymysql
from pymysql import _auth
from pymysql.constants import CLIENT, CR
from pymysql.protocol import MysqlPacket

pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import padding, rsa  # noqa: E402

PASSWORD = b"test-only-password"
SALT = b"01234567890123456789"
OK_PACKET = b"\0\0\0\2\0\0\0"
USER = b"policy-user"


def make_keypair():
    # Small but well-formed RSA key, generated fresh for test use only.
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=1024)
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_key, public_pem


@pytest.fixture(scope="module")
def approved_key():
    return make_keypair()


@pytest.fixture(scope="module")
def server_key():
    return make_keypair()


def packet(data):
    return MysqlPacket(data, "utf8")


def make_connection(plugin="caching_sha2_password", **kwargs):
    conn = pymysql.connect(
        user=USER,
        password=PASSWORD,
        ssl_disabled=True,
        defer_connect=True,
        **kwargs,
    )
    conn.server_version = "8.0.0"
    conn.server_capabilities = CLIENT.PLUGIN_AUTH | CLIENT.SECURE_CONNECTION
    conn.client_flag = CLIENT.PLUGIN_AUTH | CLIENT.SECURE_CONNECTION
    conn.salt = SALT
    conn._auth_plugin_name = plugin
    return conn


def run_auth(conn, responses):
    packets = [packet(response) for response in responses]
    with (
        mock.patch.object(conn, "write_packet") as write_packet,
        mock.patch.object(conn, "_read_packet", side_effect=packets),
    ):
        conn._request_authentication()
    return [call.args[0] for call in write_packet.call_args_list]


def handshake_authresp(data):
    # data_init (32 bytes) + user + NUL, then 1-byte length + auth response.
    pos = 32 + len(USER) + 1
    (length,) = struct.unpack("B", data[pos : pos + 1])
    return data[pos + 1 : pos + 1 + length]


def decrypt_message(private_key, encrypted):
    return private_key.decrypt(
        encrypted,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA1()),
            algorithm=hashes.SHA1(),
            label=None,
        ),
    )


def expected_message():
    return _auth._xor_password(PASSWORD + b"\0", SALT)


class TestPinnedPolicy:
    def test_caching_sha2_full_auth_uses_configured_key(self, approved_key):
        private_key, public_pem = approved_key
        conn = make_connection(
            server_public_key=public_pem,
            server_public_key_policy="pinned",
        )
        writes = run_auth(conn, [b"\x01\x04", OK_PACKET])

        # Handshake (fast-path scramble) + encrypted password; no key request.
        assert len(writes) == 2
        assert writes[1] != b"\x02"
        assert decrypt_message(private_key, writes[1]) == expected_message()
        assert conn.server_public_key == public_pem

    def test_caching_sha2_without_key_fails_before_credentials(self):
        conn = make_connection(server_public_key_policy="pinned")
        packets = [packet(b"\x01\x04")]
        with (
            mock.patch.object(conn, "write_packet") as write_packet,
            mock.patch.object(conn, "_read_packet", side_effect=packets),
            pytest.raises(
                pymysql.err.OperationalError,
                match="server public key is not configured",
            ),
        ):
            conn._request_authentication()

        # Only the handshake (fast-path scramble) was written; neither the
        # key request nor any encrypted credential left the client.
        writes = [call.args[0] for call in write_packet.call_args_list]
        assert len(writes) == 1
        assert b"\x02" not in writes

    def test_sha256_auth_switch_uses_configured_key(self, approved_key):
        private_key, public_pem = approved_key
        conn = make_connection(
            server_public_key=public_pem,
            server_public_key_policy="pinned",
        )
        writes = run_auth(
            conn,
            [b"\xfesha256_password\0" + SALT + b"\0", OK_PACKET],
        )

        # Handshake + encrypted password; the key is never requested.
        assert len(writes) == 2
        assert b"\1" not in writes[1:]
        assert decrypt_message(private_key, writes[1]) == expected_message()
        assert conn.server_public_key == public_pem

    def test_sha256_rejects_key_pushed_by_server(self, approved_key, server_key):
        _, public_pem = approved_key
        _, server_pem = server_key
        conn = make_connection(
            plugin="sha256_password",
            server_public_key=public_pem,
            server_public_key_policy="pinned",
        )
        packets = [packet(b"\x01" + server_pem)]
        with (
            mock.patch.object(conn, "write_packet") as write_packet,
            mock.patch.object(conn, "_read_packet", side_effect=packets),
            pytest.raises(
                pymysql.err.OperationalError,
                match="refusing public key sent by server",
            ),
        ):
            conn._request_authentication()

        # The server-sent key must not replace the pre-approved one.
        assert conn.server_public_key == public_pem
        # Nothing was sent after the handshake response.
        assert len(write_packet.call_args_list) == 1

    def test_sha256_initial_handshake_uses_configured_key(self, approved_key):
        private_key, public_pem = approved_key
        conn = make_connection(
            plugin="sha256_password",
            server_public_key=public_pem,
            server_public_key_policy="pinned",
        )
        writes = run_auth(conn, [OK_PACKET])

        authresp = handshake_authresp(writes[0])
        assert authresp != b"\1"  # no key request
        assert decrypt_message(private_key, authresp) == expected_message()

    def test_sha256_initial_handshake_without_key_fails_before_write(self):
        conn = make_connection(
            plugin="sha256_password",
            server_public_key_policy="pinned",
        )
        with (
            mock.patch.object(conn, "write_packet") as write_packet,
            mock.patch.object(conn, "_read_packet"),
            pytest.raises(
                pymysql.err.OperationalError,
                match="server public key is not configured",
            ),
        ):
            conn._request_authentication()

        # Rejection happens before even the handshake response is sent.
        write_packet.assert_not_called()

    def test_pinned_key_survives_reconnect(self, approved_key):
        private_key, public_pem = approved_key
        for _ in range(2):
            conn = make_connection(
                server_public_key=public_pem,
                server_public_key_policy="pinned",
            )
            writes = run_auth(conn, [b"\x01\x04", OK_PACKET])
            assert decrypt_message(private_key, writes[1]) == expected_message()
            # The pinned key is never replaced across connections.
            assert conn.server_public_key == public_pem


class TestDynamicPolicy:
    def test_caching_sha2_retrieves_server_key(self, server_key):
        private_key, server_pem = server_key
        conn = make_connection()
        writes = run_auth(
            conn,
            [b"\x01\x04", b"\x01" + server_pem, OK_PACKET],
        )

        assert writes[1] == b"\x02"  # key request
        assert decrypt_message(private_key, writes[2]) == expected_message()
        assert conn.server_public_key == server_pem

    def test_sha256_retrieves_server_key(self, server_key):
        private_key, server_pem = server_key
        conn = make_connection(plugin="sha256_password")
        writes = run_auth(conn, [b"\x01" + server_pem, OK_PACKET])

        assert handshake_authresp(writes[0]) == b"\1"  # key request
        assert decrypt_message(private_key, writes[1]) == expected_message()
        assert conn.server_public_key == server_pem

    def test_dynamic_is_the_default_policy(self):
        conn = make_connection()
        assert conn.server_public_key_policy == "dynamic"


class TestPolicySafety:
    def test_invalid_policy_is_rejected(self):
        with pytest.raises(ValueError, match="server_public_key_policy"):
            pymysql.connect(defer_connect=True, server_public_key_policy="bogus")

    @pytest.mark.parametrize("plugin", ["caching_sha2_password", "sha256_password"])
    def test_error_exposes_only_reason_category(self, plugin, server_key):
        _, server_pem = server_key
        conn = make_connection(plugin=plugin, server_public_key_policy="pinned")
        packets = [packet(b"\x01\x04")]
        with (
            mock.patch.object(conn, "write_packet"),
            mock.patch.object(conn, "_read_packet", side_effect=packets),
            pytest.raises(pymysql.err.OperationalError) as exc_info,
        ):
            conn._request_authentication()

        exc = exc_info.value
        assert exc.args[0] == CR.CR_AUTH_PLUGIN_ERR
        message = str(exc)
        # Only the safe reason category is exposed.
        assert "public key" in message
        assert "pinned" in message
        # Never credentials, key material, or packet payloads.
        assert PASSWORD.decode() not in message
        assert "BEGIN" not in message
        assert server_pem.splitlines()[1].decode() not in message
        assert SALT.decode() not in message
