"""Unit tests for handshake/authentication phase byte and roundtrip budgets.

These tests drive Connection.connect() against short synthetic protocol
streams (no server needed) and verify that the configurable budgets cover
every read of the connection phase: the handshake, fragmented packets and
reads triggered by custom authentication plugins. Query traffic after
authentication is governed by separate rules and is not limited.
"""

import io
import struct

import pytest

import pymysql
from pymysql import connections
from pymysql.constants import CLIENT, FIELD_TYPE

SALT = b"01234567890123456789"
OK_PAYLOAD = b"\x00\x00\x00\x02\x00\x00\x00"
SERVER_CAPABILITIES = CLIENT.PROTOCOL_41 | CLIENT.SECURE_CONNECTION | CLIENT.PLUGIN_AUTH


def frame(payload, seq):
    return connections._pack_int24(len(payload)) + bytes([seq]) + payload


def handshake_payload(plugin_name=b"mysql_native_password"):
    return (
        b"\x0a"  # protocol version
        b"8.0.0\x00"  # server version
        + struct.pack("<I", 1234)  # thread id
        + SALT[:8]
        + b"\x00"  # filler
        + struct.pack("<H", SERVER_CAPABILITIES & 0xFFFF)
        + b"\x21"  # charset
        + b"\x00\x00"  # server status (autocommit off)
        + struct.pack("<H", SERVER_CAPABILITIES >> 16)
        + bytes([len(SALT) + 1])  # auth_plugin_data_len
        + b"\x00" * 10  # reserved
        + SALT[8:]
        + b"\x00"  # salt terminator
        + plugin_name
        + b"\x00"
    )


def field_descriptor_payload(name):
    return (
        b"\x03def"  # catalog
        b"\x00"  # schema
        b"\x00"  # table
        b"\x00"  # org_table
        + bytes([len(name)])
        + name
        + b"\x00"  # org_name
        + b"\x0c"  # length of fixed-size fields
        + struct.pack("<H", 63)  # charsetnr: binary
        + struct.pack("<I", 11)  # column length
        + bytes([FIELD_TYPE.LONG])
        + b"\x00\x00"  # flags
        + b"\x00"  # decimals
        + b"\x00\x00"  # filler
    )


EOF_PAYLOAD = b"\xfe\x00\x00\x00\x00"


def fragment(payload, size):
    """Split a payload into MAX_PACKET_LEN-style continuation frames."""
    chunks = [payload[i : i + size] for i in range(0, len(payload), size)]
    if len(payload) % size == 0:
        chunks.append(b"")  # exact multiple is terminated by an empty packet
    return chunks


class FakeSocket:
    """Minimal socket stand-in feeding a synthetic server byte stream."""

    def __init__(self, data):
        self._rfile = io.BytesIO(data)
        self.written = bytearray()
        self.write_calls = 0
        self.closed = False
        self.timeout = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def makefile(self, mode):
        return self._rfile

    def sendall(self, data):
        self.write_calls += 1
        self.written += data

    def close(self):
        self.closed = True


def connect(stream, **kwargs):
    conn = pymysql.connect(
        user="budget-user",
        password=b"secret",
        ssl_disabled=True,
        defer_connect=True,
        **kwargs,
    )
    sock = FakeSocket(stream)
    conn.connect(sock=sock)
    return conn, sock


def simple_auth_stream(plugin_name=b"mysql_native_password"):
    handshake = handshake_payload(plugin_name)
    return handshake, frame(handshake, 0) + frame(OK_PAYLOAD, 2) + frame(OK_PAYLOAD, 1)


class TestAuthPhaseBudgets:
    def test_exact_byte_budget_allows_connect(self):
        handshake, stream = simple_auth_stream()
        auth_bytes = (4 + len(handshake)) + (4 + len(OK_PAYLOAD))

        conn, sock = connect(stream, auth_max_bytes=auth_bytes)

        assert conn.open
        assert not sock.closed
        # The SET NAMES OK packet is query-phase traffic and is not counted.
        assert conn._auth_bytes_read == auth_bytes
        assert conn._auth_roundtrips == 2
        conn.close()

    def test_byte_budget_exceeded_aborts_connection(self):
        handshake, stream = simple_auth_stream()
        auth_bytes = (4 + len(handshake)) + (4 + len(OK_PAYLOAD))

        with pytest.raises(pymysql.err.OperationalError, match="exceeded byte budget"):
            connect(stream, auth_max_bytes=auth_bytes - 1)

    def test_byte_budget_exceeded_releases_connection(self):
        handshake, stream = simple_auth_stream()
        auth_bytes = (4 + len(handshake)) + (4 + len(OK_PAYLOAD))
        conn = pymysql.connect(
            user="budget-user",
            password=b"secret",
            ssl_disabled=True,
            defer_connect=True,
            auth_max_bytes=auth_bytes - 1,
        )
        sock = FakeSocket(stream)

        with pytest.raises(pymysql.err.OperationalError):
            conn.connect(sock=sock)

        # The actual read amount is recorded and the connection is released:
        # it must not be returned to a pool.
        assert conn._auth_bytes_read == auth_bytes
        assert sock.closed
        assert not conn.open
        assert conn._rfile is None
        with pytest.raises(pymysql.err.Error, match="Already closed"):
            conn.ping()

    def test_exact_roundtrip_budget_allows_connect(self):
        _, stream = simple_auth_stream()

        conn, sock = connect(stream, auth_max_roundtrips=2)

        assert conn.open
        assert conn._auth_roundtrips == 2
        conn.close()

    def test_roundtrip_budget_exceeded_aborts_before_reading_more(self):
        _, stream = simple_auth_stream()
        conn = pymysql.connect(
            user="budget-user",
            password=b"secret",
            ssl_disabled=True,
            defer_connect=True,
            auth_max_roundtrips=1,
        )
        sock = FakeSocket(stream)

        with pytest.raises(
            pymysql.err.OperationalError, match="exceeded roundtrip budget"
        ):
            conn.connect(sock=sock)

        assert conn._auth_roundtrips == 2
        assert sock.closed
        assert not conn.open
        # Only the initial handshake response was written; the connection
        # did not continue the exchange after the budget was exceeded.
        assert sock.write_calls == 1

    def test_credentials_not_sent_after_byte_budget_exceeded(self):
        handshake = handshake_payload()
        switch = b"\xfemysql_native_password\x00" + SALT + b"\x00"
        stream = frame(handshake, 0) + frame(switch, 2) + frame(OK_PAYLOAD, 3)
        # Budget is exceeded while reading the auth switch request, before
        # the client would answer it with scrambled credentials.
        budget = (4 + len(handshake)) + (4 + len(switch)) - 1
        conn = pymysql.connect(
            user="budget-user",
            password=b"secret",
            ssl_disabled=True,
            defer_connect=True,
            auth_max_bytes=budget,
        )
        sock = FakeSocket(stream)

        with pytest.raises(pymysql.err.OperationalError, match="exceeded byte budget"):
            conn.connect(sock=sock)

        assert sock.closed
        assert not conn.open
        # Only the initial handshake response was sent; no further
        # credentials (the auth switch answer) left the client.
        assert sock.write_calls == 1

    def test_multi_step_auth_within_budget_passes(self):
        # caching_sha2_password fast path: extra auth data + OK.
        handshake = handshake_payload(b"caching_sha2_password")
        stream = (
            frame(handshake, 0)
            + frame(b"\x01\x03", 2)  # fast auth succeeded
            + frame(OK_PAYLOAD, 3)
            + frame(OK_PAYLOAD, 1)  # SET NAMES
        )
        auth_bytes = (4 + len(handshake)) + (4 + 2) + (4 + len(OK_PAYLOAD))

        conn, sock = connect(stream, auth_max_bytes=auth_bytes, auth_max_roundtrips=3)

        assert conn.open
        assert conn._auth_bytes_read == auth_bytes
        assert conn._auth_roundtrips == 3
        conn.close()

    def test_auth_switch_within_budget_passes(self):
        handshake = handshake_payload(b"caching_sha2_password")
        switch = b"\xfemysql_native_password\x00" + SALT + b"\x00"
        stream = (
            frame(handshake, 0)
            + frame(switch, 2)
            + frame(OK_PAYLOAD, 4)  # client answers the switch with seq 3
            + frame(OK_PAYLOAD, 1)  # SET NAMES
        )

        conn, sock = connect(stream, auth_max_roundtrips=3)

        assert conn.open
        assert conn._auth_roundtrips == 3
        conn.close()

    def test_fragmented_packets_count_toward_byte_budget(self, monkeypatch):
        monkeypatch.setattr(connections, "MAX_PACKET_LEN", 8)
        handshake = handshake_payload()
        chunks = fragment(handshake, 8)
        assert len(chunks) > 1  # handshake is genuinely fragmented
        stream = b"".join(frame(chunk, i) for i, chunk in enumerate(chunks))
        # The client's handshake response consumes one sequence id. With the
        # shrunken MAX_PACKET_LEN the SET NAMES command is written as 3
        # packets, so its OK packet arrives with sequence id 3.
        stream += frame(OK_PAYLOAD, len(chunks) + 1) + frame(OK_PAYLOAD, 3)
        # Every fragment, headers included, is accounted for.
        auth_bytes = len(handshake) + 4 * len(chunks) + (4 + len(OK_PAYLOAD))

        conn, sock = connect(stream, auth_max_bytes=auth_bytes)

        assert conn.open
        assert conn._auth_bytes_read == auth_bytes
        conn.close()

    def test_fragmented_packets_over_byte_budget_abort(self, monkeypatch):
        monkeypatch.setattr(connections, "MAX_PACKET_LEN", 8)
        handshake = handshake_payload()
        chunks = fragment(handshake, 8)
        stream = b"".join(frame(chunk, i) for i, chunk in enumerate(chunks))
        stream += frame(OK_PAYLOAD, len(chunks) + 1) + frame(OK_PAYLOAD, 3)
        auth_bytes = len(handshake) + 4 * len(chunks) + (4 + len(OK_PAYLOAD))
        conn = pymysql.connect(
            user="budget-user",
            password=b"secret",
            ssl_disabled=True,
            defer_connect=True,
            auth_max_bytes=auth_bytes - 1,
        )
        sock = FakeSocket(stream)

        with pytest.raises(pymysql.err.OperationalError, match="exceeded byte budget"):
            conn.connect(sock=sock)

        assert conn._auth_bytes_read == auth_bytes
        assert sock.closed
        assert not conn.open


class ChainHandler:
    """Custom plugin that keeps the exchange alive by reading follow-ups."""

    def __init__(self, conn):
        self.conn = conn

    def authenticate(self, packet):
        return self.conn._read_packet()


def chain_stream(challenges):
    handshake = handshake_payload(b"chain")
    packets = [frame(handshake, 0)]
    for i in range(challenges):
        packets.append(frame(b"\x01challenge-%d" % i, 2 + i))
    packets.append(frame(OK_PAYLOAD, 2 + challenges))
    packets.append(frame(OK_PAYLOAD, 1))  # SET NAMES
    return handshake, b"".join(packets)


class TestCustomPluginBudgets:
    def test_plugin_chain_within_roundtrip_budget_passes(self):
        _, stream = chain_stream(3)

        conn, sock = connect(
            stream,
            auth_plugin_map={"chain": ChainHandler},
            auth_max_roundtrips=5,
        )

        # handshake + 3 plugin-triggered reads + final OK
        assert conn.open
        assert conn._auth_roundtrips == 5
        conn.close()

    def test_continuous_plugin_switching_is_cut_by_budget(self):
        # The plugin keeps the exchange alive far longer than any legitimate
        # handshake; the roundtrip budget must terminate it.
        _, stream = chain_stream(50)
        conn = pymysql.connect(
            user="budget-user",
            password=b"secret",
            ssl_disabled=True,
            defer_connect=True,
            auth_plugin_map={"chain": ChainHandler},
            auth_max_roundtrips=10,
        )
        sock = FakeSocket(stream)

        with pytest.raises(
            pymysql.err.OperationalError, match="exceeded roundtrip budget"
        ):
            conn.connect(sock=sock)

        assert conn._auth_roundtrips == 11
        assert sock.closed
        assert not conn.open

    def test_continuous_plugin_switching_is_cut_by_byte_budget(self):
        handshake, _ = chain_stream(0)
        challenge = b"\x01challenge"
        stream = frame(handshake, 0) + b"".join(
            frame(challenge, 2 + i) for i in range(50)
        )
        # Allow the handshake plus two plugin-triggered reads.
        budget = (4 + len(handshake)) + 2 * (4 + len(challenge))
        conn = pymysql.connect(
            user="budget-user",
            password=b"secret",
            ssl_disabled=True,
            defer_connect=True,
            auth_plugin_map={"chain": ChainHandler},
            auth_max_bytes=budget,
        )
        sock = FakeSocket(stream)

        with pytest.raises(pymysql.err.OperationalError, match="exceeded byte budget"):
            conn.connect(sock=sock)

        # Reads are accounted per socket read: the 4-byte header of the
        # next challenge packet is what pushes the total over the budget.
        assert conn._auth_bytes_read == budget + 4
        assert sock.closed
        assert not conn.open


class TestQueryPhaseIndependentRules:
    def test_query_traffic_is_not_limited_by_auth_budget(self):
        handshake, stream = simple_auth_stream()
        auth_bytes = (4 + len(handshake)) + (4 + len(OK_PAYLOAD))
        result_set = (
            frame(b"\x01", 1)  # column count
            + frame(field_descriptor_payload(b"1"), 2)
            + frame(EOF_PAYLOAD, 3)
            + frame(b"\x011", 4)  # row: "1"
            + frame(EOF_PAYLOAD, 5)
        )
        conn, sock = connect(
            stream + result_set,
            auth_max_bytes=auth_bytes,
            auth_max_roundtrips=2,
        )

        # The result set far exceeds the auth budgets but is query-phase
        # traffic governed by separate rules.
        assert conn.query("SELECT 1") == 1
        assert conn._result.rows == ((1,),)
        assert conn._auth_bytes_read == auth_bytes
        assert conn._auth_roundtrips == 2
        conn.close()


class TestBudgetValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"auth_max_bytes": 0},
            {"auth_max_roundtrips": 0},
            {"auth_max_roundtrips": -1},
        ],
    )
    def test_non_positive_budgets_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            pymysql.connect(
                user="budget-user", ssl_disabled=True, defer_connect=True, **kwargs
            )
