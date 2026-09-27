"""Unit tests for the zlib compressed protocol (CLIENT.COMPRESS).

These tests use a fake socket to exercise the compressed framing layer
without a MySQL server: compressed frames and inner MySQL packets don't
align (a frame may hold several packets, a packet may span frames), and
each layer has its own sequence id.
"""

import struct
import zlib

import pytest

import pymysql
from pymysql import err
from pymysql.connections import MAX_PACKET_LEN, _lenenc_int
from pymysql.constants import CLIENT, COMMAND, FIELD_TYPE, SERVER_STATUS
from pymysql.cursors import SSCursor

SERVER_CAPS = (
    CLIENT.PROTOCOL_41 | CLIENT.SECURE_CONNECTION | CLIENT.PLUGIN_AUTH | CLIENT.COMPRESS
)
AUTOCOMMIT = SERVER_STATUS.SERVER_STATUS_AUTOCOMMIT
MORE_RESULTS = SERVER_STATUS.SERVER_MORE_RESULTS_EXISTS


def mysql_packet(payload, seq_id):
    """Build one MySQL packet (3-byte length, 1-byte sequence id)."""
    return struct.pack("<I", len(payload))[:3] + bytes([seq_id]) + payload


def compressed_frame(data, seq_id, compress=True):
    """Wrap raw MySQL packet bytes into one compressed protocol frame."""
    if compress:
        payload = zlib.compress(data)
        if len(payload) < len(data):
            return (
                struct.pack("<I", len(payload))[:3]
                + bytes([seq_id])
                + struct.pack("<I", len(data))[:3]
                + payload
            )
    return struct.pack("<I", len(data))[:3] + bytes([seq_id]) + b"\x00\x00\x00" + data


def frames_for(data, first_seq, chunk_size=None, compress=True):
    """Split a MySQL packet stream into compressed frames."""
    if chunk_size is None:
        chunk_size = max(1, len(data))
    frames = b""
    for i in range(0, len(data), chunk_size):
        frames += compressed_frame(
            data[i : i + chunk_size], (first_seq + i // chunk_size) % 256, compress
        )
    return frames


def handshake_payload(capabilities):
    salt = b"01234567890123456789"
    return (
        b"\x0a"  # protocol version
        + b"8.0.0\x00"
        + struct.pack("<I", 1234)  # thread id
        + salt[:8]
        + b"\x00"
        + struct.pack("<H", capabilities & 0xFFFF)
        + b"\x21"  # charset
        + struct.pack("<H", AUTOCOMMIT)
        + struct.pack("<H", (capabilities >> 16) & 0xFFFF)
        + bytes([len(salt) + 1])
        + b"\x00" * 10
        + salt[8:]
        + b"\x00"
        + b"mysql_native_password\x00"
    )


def ok_payload(status=AUTOCOMMIT):
    return b"\x00\x00\x00" + struct.pack("<HH", status, 0)


def eof_payload(status=AUTOCOMMIT):
    return b"\xfe" + struct.pack("<HH", 0, status)


def lenenc_str(data):
    return _lenenc_int(len(data)) + data


def column_def(name):
    return (
        lenenc_str(b"def")  # catalog
        + lenenc_str(b"")  # schema
        + lenenc_str(b"")  # table
        + lenenc_str(b"")  # org_table
        + lenenc_str(name)
        + lenenc_str(b"")  # org_name
        + b"\x0c"  # length of fixed fields
        + struct.pack("<H", 33)  # charsetnr (utf8)
        + struct.pack("<I", 1024)  # column length
        + bytes([FIELD_TYPE.VAR_STRING])
        + struct.pack("<H", 0)  # flags
        + b"\x00"  # decimals
        + b"\x00\x00"  # filler
    )


def result_set_payloads(columns, rows, status=AUTOCOMMIT):
    payloads = [_lenenc_int(len(columns))]
    payloads += [column_def(name) for name in columns]
    payloads.append(eof_payload(status))
    for row in rows:
        payloads.append(b"".join(lenenc_str(value) for value in row))
    payloads.append(eof_payload(status))
    return payloads


def wrap_packets(payloads, first_seq=1):
    return b"".join(
        mysql_packet(payload, (first_seq + i) % 256)
        for i, payload in enumerate(payloads)
    )


def handshake_stream(capabilities=SERVER_CAPS, compressed=None):
    """Server bytes for the connect exchange (greeting, auth OK, SET NAMES OK)."""
    if compressed is None:
        compressed = bool(capabilities & CLIENT.COMPRESS)
    stream = mysql_packet(handshake_payload(capabilities), 0)
    if compressed:
        # Compressed frames start after the client's handshake response.
        # The compressed sequence id is shared by both directions: the
        # client's SET NAMES command takes frame #1.
        stream += compressed_frame(mysql_packet(ok_payload(), 2), 0)
        stream += compressed_frame(mysql_packet(ok_payload(), 1), 2)
    else:
        stream += mysql_packet(ok_payload(), 2)
        stream += mysql_packet(ok_payload(), 1)
    return stream


class FakeSocket:
    """Serves queued server bytes and records everything the client writes."""

    def __init__(self, incoming=b""):
        self._incoming = bytearray(incoming)
        self.outgoing = bytearray()
        self.closed = False

    def read(self, num_bytes):
        data = bytes(self._incoming[:num_bytes])
        del self._incoming[:num_bytes]
        return data

    def makefile(self, mode):
        return self

    def sendall(self, data):
        self.outgoing += data

    def settimeout(self, value):
        pass

    def setsockopt(self, *args):
        pass

    def close(self):
        self.closed = True


def make_connection(sock, **kwargs):
    kwargs.setdefault("user", "test")
    kwargs.setdefault("autocommit", None)
    kwargs.setdefault("defer_connect", True)
    conn = pymysql.connect(**kwargs)
    conn.connect(sock=sock)
    return conn


def parse_client_bytes(data, compressed=True):
    """Parse recorded client->server bytes.

    Returns (handshake_response, frame_seq_ids, frame_payloads). Frame
    payloads are the decompressed MySQL packet streams of each frame.
    """
    length = int.from_bytes(data[:3], "little")
    handshake_response = data[: 4 + length]
    rest = data[4 + length :]
    if not compressed:
        return handshake_response, None, [rest]
    seq_ids = []
    payloads = []
    i = 0
    while i < len(rest):
        payload_len = int.from_bytes(rest[i : i + 3], "little")
        assert payload_len <= MAX_PACKET_LEN
        seq_id = rest[i + 3]
        uncompressed_len = int.from_bytes(rest[i + 4 : i + 7], "little")
        payload = rest[i + 7 : i + 7 + payload_len]
        assert len(payload) == payload_len
        if uncompressed_len:
            payload = zlib.decompress(payload)
            assert len(payload) == uncompressed_len
        seq_ids.append(seq_id)
        payloads.append(payload)
        i += 7 + payload_len
    return handshake_response, seq_ids, payloads


def parse_mysql_packets(data):
    packets = []
    i = 0
    while i < len(data):
        length = int.from_bytes(data[i : i + 3], "little")
        packets.append((data[i + 3], data[i + 4 : i + 4 + length]))
        i += 4 + length
    return packets


class TestCompressNegotiation:
    def test_compression_negotiated(self):
        sock = FakeSocket(handshake_stream())
        conn = make_connection(sock, compress=True)
        assert conn._compressed
        assert conn.client_flag & CLIENT.COMPRESS

        handshake_response, seq_ids, payloads = parse_client_bytes(sock.outgoing)
        # The handshake response itself is never compressed.
        assert handshake_response[3] == 1
        capabilities = struct.unpack("<I", handshake_response[4:8])[0]
        assert capabilities & CLIENT.COMPRESS
        # SET NAMES went through the compressed layer as frame #1.
        assert seq_ids == [1]
        packets = parse_mysql_packets(payloads[0])
        assert packets == [(0, bytes([COMMAND.COM_QUERY]) + b"SET NAMES utf8mb4")]

    def test_compression_not_used_without_server_support(self):
        capabilities = SERVER_CAPS & ~CLIENT.COMPRESS
        sock = FakeSocket(handshake_stream(capabilities))
        conn = make_connection(sock, compress=True)
        assert not conn._compressed

        # The wire format stays plain MySQL packets.
        handshake_response, _, payloads = parse_client_bytes(
            sock.outgoing, compressed=False
        )
        capabilities = struct.unpack("<I", handshake_response[4:8])[0]
        assert capabilities & CLIENT.COMPRESS
        packets = parse_mysql_packets(payloads[0])
        assert packets == [(0, bytes([COMMAND.COM_QUERY]) + b"SET NAMES utf8mb4")]

    def test_no_compression_by_default(self):
        sock = FakeSocket(handshake_stream(compressed=False))
        conn = make_connection(sock)
        assert not conn._compressed
        handshake_response, _, _ = parse_client_bytes(sock.outgoing, compressed=False)
        capabilities = struct.unpack("<I", handshake_response[4:8])[0]
        assert not capabilities & CLIENT.COMPRESS


class TestCompressQueries:
    def test_simple_query(self):
        payloads = result_set_payloads([b"1"], [[b"1"]])
        stream = handshake_stream()
        # Several MySQL packets in one compressed frame, and the rest in a
        # frame stored uncompressed.
        stream += compressed_frame(wrap_packets(payloads[:2]), 4)
        stream += compressed_frame(
            wrap_packets(payloads[2:], first_seq=3), 5, compress=False
        )
        sock = FakeSocket(stream)
        conn = make_connection(sock, compress=True)

        cur = conn.cursor()
        cur.execute("SELECT 1")
        assert cur.fetchall() == (("1",),)
        conn.close()

        _, seq_ids, payloads = parse_client_bytes(sock.outgoing)
        assert seq_ids == [1, 3, 6]
        packets = parse_mysql_packets(payloads[1])
        assert packets == [(0, bytes([COMMAND.COM_QUERY]) + b"SELECT 1")]
        # COM_QUIT is sent through the compressed layer too.
        packets = parse_mysql_packets(payloads[2])
        assert packets == [(0, bytes([COMMAND.COM_QUIT]))]

    def test_multi_result_sets(self):
        stream = handshake_stream()
        rs1 = result_set_payloads([b"a"], [[b"1"]], status=AUTOCOMMIT | MORE_RESULTS)
        rs2 = result_set_payloads([b"b"], [[b"2"]])
        data = wrap_packets(rs1) + wrap_packets(rs2, first_seq=1 + len(rs1))
        # Tiny frames force MySQL packets to span compressed frames.
        stream += frames_for(data, first_seq=4, chunk_size=7)
        sock = FakeSocket(stream)
        conn = make_connection(sock, compress=True)

        cur = conn.cursor()
        cur.execute("CALL two_results")
        assert cur.fetchall() == (("1",),)
        assert cur.nextset()
        assert cur.fetchall() == (("2",),)
        assert cur.nextset() is None

    def test_sscursor(self):
        stream = handshake_stream()
        rows = [[str(i).encode()] for i in range(5)]
        data = wrap_packets(result_set_payloads([b"n"], rows))
        stream += frames_for(data, first_seq=4, chunk_size=11)
        sock = FakeSocket(stream)
        conn = make_connection(sock, compress=True)

        cur = conn.cursor(SSCursor)
        cur.execute("SELECT n")
        assert cur.fetchall() == [(str(i),) for i in range(5)]

    def test_large_row_spanning_packets_and_frames(self):
        value = "x" * (MAX_PACKET_LEN + 100)
        stream = handshake_stream()
        packets = []
        seq = 1
        for payload in result_set_payloads([b"c"], [[value.encode()]]):
            while True:
                chunk, payload = payload[:MAX_PACKET_LEN], payload[MAX_PACKET_LEN:]
                packets.append(mysql_packet(chunk, seq % 256))
                seq += 1
                if len(chunk) < MAX_PACKET_LEN:
                    break
        # Re-chunk the packet stream into frames at unaligned boundaries.
        stream += frames_for(b"".join(packets), first_seq=4, chunk_size=3 * 1024 * 1024)
        sock = FakeSocket(stream)
        conn = make_connection(sock, compress=True)

        cur = conn.cursor()
        cur.execute("SELECT big")
        assert cur.fetchall() == ((value,),)

    def test_large_query_is_fragmented(self):
        stream = handshake_stream()
        # The big command takes compressed frames #3, #4 and #5.
        stream += compressed_frame(mysql_packet(ok_payload(), 2), 6)
        sock = FakeSocket(stream)
        conn = make_connection(sock, compress=True)

        sql = "SELECT '" + "x" * (MAX_PACKET_LEN + 100) + "'"
        cur = conn.cursor()
        cur.execute(sql)

        _, seq_ids, payloads = parse_client_bytes(sock.outgoing)
        assert seq_ids == [1, 3, 4, 5]
        packets = parse_mysql_packets(b"".join(payloads[1:]))
        assert [seq_id for seq_id, _ in packets] == [0, 1]
        sent = packets[0][1] + packets[1][1]
        assert sent == bytes([COMMAND.COM_QUERY]) + sql.encode("utf8")


class TestCompressErrors:
    def query_with_response(self, response):
        sock = FakeSocket(handshake_stream() + response)
        conn = make_connection(sock, compress=True)
        return conn, conn.cursor()

    def test_corrupt_compressed_payload_closes_connection(self):
        response = (
            struct.pack("<I", 8)[:3]
            + bytes([4])
            + struct.pack("<I", 100)[:3]
            + b"garbage!"
        )
        conn, cur = self.query_with_response(response)
        with pytest.raises(err.OperationalError, match="can't decompress"):
            cur.execute("SELECT 1")
        assert not conn.open

    def test_uncompressed_length_mismatch_closes_connection(self):
        payload = zlib.compress(b"abc")
        response = (
            struct.pack("<I", len(payload))[:3]
            + bytes([4])
            + struct.pack("<I", 10)[:3]  # declared length doesn't match
            + payload
        )
        conn, cur = self.query_with_response(response)
        with pytest.raises(err.OperationalError, match="decompressed bytes"):
            cur.execute("SELECT 1")
        assert not conn.open

    def test_compressed_sequence_mismatch_closes_connection(self):
        response = compressed_frame(mysql_packet(ok_payload(), 1), 7)
        conn, cur = self.query_with_response(response)
        with pytest.raises(err.InternalError, match="Compressed packet sequence"):
            cur.execute("SELECT 1")
        assert not conn.open
