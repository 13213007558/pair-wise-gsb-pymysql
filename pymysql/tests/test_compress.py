"""Unit tests for the compressed protocol layer.

These tests use a fake socket so they don't need a MySQL server.
"""

import io
import random
import struct
import zlib

import pytest

from pymysql import err
from pymysql.connections import MAX_PACKET_LEN, Connection, _pack_int24
from pymysql.constants import CLIENT


def _compressed_frame(payload, seq_id):
    """Build a compressed protocol frame carrying payload."""
    body = zlib.compress(payload)
    return _pack_int24(len(body)) + bytes([seq_id]) + _pack_int24(len(payload)) + body


def _uncompressed_frame(payload, seq_id):
    """Build a compressed protocol frame whose payload is stored as is."""
    return _pack_int24(len(payload)) + bytes([seq_id]) + _pack_int24(0) + payload


def _mysql_packet(payload, seq_id):
    return _pack_int24(len(payload)) + bytes([seq_id]) + payload


def _parse_frames(data):
    """Split a byte string into (seq_id, uncompressed_len, payload) frames."""
    frames = []
    pos = 0
    while pos < len(data):
        compressed_len = data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16)
        seq_id = data[pos + 3]
        uncompressed_len = data[pos + 4] | (data[pos + 5] << 8) | (data[pos + 6] << 16)
        payload = data[pos + 7 : pos + 7 + compressed_len]
        assert len(payload) == compressed_len
        frames.append((seq_id, uncompressed_len, payload))
        pos += 7 + compressed_len
    return frames


class FakeSocket:
    def __init__(self, read_data=b""):
        self._read_file = io.BytesIO(read_data)
        self.sent = bytearray()
        self.closed = False
        self.timeout = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def setsockopt(self, *args):
        pass

    def makefile(self, mode):
        assert mode == "rb"
        return self._read_file

    def sendall(self, data):
        self.sent += data

    def close(self):
        self.closed = True


def _compressed_connection(read_data=b""):
    """A Connection with compression already negotiated, bound to a fake socket."""
    conn = Connection(defer_connect=True, ssl_disabled=True)
    sock = FakeSocket(read_data)
    conn._sock = sock
    conn._rfile = sock.makefile("rb")
    conn._current_timeout = None
    conn._compressed = True
    conn._compressed_read_seq_id = 0
    conn._compressed_write_seq_id = 0
    conn._compressed_buff = b""
    return conn, sock


def test_write_single_packet():
    conn, sock = _compressed_connection()
    packet = _mysql_packet(b"SELECT 1 " * 100, 0)
    conn._write_bytes(packet)

    frames = _parse_frames(bytes(sock.sent))
    assert len(frames) == 1
    seq_id, uncompressed_len, payload = frames[0]
    assert seq_id == 0
    assert uncompressed_len == len(packet)
    assert zlib.decompress(payload) == packet
    assert conn._compressed_write_seq_id == 1


def test_write_incompressible_data_stored_uncompressed():
    conn, sock = _compressed_connection()
    payload = random.Random(42).randbytes(1024)
    conn._write_bytes(payload)

    frames = _parse_frames(bytes(sock.sent))
    assert len(frames) == 1
    seq_id, uncompressed_len, body = frames[0]
    assert seq_id == 0
    assert uncompressed_len == 0
    assert body == payload


def test_write_empty_packet():
    conn, sock = _compressed_connection()
    conn._write_bytes(b"")
    frames = _parse_frames(bytes(sock.sent))
    assert len(frames) == 1
    assert frames[0][2] == b""


def test_write_large_packet_spans_frames():
    conn, sock = _compressed_connection()
    # Incompressible payload larger than one frame can hold.
    data = random.Random(42).randbytes(MAX_PACKET_LEN + 100)
    assert len(data) > MAX_PACKET_LEN
    conn._write_bytes(data)

    frames = _parse_frames(bytes(sock.sent))
    assert len(frames) == 2
    assert [f[0] for f in frames] == [0, 1]
    # Random data doesn't shrink, so both frames are stored uncompressed.
    assert [f[1] for f in frames] == [0, 0]
    restored = b"".join(f[2] for f in frames)
    assert restored == data


def test_read_multiple_packets_in_one_frame():
    packets = _mysql_packet(b"first", 0) + _mysql_packet(b"second", 1)
    conn, _ = _compressed_connection(_compressed_frame(packets, 0))
    assert conn._read_bytes(len(packets)) == packets
    assert conn._compressed_read_seq_id == 1


def test_read_packet_spanning_frames():
    packet = _mysql_packet(b"split across frames", 0)
    part1 = packet[:5]
    part2 = packet[5:]
    data = _compressed_frame(part1, 0) + _compressed_frame(part2, 1)
    conn, _ = _compressed_connection(data)
    assert conn._read_bytes(len(packet)) == packet


def test_read_uncompressed_frame():
    packet = _mysql_packet(b"stored as is", 0)
    conn, _ = _compressed_connection(_uncompressed_frame(packet, 0))
    assert conn._read_bytes(len(packet)) == packet


def test_read_mixed_frames():
    packet = _mysql_packet(b"mixed", 0)
    data = _compressed_frame(packet, 0) + _uncompressed_frame(packet, 1)
    conn, _ = _compressed_connection(data)
    assert conn._read_bytes(len(packet)) == packet
    assert conn._read_bytes(len(packet)) == packet


def test_read_large_packet_spanning_frames():
    payload = b"x" * (MAX_PACKET_LEN + 100)
    data = _compressed_frame(payload[:MAX_PACKET_LEN], 0) + _compressed_frame(
        payload[MAX_PACKET_LEN:], 1
    )
    conn, _ = _compressed_connection(data)
    assert conn._read_bytes(len(payload)) == payload


def test_read_invalid_zlib_closes_connection():
    payload = b"not a zlib stream at all"
    data = _pack_int24(len(payload)) + b"\x00" + _pack_int24(100) + payload
    conn, sock = _compressed_connection(data)
    with pytest.raises(err.OperationalError):
        conn._read_bytes(1)
    assert sock.closed
    assert conn._sock is None


def test_read_length_mismatch_closes_connection():
    payload = b"real data"
    body = zlib.compress(payload)
    # Declared uncompressed length doesn't match the actual data.
    data = _pack_int24(len(body)) + b"\x00" + _pack_int24(len(payload) + 1) + body
    conn, sock = _compressed_connection(data)
    with pytest.raises(err.OperationalError):
        conn._read_bytes(1)
    assert sock.closed
    assert conn._sock is None


def test_read_sequence_mismatch_closes_connection():
    conn, sock = _compressed_connection(_compressed_frame(b"data", 1))
    with pytest.raises(err.InternalError):
        conn._read_bytes(1)
    assert sock.closed
    assert conn._sock is None


def test_compressed_seq_ids_wrap_around():
    conn, sock = _compressed_connection()
    conn._compressed_write_seq_id = 255
    conn._write_bytes(b"a")
    conn._write_bytes(b"b")
    frames = _parse_frames(bytes(sock.sent))
    assert [f[0] for f in frames] == [255, 0]


def _server_greeting(server_capabilities):
    payload = b"\x0a"  # protocol version 10
    payload += b"8.0.0-test\x00"
    payload += struct.pack("<I", 1234)  # thread id
    payload += b"12345678\x00"  # auth-plugin-data-part-1 + filler
    payload += struct.pack("<H", server_capabilities & 0xFFFF)
    payload += struct.pack("<BHHB", 45, 0, (server_capabilities >> 16) & 0xFFFF, 21)
    payload += b"\x00" * 10  # reserved
    payload += b"abcdefghijkl\x00"  # auth-plugin-data-part-2
    payload += b"mysql_native_password\x00"
    return _mysql_packet(payload, 0)


def _ok_packet(seq_id):
    # header 0x00, affected rows 0, insert id 0, status 0, warnings 0
    return _mysql_packet(b"\x00\x00\x00\x00\x00\x00\x00", seq_id)


def _connect_with_script(server_capabilities, compressed):
    """Run the connection handshake against a scripted fake server."""
    auth_ok = _ok_packet(2)
    set_names_ok = _ok_packet(1)
    if compressed:
        read_data = (
            _server_greeting(server_capabilities)
            + _compressed_frame(auth_ok, 0)
            + _compressed_frame(set_names_ok, 1)
        )
    else:
        read_data = _server_greeting(server_capabilities) + auth_ok + set_names_ok
    sock = FakeSocket(read_data)
    conn = Connection(
        user="root",
        compress=True,
        ssl_disabled=True,
        defer_connect=True,
    )
    conn.connect(sock)
    return conn, sock


def test_handshake_enables_compression():
    caps = CLIENT.CAPABILITIES | CLIENT.MULTI_RESULTS | CLIENT.COMPRESS
    conn, sock = _connect_with_script(caps, compressed=True)
    try:
        assert conn._compressed

        sent = bytes(sock.sent)
        # The handshake response itself is sent uncompressed, seq id 1.
        resp_len = sent[0] | (sent[1] << 8) | (sent[2] << 16)
        assert sent[3] == 1
        client_flags = struct.unpack("<I", sent[4:8])[0]
        assert client_flags & CLIENT.COMPRESS
        rest = sent[4 + resp_len :]

        # Everything after that (SET NAMES) uses compressed frames,
        # with the compressed sequence starting at 0.
        frames = _parse_frames(rest)
        assert len(frames) == 1
        assert frames[0][0] == 0
        # Small payloads are stored uncompressed.
        assert frames[0][1] == 0
        inner = frames[0][2]
        assert inner[3] == 0  # inner MySQL packet sequence restarts too
        assert b"SET NAMES" in inner[4:]
    finally:
        conn.close()


def test_handshake_without_server_support_stays_uncompressed():
    caps = CLIENT.CAPABILITIES | CLIENT.MULTI_RESULTS
    conn, sock = _connect_with_script(caps, compressed=False)
    try:
        assert not conn._compressed
        sent = bytes(sock.sent)
        # Handshake response must not request CLIENT.COMPRESS.
        client_flags = struct.unpack("<I", sent[4:8])[0]
        assert not client_flags & CLIENT.COMPRESS
    finally:
        conn.close()


def test_named_pipe_still_unsupported():
    with pytest.raises(NotImplementedError):
        Connection(named_pipe=True)


def test_compress_disabled_by_default():
    conn = Connection(defer_connect=True, ssl_disabled=True)
    assert not conn.compress
    assert not conn._compressed
