"""Tests for per-execution LOAD DATA LOCAL INFILE upload sources.

These tests use a fake MySQL server (a plain socket speaking just enough
of the MySQL protocol) so they do not need a real server.  The fake
server records the bytes the client sends during a LOCAL INFILE exchange
and then answers a normal query, which verifies that the connection is
back in sync (or was clearly closed) after the exchange.
"""

import io
import os
import socket
import struct
import threading

import pytest

import pymysql
from pymysql import err
from pymysql.constants import CLIENT, COMMAND, CR, ER

# Do not advertise CLIENT.SSL: the client would otherwise start TLS.
SERVER_CAPABILITIES = (
    CLIENT.LONG_PASSWORD
    | CLIENT.LONG_FLAG
    | CLIENT.PROTOCOL_41
    | CLIENT.TRANSACTIONS
    | CLIENT.SECURE_CONNECTION
    | CLIENT.MULTI_RESULTS
    | CLIENT.PLUGIN_AUTH
    | CLIENT.PLUGIN_AUTH_LENENC_CLIENT_DATA
    | CLIENT.CONNECT_ATTRS
    | CLIENT.LOCAL_FILES
)

OK_PAYLOAD = b"\x00\x00\x00\x02\x00\x00\x00"

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def _packet(payload, seq):
    return struct.pack("<I", len(payload))[:3] + bytes([seq % 256]) + payload


def _err_packet(errno, message):
    return b"\xff" + struct.pack("<H", errno) + b"#HY000" + message.encode()


class FakeMySQLServer(threading.Thread):
    """A minimal fake MySQL server running a scripted handler."""

    def __init__(self, handler):
        super().__init__(daemon=True)
        self.handler = handler
        # socketpair() needs no bind/listen, which sandboxes may forbid.
        self.client_sock, self.server_sock = socket.socketpair()
        self.error = None
        self.queries = []
        self.loads = []  # reassembled data of each LOCAL INFILE exchange
        self.packet_sizes = []  # individual data packet sizes

    def run(self):
        try:
            with self.server_sock as conn:
                conn.settimeout(10)
                self._handshake(conn)
                self.handler(self, conn)
        except BaseException as e:  # noqa: BLE001
            self.error = e
        finally:
            pass  # client_sock is owned by the Connection under test

    # -- protocol helpers -------------------------------------------------

    def _handshake(self, conn):
        salt = b"12345678" + b"abcdefghijkl"
        payload = (
            b"\x0a"
            + b"8.0.0-fake\x00"
            + struct.pack("<I", 1)
            + salt[:8]
            + b"\x00"
            + struct.pack("<H", SERVER_CAPABILITIES & 0xFFFF)
            + b"\x21"
            + struct.pack("<H", 2)  # SERVER_STATUS_AUTOCOMMIT
            + struct.pack("<H", SERVER_CAPABILITIES >> 16)
            + bytes([len(salt) + 1])
            + b"\x00" * 10
            + salt[8:]
            + b"\x00"
            + b"mysql_native_password\x00"
        )
        conn.sendall(_packet(payload, 0))
        self._read_packet(conn)  # handshake response
        conn.sendall(_packet(OK_PAYLOAD, 2))  # auth OK

    @staticmethod
    def _read_exact(conn, size):
        data = b""
        while len(data) < size:
            chunk = conn.recv(size - len(data))
            if not chunk:
                raise ConnectionError("client closed the connection")
            data += chunk
        return data

    def _read_packet(self, conn):
        header = self._read_exact(conn, 4)
        length = struct.unpack("<I", header[:3] + b"\x00")[0]
        payload = self._read_exact(conn, length) if length else b""
        return header[3], payload

    def read_query(self, conn):
        seq, payload = self._read_packet(conn)
        assert seq == 0, f"command packet must reset the sequence, got {seq}"
        assert payload and payload[0] == COMMAND.COM_QUERY
        sql = payload[1:].decode("utf-8")
        self.queries.append(sql)
        return sql

    def answer_init_queries(self, conn):
        """Answer the SET NAMES query sent right after connect."""
        sql = self.read_query(conn)
        assert sql.startswith("SET "), sql
        conn.sendall(_packet(OK_PAYLOAD, 1))

    def serve_load(self, conn, request_name):
        """Request a LOCAL INFILE and read data until the empty terminator.

        Returns the sequence number for the server's next packet.
        """
        conn.sendall(_packet(b"\xfb" + request_name, 1))
        data = b""
        seq = 2
        while True:
            got, payload = self._read_packet(conn)
            assert got == seq % 256, f"bad sequence {got}, expected {seq % 256}"
            seq += 1
            if not payload:
                break
            data += payload
            self.packet_sizes.append(len(payload))
        self.loads.append(data)
        return seq

    @staticmethod
    def send_ok(conn, seq):
        conn.sendall(_packet(OK_PAYLOAD, seq))

    @staticmethod
    def send_err(conn, seq, errno, message):
        conn.sendall(_packet(_err_packet(errno, message), seq))

    def answer_normal_query(self, conn):
        """Read a normal query and answer it with an OK packet."""
        self.read_query(conn)
        self.send_ok(conn, 1)


@pytest.fixture
def fake_server():
    servers = []

    def factory(handler):
        server = FakeMySQLServer(handler)
        server.start()
        servers.append(server)
        return server

    yield factory

    for server in servers:
        server.join(10)
        assert server.error is None, f"fake server failed: {server.error!r}"


def connect(server, **kwargs):
    kwargs.setdefault("local_infile", True)
    conn = pymysql.connect(
        user="fake",
        autocommit=None,
        read_timeout=10,
        write_timeout=10,
        defer_connect=True,
        **kwargs,
    )
    conn.connect(server.client_sock)
    return conn


def test_authorized_stream_upload(fake_server):
    payload = b"a,b\n1,2\n3,4\n"

    def handler(server, conn):
        server.answer_init_queries(conn)
        sql = server.read_query(conn)
        assert sql == "LOAD DATA LOCAL INFILE 'csv-upload-1' INTO TABLE t"
        seq = server.serve_load(conn, b"csv-upload-1")
        server.send_ok(conn, seq)
        server.answer_normal_query(conn)

    server = fake_server(handler)
    conn = connect(server)
    cur = conn.cursor()
    source = pymysql.LocalInfileSource("csv-upload-1", io.BytesIO(payload))
    cur.execute(
        "LOAD DATA LOCAL INFILE 'csv-upload-1' INTO TABLE t",
        local_infile_source=source,
    )
    assert cur.execute("DO 0") == 0
    conn.close()

    assert server.loads == [payload]
    assert server.queries[-1] == "DO 0"


def test_authorized_iterator_upload_crosses_packet_boundary(fake_server):
    # A chunk larger than the 16KB packet size must be split across
    # multiple MySQL packets and reassembled by the server.
    big = b"x" * 40000
    payload = b"head," + big + b",tail"

    def chunks():
        yield b"head,"
        yield big
        yield b""
        yield b",tail"

    def handler(server, conn):
        server.answer_init_queries(conn)
        server.read_query(conn)
        seq = server.serve_load(conn, b"csv-upload-2")
        server.send_ok(conn, seq)
        server.answer_normal_query(conn)

    server = fake_server(handler)
    conn = connect(server)
    cur = conn.cursor()
    source = pymysql.LocalInfileSource("csv-upload-2", chunks())
    cur.execute(
        "LOAD DATA LOCAL INFILE 'csv-upload-2' INTO TABLE t",
        local_infile_source=source,
    )
    assert cur.execute("DO 0") == 0
    conn.close()

    assert server.loads == [payload]
    assert max(server.packet_sizes) <= 16 * 1024
    assert len(server.packet_sizes) > 1


def test_empty_source(fake_server):
    def handler(server, conn):
        server.answer_init_queries(conn)
        server.read_query(conn)
        seq = server.serve_load(conn, b"csv-upload-3")
        server.send_ok(conn, seq)
        server.answer_normal_query(conn)

    server = fake_server(handler)
    conn = connect(server)
    cur = conn.cursor()
    source = pymysql.LocalInfileSource("csv-upload-3", iter([]))
    cur.execute(
        "LOAD DATA LOCAL INFILE 'csv-upload-3' INTO TABLE t",
        local_infile_source=source,
    )
    assert cur.execute("DO 0") == 0
    conn.close()

    assert server.loads == [b""]


def test_wrong_identifier_is_rejected_without_path_fallback(fake_server):
    # The server asks for a real file on disk while the authorization
    # names an unrelated token.  The client must end the exchange with an
    # empty response and must not fall back to reading the file.
    existing_file = os.path.join(DATA_DIR, "load_local_data.txt").encode()

    def handler(server, conn):
        server.answer_init_queries(conn)
        server.read_query(conn)
        seq = server.serve_load(conn, existing_file)
        server.send_ok(conn, seq)
        server.answer_normal_query(conn)

    server = fake_server(handler)
    conn = connect(server)
    cur = conn.cursor()
    source = pymysql.LocalInfileSource("csv-upload-4", io.BytesIO(b"secret"))
    with pytest.raises(err.OperationalError) as exc_info:
        cur.execute(
            "LOAD DATA LOCAL INFILE 'other' INTO TABLE t",
            local_infile_source=source,
        )
    assert exc_info.value.args[0] == CR.CR_UNKNOWN_ERROR
    # The connection completed the exchange per protocol and stays usable.
    assert cur.execute("DO 0") == 0
    conn.close()

    assert server.loads == [b""]
    assert server.queries[-1] == "DO 0"


def test_source_read_error_keeps_connection_in_sync(fake_server):
    class FailingStream:
        def __init__(self):
            self.calls = 0

        def read(self, size):
            self.calls += 1
            if self.calls == 1:
                return b"partial"
            raise ValueError("stream is broken")

    def handler(server, conn):
        server.answer_init_queries(conn)
        server.read_query(conn)
        seq = server.serve_load(conn, b"csv-upload-5")
        server.send_ok(conn, seq)
        server.answer_normal_query(conn)

    server = fake_server(handler)
    conn = connect(server)
    cur = conn.cursor()
    source = pymysql.LocalInfileSource("csv-upload-5", FailingStream())
    with pytest.raises(ValueError, match="stream is broken"):
        cur.execute(
            "LOAD DATA LOCAL INFILE 'csv-upload-5' INTO TABLE t",
            local_infile_source=source,
        )
    assert cur.execute("DO 0") == 0
    conn.close()

    assert server.loads == [b"partial"]


def test_server_early_error_is_raised(fake_server):
    def handler(server, conn):
        server.answer_init_queries(conn)
        server.read_query(conn)
        seq = server.serve_load(conn, b"csv-upload-6")
        server.send_err(conn, seq, ER.NOT_ALLOWED_COMMAND, "not allowed here")
        server.answer_normal_query(conn)

    server = fake_server(handler)
    conn = connect(server)
    cur = conn.cursor()
    source = pymysql.LocalInfileSource("csv-upload-6", io.BytesIO(b"data"))
    with pytest.raises(err.OperationalError) as exc_info:
        cur.execute(
            "LOAD DATA LOCAL INFILE 'csv-upload-6' INTO TABLE t",
            local_infile_source=source,
        )
    assert exc_info.value.args[0] == ER.NOT_ALLOWED_COMMAND
    assert cur.execute("DO 0") == 0
    conn.close()

    assert server.loads == [b"data"]


def test_repeated_request_within_one_execution(fake_server):
    payload = b"one-shot"

    def handler(server, conn):
        server.answer_init_queries(conn)
        server.read_query(conn)
        seq = server.serve_load(conn, b"csv-upload-7")
        # Instead of the final OK, the server asks for the same file
        # again.  The client must not resend the one-shot source.
        conn.sendall(_packet(b"\xfbcsv-upload-7", seq))
        got, empty = server._read_packet(conn)
        assert got == (seq + 1) % 256
        assert empty == b""
        server.send_ok(conn, seq + 2)
        server.answer_normal_query(conn)

    server = fake_server(handler)
    conn = connect(server)
    cur = conn.cursor()
    source = pymysql.LocalInfileSource("csv-upload-7", io.BytesIO(payload))
    with pytest.raises(err.OperationalError) as exc_info:
        cur.execute(
            "LOAD DATA LOCAL INFILE 'csv-upload-7' INTO TABLE t",
            local_infile_source=source,
        )
    assert exc_info.value.args[0] == CR.CR_COMMANDS_OUT_OF_SYNC
    assert cur.execute("DO 0") == 0
    conn.close()

    # The payload was sent exactly once.
    assert server.loads == [payload]


def test_authorization_does_not_leak_into_next_execute(fake_server):
    payload = b"first only"

    def handler(server, conn):
        server.answer_init_queries(conn)
        server.read_query(conn)
        seq = server.serve_load(conn, b"csv-upload-8")
        server.send_ok(conn, seq)
        # Second LOAD DATA without a fresh authorization: the client must
        # not serve the previous source again.  It falls back to the
        # legacy file-path mode, fails to open the token as a path and
        # terminates the exchange with an empty packet.
        server.read_query(conn)
        seq = server.serve_load(conn, b"csv-upload-8")
        server.send_ok(conn, seq)
        server.answer_normal_query(conn)

    server = fake_server(handler)
    conn = connect(server)
    cur = conn.cursor()
    source = pymysql.LocalInfileSource("csv-upload-8", io.BytesIO(payload))
    cur.execute(
        "LOAD DATA LOCAL INFILE 'csv-upload-8' INTO TABLE t",
        local_infile_source=source,
    )
    with pytest.raises(err.OperationalError) as exc_info:
        cur.execute("LOAD DATA LOCAL INFILE 'csv-upload-8' INTO TABLE t")
    assert exc_info.value.args[0] == ER.FILE_NOT_FOUND
    assert cur.execute("DO 0") == 0
    conn.close()

    assert server.loads == [payload, b""]


def test_legacy_file_path_mode_still_works(fake_server):
    filename = os.path.join(DATA_DIR, "load_local_data.txt")
    with open(filename, "rb") as f:
        payload = f.read()

    def handler(server, conn):
        server.answer_init_queries(conn)
        server.read_query(conn)
        seq = server.serve_load(conn, filename.encode())
        server.send_ok(conn, seq)
        server.answer_normal_query(conn)

    server = fake_server(handler)
    conn = connect(server)
    cur = conn.cursor()
    cur.execute(f"LOAD DATA LOCAL INFILE '{filename}' INTO TABLE t")
    assert cur.execute("DO 0") == 0
    conn.close()

    assert server.loads == [payload]


def test_unbuffered_cursor_upload(fake_server):
    payload = b"unbuffered"

    def handler(server, conn):
        server.answer_init_queries(conn)
        server.read_query(conn)
        seq = server.serve_load(conn, b"csv-upload-9")
        server.send_ok(conn, seq)
        server.answer_normal_query(conn)

    server = fake_server(handler)
    conn = connect(server)
    cur = conn.cursor(pymysql.cursors.SSCursor)
    source = pymysql.LocalInfileSource("csv-upload-9", io.BytesIO(payload))
    cur.execute(
        "LOAD DATA LOCAL INFILE 'csv-upload-9' INTO TABLE t",
        local_infile_source=source,
    )
    assert cur.execute("DO 0") == 0
    cur.close()
    conn.close()

    assert server.loads == [payload]


def test_local_infile_source_type_check():
    with pytest.raises(TypeError):
        pymysql.LocalInfileSource(123, io.BytesIO(b""))

    conn = pymysql.connect(defer_connect=True, local_infile=True)
    with pytest.raises(TypeError):
        conn.query("SELECT 1", local_infile_source=object())
