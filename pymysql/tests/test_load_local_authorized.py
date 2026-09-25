"""Tests for per-execution authorized LOCAL INFILE upload sources.

These tests use a fake server over a socketpair: no real MySQL server is
required. The fake server records every data byte the client sends and
drives the LOCAL INFILE exchange with correct and incorrect identifiers.
"""

import io
import os
import socket
import struct
import tempfile
import threading
import unittest

import pymysql
from pymysql.connections import Connection

OK_PAYLOAD = b"\x00\x00\x00\x02\x00\x00\x00"


def _read_exact(sock, num_bytes):
    data = bytearray()
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionError("connection closed by peer")
        data += chunk
    return bytes(data)


class FakeServer(threading.Thread):
    """A scripted fake MySQL server speaking just enough of the protocol."""

    def __init__(self, script):
        super().__init__(daemon=True)
        self._script = script
        self.client_sock, self.sock = socket.socketpair()
        self.sock.settimeout(10)
        self.client_sock.settimeout(10)
        # All payload bytes received in LOCAL INFILE data packets.
        self.received = bytearray()
        self.error = None

    def run(self):
        try:
            self._script(self)
        except Exception as e:  # noqa: BLE001 - reported to the test body
            self.error = e
        finally:
            self.sock.close()

    # -- protocol primitives -------------------------------------------------

    def read_packet(self):
        header = _read_exact(self.sock, 4)
        length = header[0] | (header[1] << 8) | (header[2] << 16)
        return header[3], _read_exact(self.sock, length)

    def send_packet(self, seq, payload):
        self.sock.sendall(struct.pack("<I", len(payload))[:3] + bytes([seq]) + payload)

    def send_ok(self, seq):
        self.send_packet(seq, OK_PAYLOAD)

    def send_error(self, seq, errno=1064, message=b"fake server error"):
        self.send_packet(seq, b"\xff" + struct.pack("<H", errno) + message)

    # -- exchange helpers ----------------------------------------------------

    def read_query(self):
        """Read a COM_QUERY command packet; returns the SQL bytes."""
        seq, payload = self.read_packet()
        assert seq == 0, f"expected seq 0, got {seq}"
        assert payload[0] == 0x03, f"expected COM_QUERY, got {payload[0]}"
        return payload[1:]

    def request_local_infile(self, name, seq=1):
        self.send_packet(seq, b"\xfb" + name)

    def read_infile_data(self, first_seq=2):
        """Read data packets until the empty terminator; returns last seq."""
        seq = first_seq
        while True:
            seq, payload = self.read_packet()
            if not payload:
                return seq
            self.received += payload


def make_connection(server, **kwargs):
    kwargs.setdefault("read_timeout", 10)
    kwargs.setdefault("write_timeout", 10)
    conn = Connection(defer_connect=True, **kwargs)
    conn._sock = server.client_sock
    conn._rfile = server.client_sock.makefile("rb")
    conn._next_seq_id = 0
    conn._current_timeout = None
    return conn


class AuthorizedLocalInfileTestCase(unittest.TestCase):
    def run_exchange(self, script, client, **conn_kwargs):
        server = FakeServer(script)
        conn = make_connection(server, **conn_kwargs)
        server.start()
        try:
            client(conn)
        finally:
            server.join(10)
            conn.close()
            server.client_sock.close()
        if server.error is not None:
            raise server.error
        return server

    def test_authorized_stream_upload(self):
        data = b"1,alpha\n2,beta\n" * 3000  # ~48KB, crosses packet boundary

        def script(srv):
            srv.read_query()
            srv.request_local_infile(b"upload-1")
            last = srv.read_infile_data()
            srv.send_ok((last + 1) % 256)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            conn.authorize_local_infile("upload-1", io.BytesIO(data))
            conn.query("LOAD DATA LOCAL INFILE 'upload-1' INTO TABLE t")
            self.assertIsNone(conn._local_infile_auth)
            # the connection is still in sync afterwards
            conn.query("DO 1")

        server = self.run_exchange(script, client)
        self.assertEqual(bytes(server.received), data)

    def test_authorized_iterator_upload_across_packets(self):
        # odd-sized iterator chunks are repacked to the packet boundary
        data = bytes(range(256)) * 20  # 5120 bytes
        chunks = (data[i : i + 999] for i in range(0, len(data), 999))

        def script(srv):
            srv.read_query()
            srv.request_local_infile(b"upload-2")
            last = srv.read_infile_data()
            srv.send_ok((last + 1) % 256)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            conn.authorize_local_infile(b"upload-2", chunks)
            conn.query("LOAD DATA LOCAL INFILE 'upload-2' INTO TABLE t")
            conn.query("DO 1")

        server = self.run_exchange(script, client, max_allowed_packet=1024)
        self.assertEqual(bytes(server.received), data)

    def test_empty_source(self):
        def script(srv):
            srv.read_query()
            srv.request_local_infile(b"upload-empty")
            last = srv.read_infile_data()
            srv.send_ok((last + 1) % 256)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            conn.authorize_local_infile("upload-empty", iter([]))
            conn.query("LOAD DATA LOCAL INFILE 'upload-empty' INTO TABLE t")
            conn.query("DO 1")

        server = self.run_exchange(script, client)
        self.assertEqual(bytes(server.received), b"")

    def test_wrong_token_is_refused_without_file_fallback(self):
        # The server asks for a real file on disk; the authorization for a
        # different token must prevent any file access.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"secret-file-contents")
            path = f.name
        self.addCleanup(os.unlink, path)

        def script(srv):
            srv.read_query()
            srv.request_local_infile(os.fsencode(path))
            last = srv.read_infile_data()
            srv.send_ok((last + 1) % 256)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            conn.authorize_local_infile("upload-3", io.BytesIO(b"csv-data"))
            with self.assertRaises(pymysql.OperationalError):
                conn.query(f"LOAD DATA LOCAL INFILE '{path}' INTO TABLE t")
            self.assertIsNone(conn._local_infile_auth)
            # the exchange was terminated per protocol: connection resynced
            conn.query("DO 1")

        server = self.run_exchange(script, client, local_infile=True)
        self.assertEqual(bytes(server.received), b"")

    def test_request_without_authorization_and_local_infile_disabled(self):
        def script(srv):
            srv.read_query()
            srv.request_local_infile(b"anything")
            last = srv.read_infile_data()
            srv.send_ok((last + 1) % 256)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            with self.assertRaises(RuntimeError):
                conn.query("LOAD DATA LOCAL INFILE 'anything' INTO TABLE t")
            conn.query("DO 1")

        server = self.run_exchange(script, client)
        self.assertEqual(bytes(server.received), b"")

    def test_source_read_error(self):
        def broken_source():
            yield b"x" * 20000  # fills one 16KB packet, then some
            raise ValueError("source exploded")

        def script(srv):
            srv.read_query()
            srv.request_local_infile(b"upload-4")
            last = srv.read_infile_data()
            srv.send_ok((last + 1) % 256)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            conn.authorize_local_infile("upload-4", broken_source())
            with self.assertRaises(ValueError):
                conn.query("LOAD DATA LOCAL INFILE 'upload-4' INTO TABLE t")
            self.assertIsNone(conn._local_infile_auth)
            conn.query("DO 1")

        server = self.run_exchange(script, client)
        # complete packets were sent; the partial tail was discarded
        self.assertEqual(bytes(server.received), b"x" * 16384)

    def test_server_error_instead_of_request(self):
        def script(srv):
            srv.read_query()
            srv.send_error(1)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            conn.authorize_local_infile("upload-5", io.BytesIO(b"csv-data"))
            with self.assertRaises(pymysql.MySQLError):
                conn.query("LOAD DATA LOCAL INFILE 'upload-5' INTO TABLE t")
            self.assertIsNone(conn._local_infile_auth)
            conn.query("DO 1")

        self.run_exchange(script, client)

    def test_server_error_after_data(self):
        data = b"1,x\n" * 100

        def script(srv):
            srv.read_query()
            srv.request_local_infile(b"upload-6")
            last = srv.read_infile_data()
            srv.send_error((last + 1) % 256)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            conn.authorize_local_infile("upload-6", io.BytesIO(data))
            with self.assertRaises(pymysql.MySQLError):
                conn.query("LOAD DATA LOCAL INFILE 'upload-6' INTO TABLE t")
            conn.query("DO 1")

        server = self.run_exchange(script, client)
        self.assertEqual(bytes(server.received), data)

    def test_repeated_request_is_rejected(self):
        data = b"1,y\n" * 100

        def script(srv):
            srv.read_query()
            srv.request_local_infile(b"upload-7")
            last = srv.read_infile_data()
            # ask for the same token a second time within one execution
            srv.request_local_infile(b"upload-7", seq=(last + 1) % 256)
            last = srv.read_infile_data(first_seq=(last + 2) % 256)
            srv.send_ok((last + 1) % 256)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            conn.authorize_local_infile("upload-7", io.BytesIO(data))
            with self.assertRaises(pymysql.OperationalError):
                conn.query("LOAD DATA LOCAL INFILE 'upload-7' INTO TABLE t")
            self.assertIsNone(conn._local_infile_auth)
            conn.query("DO 1")

        server = self.run_exchange(script, client)
        # only the first exchange carried data
        self.assertEqual(bytes(server.received), data)

    def test_authorization_does_not_leak_into_next_statement(self):
        data = b"1,z\n" * 100

        def script(srv):
            srv.read_query()
            srv.request_local_infile(b"upload-8")
            last = srv.read_infile_data()
            srv.send_ok((last + 1) % 256)
            # next statement: server asks for the same token again, but the
            # authorization was consumed with the previous execution
            srv.read_query()
            srv.request_local_infile(b"upload-8")
            last = srv.read_infile_data()
            srv.send_ok((last + 1) % 256)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            conn.authorize_local_infile("upload-8", io.BytesIO(data))
            conn.query("LOAD DATA LOCAL INFILE 'upload-8' INTO TABLE t")
            with self.assertRaises(RuntimeError):
                conn.query("LOAD DATA LOCAL INFILE 'upload-8' INTO TABLE t")
            conn.query("DO 1")

        server = self.run_exchange(script, client)
        self.assertEqual(bytes(server.received), data)

    def test_legacy_file_mode_still_works(self):
        data = b"9,legacy\n" * 100
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            path = f.name
        self.addCleanup(os.unlink, path)

        def script(srv):
            srv.read_query()
            srv.request_local_infile(os.fsencode(path))
            last = srv.read_infile_data()
            srv.send_ok((last + 1) % 256)
            srv.read_query()
            srv.send_ok(1)

        def client(conn):
            conn.query(f"LOAD DATA LOCAL INFILE '{path}' INTO TABLE t")
            conn.query("DO 1")

        server = self.run_exchange(script, client, local_infile=True)
        self.assertEqual(bytes(server.received), data)

    def test_authorize_local_infile_validates_arguments(self):
        server = FakeServer(lambda srv: None)
        conn = make_connection(server)
        try:
            with self.assertRaises(TypeError):
                conn.authorize_local_infile(123, io.BytesIO(b""))
            with self.assertRaises(TypeError):
                conn.authorize_local_infile("tok", "/etc/passwd")
            with self.assertRaises(TypeError):
                conn.authorize_local_infile("tok", 42)
        finally:
            conn.close()
            server.client_sock.close()
            server.sock.close()


if __name__ == "__main__":
    unittest.main()
