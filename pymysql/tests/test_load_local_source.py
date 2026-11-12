"""Offline protocol tests for LOAD DATA LOCAL source authorization."""

import io
import os
import struct
import tempfile

import pytest

import pymysql
from pymysql import connections as pymysql_connections
from pymysql.connections import LocalInfileSource
from pymysql.constants import CR, ER

SQL = "LOAD DATA LOCAL INFILE 'requested-name' INTO TABLE t"


def _frame(payload, seq):
    return struct.pack("<I", len(payload))[:3] + bytes([seq]) + payload


def _load_request(name):
    return b"\xfb" + name


def _ok_packet(affected=1):
    return b"\x00" + bytes([affected]) + b"\x00" + b"\x00\x00\x00\x00"


def _error_packet(errno, message=b"refused by server"):
    return b"\xff" + struct.pack("<H", errno) + b"#HY000" + message


class FakeServer:
    """Records client writes and replays a scripted server byte stream."""

    def __init__(self):
        self.incoming = bytearray()
        self.writes = []
        self.closed = False

    def settimeout(self, value):
        pass

    def sendall(self, data):
        self.writes.append(bytes(data))

    def close(self):
        self.closed = True

    def makefile(self, mode="rb"):
        return _FakeReader(self)

    def reply(self, payload, seq):
        self.incoming += _frame(payload, seq)


class _FakeReader:
    def __init__(self, server):
        self.server = server
        self.closed = False

    def read(self, size):
        incoming = self.server.incoming
        chunk = bytes(incoming[:size])
        del incoming[:size]
        return chunk

    def close(self):
        self.closed = True


class FakeConnection(pymysql_connections.Connection):
    """A Connection whose transport is fully in memory."""

    def __init__(self, server, **kwargs):
        self.server = server
        super().__init__(user="u", defer_connect=True, **kwargs)
        self._sock = server
        self._rfile = server.makefile()
        self._next_seq_id = 1
        self._current_timeout = self._read_timeout
        self.server_status = 0


def _client_payloads(conn):
    """Decode packet frames the client wrote after its COM_QUERY frame."""
    out = []
    for frame in conn.server.writes[1:]:
        length = struct.unpack("<I", frame[:3] + b"\0")[0]
        out.append((frame[3], frame[4 : 4 + length]))
    return out


def _connect(**kwargs):
    kwargs.setdefault("local_infile", True)
    server = FakeServer()
    return server, FakeConnection(server, **kwargs)


def _next_query_succeeds(server, conn, affected=7):
    server.reply(_ok_packet(affected=affected), 1)
    assert conn.query("SELECT 1") == affected
    assert conn.open
    # The query's own result must have been read, leaving no packet behind.
    assert server.incoming == b""


class TestUnauthorizedRequest:
    def test_statement_without_approved_source_is_refused(self):
        server, conn = _connect()
        server.reply(_load_request(b"/etc/passwd"), 1)
        # The client's refusal must be answered before the next statement.
        server.reply(_ok_packet(affected=0), 3)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL)

        assert exc.value.args[0] == CR.CR_LOAD_DATA_LOCAL_INFILE_REJECTED
        # Only the empty terminator was sent; no requested bytes were opened
        # or uploaded.
        assert _client_payloads(conn) == [(2, b"")]
        _next_query_succeeds(server, conn)

    def test_local_infile_disabled_rejects_a_supplied_source(self):
        server, conn = _connect(local_infile=False)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL, local_infile_source=LocalInfileSource(b"data"))

        assert exc.value.args[0] == CR.CR_LOAD_DATA_LOCAL_INFILE_REJECTED
        assert server.writes == []
        assert conn.open

    def test_local_infile_disabled_refuses_a_server_request(self):
        server, conn = _connect(local_infile=False)
        server.reply(_load_request(b"/etc/passwd"), 1)
        server.reply(_ok_packet(affected=0), 3)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL)

        assert exc.value.args[0] == CR.CR_LOAD_DATA_LOCAL_INFILE_REJECTED
        assert _client_payloads(conn) == [(2, b"")]
        _next_query_succeeds(server, conn)

    def test_source_is_rejected_before_the_statement_is_sent(self):
        server, conn = _connect(local_infile=False)

        with pytest.raises(pymysql.OperationalError):
            conn.query(SQL, local_infile_source=LocalInfileSource(b"data"))

        assert conn.server.writes == []
        assert conn.open

    def test_authorization_does_not_leak_into_the_next_statement(self):
        server, conn = _connect()
        server.reply(_load_request(b"first.csv"), 1)
        server.reply(_ok_packet(affected=1), 4)
        conn.query(SQL, local_infile_source=LocalInfileSource(b"a,b\n"))

        server.reply(_load_request(b"second.csv"), 1)
        server.reply(_ok_packet(affected=0), 3)
        with pytest.raises(pymysql.OperationalError):
            conn.query(SQL)
        _next_query_succeeds(server, conn)

    def test_second_file_request_is_not_covered_by_one_authorization(self):
        server, conn = _connect()
        server.reply(_load_request(b"first.csv"), 1)
        # First upload: payload then empty terminator.  The server then asks
        # for another file instead of finishing.
        server.reply(_load_request(b"second.csv"), 4)
        server.reply(_ok_packet(affected=0), 6)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL, local_infile_source=LocalInfileSource(b"a,b\n"))

        assert exc.value.args[0] == CR.CR_LOAD_DATA_LOCAL_INFILE_REJECTED
        assert _client_payloads(conn) == [(2, b"a,b\n"), (3, b""), (5, b"")]
        _next_query_succeeds(server, conn)

    def test_refused_request_is_not_mistaken_for_the_next_result(self):
        # The server answers the refusal with another file request.  There is
        # no authorized source for it, so the connection must close rather
        # than read later packets as a query result.
        server, conn = _connect()
        server.reply(_load_request(b"/secret"), 1)
        server.reply(_load_request(b"/secret"), 3)

        with pytest.raises(pymysql.OperationalError):
            conn.query(SQL)

        assert not conn.open
        assert conn.server.closed
        with pytest.raises(pymysql.InterfaceError):
            conn.query("SELECT 1")


class TestSourceFailure:
    def test_paths_are_never_opened(self):
        server, conn = _connect()
        with tempfile.NamedTemporaryFile("wb", delete=False) as refused:
            refused.write(b"secret-content")
            refused_path = refused.name
        approved = tempfile.NamedTemporaryFile("wb", delete=False)
        try:
            approved.write(b"approved-content")
            approved.close()
            opened = open(approved.name, "rb")
            server.reply(_load_request(refused_path.encode()), 1)
            server.reply(_ok_packet(affected=1), 4)

            result = conn.query(SQL, local_infile_source=opened)

            assert result == 1
            assert _client_payloads(conn) == [
                (2, b"approved-content"),
                (3, b""),
            ]
            with open(refused_path, "rb") as refused_read:
                assert refused_read.read() == b"secret-content"
            _next_query_succeeds(server, conn)
        finally:
            os.unlink(refused_path)
            os.unlink(approved.name)

    def test_path_like_sources_are_rejected(self):
        with pytest.raises(TypeError):
            LocalInfileSource("/etc/passwd")
        with pytest.raises(TypeError):
            LocalInfileSource(__import__("pathlib").Path("/etc/passwd"))
        text_source = io.StringIO("text data")
        with pytest.raises(TypeError):
            LocalInfileSource(text_source).read(1)

    def test_read_error_aborts_and_keeps_the_connection_usable(self):
        class BrokenFile:
            closed = False

            def read(self, size):
                raise OSError("source read failed")

            def close(self):
                self.closed = True

        broken = BrokenFile()
        server, conn = _connect()
        server.reply(_load_request(b"broken.csv"), 1)
        server.reply(_ok_packet(affected=0), 3)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL, local_infile_source=LocalInfileSource(broken))

        assert exc.value.args[0] == CR.CR_LOAD_DATA_LOCAL_INFILE_REJECTED
        assert isinstance(exc.value.__cause__, OSError)
        assert _client_payloads(conn) == [(2, b"")]
        assert broken.closed
        _next_query_succeeds(server, conn)

    def test_short_read_is_an_error(self):
        class NoneFile:
            def read(self, size):
                return None

        server, conn = _connect()
        server.reply(_load_request(b"short.csv"), 1)
        server.reply(_ok_packet(affected=0), 3)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL, local_infile_source=LocalInfileSource(NoneFile()))

        assert exc.value.args[0] == CR.CR_LOAD_DATA_LOCAL_INFILE_REJECTED
        _next_query_succeeds(server, conn)


class TestByteBudget:
    def test_over_limit_bytes_are_never_sent(self):
        server, conn = _connect()
        server.reply(_load_request(b"too-big.csv"), 1)
        server.reply(_ok_packet(affected=0), 4)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(
                SQL,
                local_infile_source=LocalInfileSource(b"1234567890", max_bytes=5),
            )

        assert exc.value.args[0] == CR.CR_NET_PACKET_TOO_LARGE
        # Exactly the budget is sent, then the empty terminator; the sixth
        # byte was never written.
        assert _client_payloads(conn) == [(2, b"12345"), (3, b"")]
        _next_query_succeeds(server, conn)

    def test_zero_budget_sends_only_the_terminator(self):
        server, conn = _connect()
        server.reply(_load_request(b"zero.csv"), 1)
        server.reply(_ok_packet(affected=0), 3)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL, local_infile_source=LocalInfileSource(b"x", max_bytes=0))

        assert exc.value.args[0] == CR.CR_NET_PACKET_TOO_LARGE
        assert _client_payloads(conn) == [(2, b"")]
        _next_query_succeeds(server, conn)

    def test_connection_wide_cap_is_combined_with_source_cap(self):
        server, conn = _connect(local_infile_max_bytes=4)
        server.reply(_load_request(b"capped.csv"), 1)
        server.reply(_ok_packet(affected=0), 4)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(
                SQL,
                local_infile_source=LocalInfileSource(b"123456", max_bytes=10),
            )

        assert exc.value.args[0] == CR.CR_NET_PACKET_TOO_LARGE
        assert _client_payloads(conn) == [(2, b"1234"), (3, b"")]
        _next_query_succeeds(server, conn)

    def test_exact_budget_is_a_normal_end(self):
        server, conn = _connect()
        server.reply(_load_request(b"exact.csv"), 1)
        server.reply(_ok_packet(affected=1), 4)

        result = conn.query(
            SQL, local_infile_source=LocalInfileSource(b"12345", max_bytes=5)
        )

        assert result == 1
        assert _client_payloads(conn) == [(2, b"12345"), (3, b"")]
        _next_query_succeeds(server, conn)


class TestNormalEnd:
    def test_bytes_are_uploaded_verbatim_then_terminated(self):
        server, conn = _connect()
        server.reply(_load_request(b"approved.csv"), 1)
        server.reply(_ok_packet(affected=2), 4)

        result = conn.query(SQL, local_infile_source=LocalInfileSource(b"1,2\n3,4\n"))

        assert result == 2
        assert _client_payloads(conn) == [(2, b"1,2\n3,4\n"), (3, b"")]
        _next_query_succeeds(server, conn)

    def test_source_is_closed_but_caller_file_is_not(self):
        server, conn = _connect()
        server.reply(_load_request(b"approved.csv"), 1)
        server.reply(_ok_packet(affected=1), 4)
        with tempfile.NamedTemporaryFile("wb", delete=False) as tf:
            tf.write(b"row\n")
            path = tf.name
        opened = open(path, "rb")
        try:
            source = LocalInfileSource(opened)
            conn.query(SQL, local_infile_source=source)
            assert opened.closed
            _next_query_succeeds(server, conn)

            server.reply(_load_request(b"approved.csv"), 1)
            server.reply(_ok_packet(affected=1), 4)
            raw = open(path, "rb")
            conn.query(SQL, local_infile_source=raw)
            assert not raw.closed
            raw.close()
        finally:
            os.unlink(path)

    def test_unbuffered_cursor_uses_the_same_one_shot_source(self):
        server, conn = _connect()
        server.reply(_load_request(b"approved.csv"), 1)
        server.reply(_ok_packet(affected=1), 4)
        cursor = conn.cursor(pymysql.cursors.SSCursor)

        cursor.execute(SQL, local_infile_source=LocalInfileSource(b"a,b\n"))
        cursor.close()

        assert _client_payloads(conn) == [(2, b"a,b\n"), (3, b"")]
        _next_query_succeeds(server, conn)

    def test_unexpected_packet_after_upload_closes_the_connection(self):
        server, conn = _connect()
        server.reply(_load_request(b"approved.csv"), 1)
        # A resultset packet where the LOAD response should be.
        server.reply(b"\x01", 4)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL, local_infile_source=LocalInfileSource(b"a,b\n"))

        assert exc.value.args[0] == CR.CR_COMMANDS_OUT_OF_SYNC
        assert not conn.open

    def test_missing_upload_response_closes_the_connection(self):
        server, conn = _connect()
        server.reply(_load_request(b"approved.csv"), 1)

        with pytest.raises(pymysql.OperationalError):
            conn.query(SQL, local_infile_source=LocalInfileSource(b"a,b\n"))

        assert not conn.open
        with pytest.raises(pymysql.InterfaceError):
            conn.query("SELECT 1")


class TestExchangeCleanup:
    def test_server_error_reply_to_refusal_is_consumed(self):
        server, conn = _connect()
        server.reply(_load_request(b"/etc/passwd"), 1)
        server.reply(_error_packet(ER.TEXTFILE_NOT_READABLE), 3)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL)

        assert exc.value.args[0] == CR.CR_LOAD_DATA_LOCAL_INFILE_REJECTED
        # The server's error packet belonged to the refused import; the next
        # statement must see its own reply, not that leftover error.
        _next_query_succeeds(server, conn)

    def test_server_error_reply_to_abort_is_consumed(self):
        class BrokenFile:
            def read(self, size):
                raise OSError("source vanished")

        server, conn = _connect()
        server.reply(_load_request(b"broken.csv"), 1)
        server.reply(_error_packet(ER.TEXTFILE_NOT_READABLE), 3)

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL, local_infile_source=LocalInfileSource(BrokenFile()))

        assert exc.value.args[0] == CR.CR_LOAD_DATA_LOCAL_INFILE_REJECTED
        _next_query_succeeds(server, conn)

    def test_lost_response_is_reported_and_connection_closed(self):
        server, conn = _connect()
        server.reply(_load_request(b"approved.csv"), 1)
        # No reply at all: reading the response hits end-of-stream.

        with pytest.raises(pymysql.OperationalError) as exc:
            conn.query(SQL, local_infile_source=LocalInfileSource(b"a,b\n"))

        assert exc.value.args[0] == CR.CR_SERVER_LOST
        assert not conn.open
