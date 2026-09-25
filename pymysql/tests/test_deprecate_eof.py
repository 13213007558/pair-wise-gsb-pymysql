"""Unit tests for CLIENT_DEPRECATE_EOF result parsing."""

from pymysql.connections import MySQLResult
from pymysql.constants import CLIENT, SERVER_STATUS
from pymysql.protocol import MysqlPacket


class FakeConnection:
    def __init__(self, packets, deprecate_eof):
        self._packets = [MysqlPacket(p, "utf8") for p in packets]
        self._deprecate_eof = deprecate_eof
        self.encoding = "utf8"

    def _read_packet(self, packet_type=MysqlPacket):
        return self._packets.pop(0)


def ok_terminator(warnings=0, status=SERVER_STATUS.SERVER_STATUS_AUTOCOMMIT):
    # OK packet with a 0xFE header, replacing the EOF packet.
    return (
        b"\xfe\x00\x00"  # header, affected rows, insert id
        + status.to_bytes(2, "little")
        + warnings.to_bytes(2, "little")
    )


def large_field_row(data):
    # A row whose first field uses the 0xFE length-encoded integer prefix.
    return b"\xfe" + len(data).to_bytes(8, "little") + data


def make_result(packets, deprecate_eof=True):
    result = MySQLResult(FakeConnection(packets, deprecate_eof))
    result.converters = [("utf8", None)]
    return result


def test_capability_is_requested():
    assert CLIENT.CAPABILITIES & CLIENT.DEPRECATE_EOF


def test_row_starting_with_0xfe_is_not_terminator():
    data = b"x" * 16
    result = make_result([large_field_row(data), ok_terminator()])
    result._read_rowdata_packet()
    assert result.rows == ((data.decode("utf8"),),)
    assert result.affected_rows == 1


def test_ok_terminator_preserves_status():
    status = (
        SERVER_STATUS.SERVER_STATUS_AUTOCOMMIT
        | SERVER_STATUS.SERVER_MORE_RESULTS_EXISTS
    )
    result = make_result(
        [large_field_row(b"a"), ok_terminator(warnings=3, status=status)]
    )
    result._read_rowdata_packet()
    assert result.warning_count == 3
    assert result.has_next == SERVER_STATUS.SERVER_MORE_RESULTS_EXISTS


def test_empty_result_with_warnings():
    result = make_result([ok_terminator(warnings=2)])
    result._read_rowdata_packet()
    assert result.rows == ()
    assert result.affected_rows == 0
    assert result.warning_count == 2
    assert not result.has_next


def test_unbuffered_reads_rows_then_terminator():
    result = make_result([large_field_row(b"a"), ok_terminator(warnings=1)])
    result.unbuffered_active = True
    assert result._read_rowdata_packet_unbuffered() == ("a",)
    assert result.unbuffered_active
    assert result._read_rowdata_packet_unbuffered() is None
    assert not result.unbuffered_active
    assert result.warning_count == 1


def test_legacy_eof_still_used_without_negotiation():
    # Old-style EOF packet: 0xFE, warnings (2 bytes), status (2 bytes).
    eof = (
        b"\xfe"
        + (2).to_bytes(2, "little")
        + (
            SERVER_STATUS.SERVER_STATUS_AUTOCOMMIT
            | SERVER_STATUS.SERVER_MORE_RESULTS_EXISTS
        ).to_bytes(2, "little")
    )
    result = make_result([eof], deprecate_eof=False)
    result._read_rowdata_packet()
    assert result.rows == ()
    assert result.warning_count == 2
    assert result.has_next == SERVER_STATUS.SERVER_MORE_RESULTS_EXISTS
