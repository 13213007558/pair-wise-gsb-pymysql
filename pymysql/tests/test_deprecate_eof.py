import pymysql
from pymysql.constants import CLIENT
from pymysql.tests import base


class TestDeprecateEOF(base.PyMySQLTestCase):
    def setUp(self):
        super().setUp()
        self.conn = self.connect()
        if not self.conn.server_capabilities & CLIENT.DEPRECATE_EOF:
            self.skipTest("server does not support CLIENT_DEPRECATE_EOF")

    def test_deprecate_eof_is_negotiated(self):
        self.assertTrue(self.conn.client_flag & CLIENT.DEPRECATE_EOF)
        cur = self.conn.cursor()
        cur.execute("SELECT 1")
        self.assertEqual(cur.fetchone(), (1,))
        self.assertEqual(cur.warning_count, 0)
        # A following query on the same connection must still work:
        # the terminating OK packet must be consumed exactly once.
        cur.execute("SELECT 2")
        self.assertEqual(cur.fetchone(), (2,))

    def test_row_starting_with_large_length_encoded_field(self):
        # A row whose first field uses the 0xFE 8-byte length-encoded
        # integer prefix must not be mistaken for an EOF/OK terminator.
        cur = self.conn.cursor()
        cur.execute("SELECT REPEAT('x', 16777216)")
        self.assertEqual(cur.fetchone(), ("x" * 16777216,))
        cur.execute("SELECT 1")
        self.assertEqual(cur.fetchone(), (1,))

    def test_empty_result_with_warnings(self):
        self.safe_create_table(
            self.conn,
            "test_deprecate_eof",
            "CREATE TABLE test_deprecate_eof (id INTEGER)",
        )
        cur = self.conn.cursor()
        # Empty result set; the terminating OK packet carries the warning.
        cur.execute("SELECT * FROM test_deprecate_eof WHERE id = 'abc'")
        self.assertEqual(cur.fetchall(), ())
        self.assertEqual(cur.warning_count, 1)
        cur.execute("SELECT 1")
        self.assertEqual(cur.fetchone(), (1,))
        self.assertEqual(cur.warning_count, 0)

    def test_sscursor(self):
        cur = self.conn.cursor(pymysql.cursors.SSCursor)
        cur.execute("SELECT 1 UNION ALL SELECT 2")
        self.assertEqual(cur.fetchall(), ((1,), (2,)))
        cur.close()
        cur = self.conn.cursor()
        cur.execute("SELECT 3")
        self.assertEqual(cur.fetchone(), (3,))

    def test_stored_procedure_multi_results(self):
        cur = self.conn.cursor()
        cur.execute("DROP PROCEDURE IF EXISTS test_deprecate_eof_proc")
        cur.execute(
            """CREATE PROCEDURE test_deprecate_eof_proc ()
            BEGIN
                SELECT 1;
                SELECT 2;
            END"""
        )
        self.addCleanup(cur.execute, "DROP PROCEDURE IF EXISTS test_deprecate_eof_proc")
        cur.callproc("test_deprecate_eof_proc")
        self.assertEqual(cur.fetchall(), ((1,),))
        self.assertTrue(cur.nextset())
        self.assertEqual(cur.fetchall(), ((2,),))
        while cur.nextset():
            pass
        cur.execute("SELECT 3")
        self.assertEqual(cur.fetchone(), (3,))
