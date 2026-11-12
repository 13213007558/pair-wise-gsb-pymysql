import os

from pymysql import LocalInfileSource, OperationalError, cursors
from pymysql.constants import CR, ER
from pymysql.tests import base

__all__ = ["TestLoadLocal"]

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


class TestLoadLocal(base.PyMySQLTestCase):
    def test_no_source(self):
        """A server file request without an approved source is refused"""
        conn = self.connect()
        c = conn.cursor()
        c.execute("CREATE TABLE test_load_local (a INTEGER, b INTEGER)")
        try:
            self.assertRaises(
                OperationalError,
                c.execute,
                (
                    "LOAD DATA LOCAL INFILE 'no_data.txt' INTO TABLE "
                    "test_load_local fields terminated by ','"
                ),
            )
            # The refusal must be fully resolved so the connection stays usable.
            c.execute("SELECT 1")
            self.assertEqual((1,), c.fetchone())
        finally:
            c.execute("DROP TABLE test_load_local")
            c.close()

    def test_load_file(self):
        """Test load local infile with an approved source"""
        conn = self.connect()
        c = conn.cursor()
        c.execute("CREATE TABLE test_load_local (a INTEGER, b INTEGER)")
        filename = os.path.join(DATA_DIR, "load_local_data.txt")
        try:
            with open(filename, "rb") as f:
                c.execute(
                    f"LOAD DATA LOCAL INFILE '{filename}' INTO TABLE test_load_local"
                    + " FIELDS TERMINATED BY ','",
                    local_infile_source=LocalInfileSource(f),
                )
            c.execute("SELECT COUNT(*) FROM test_load_local")
            self.assertEqual(22749, c.fetchone()[0])
        finally:
            c.execute("DROP TABLE test_load_local")

    def test_unbuffered_load_file(self):
        """Test unbuffered load local infile with an approved source"""
        conn = self.connect()
        c = conn.cursor(cursors.SSCursor)
        c.execute("CREATE TABLE test_load_local (a INTEGER, b INTEGER)")
        filename = os.path.join(DATA_DIR, "load_local_data.txt")
        try:
            with open(filename, "rb") as f:
                c.execute(
                    f"LOAD DATA LOCAL INFILE '{filename}' INTO TABLE test_load_local"
                    + " FIELDS TERMINATED BY ','",
                    local_infile_source=LocalInfileSource(f),
                )
            c.execute("SELECT COUNT(*) FROM test_load_local")
            self.assertEqual(22749, c.fetchone()[0])
        finally:
            c.close()
            conn.close()
            conn.connect()
            c = conn.cursor()
            c.execute("DROP TABLE test_load_local")

    def test_load_warnings(self):
        """Test load local infile produces the appropriate warnings"""
        conn = self.connect()
        c = conn.cursor()
        c.execute("CREATE TABLE test_load_local (a INTEGER, b INTEGER)")
        filename = os.path.join(DATA_DIR, "load_local_warn_data.txt")
        try:
            with open(filename, "rb") as f:
                c.execute(
                    (
                        "LOAD DATA LOCAL INFILE '{0}' INTO TABLE "
                        + "test_load_local FIELDS TERMINATED BY ','"
                    ).format(filename),
                    local_infile_source=LocalInfileSource(f),
                )
            self.assertEqual(1, c.warning_count)

            c.execute("SHOW WARNINGS")
            w = c.fetchone()

            self.assertEqual(ER.TRUNCATED_WRONG_VALUE_FOR_FIELD, w[1])
            self.assertIn(
                "incorrect integer value",
                w[2].lower(),
            )
        finally:
            c.execute("DROP TABLE test_load_local")
            c.close()

    def test_source_limit_is_enforced(self):
        """An over-budget source aborts without corrupting the connection"""
        conn = self.connect()
        c = conn.cursor()
        c.execute("CREATE TABLE test_load_local (a INTEGER, b INTEGER)")
        filename = os.path.join(DATA_DIR, "load_local_data.txt")
        try:
            source = LocalInfileSource(bytes(10240), max_bytes=10)
            with self.assertRaises(OperationalError) as ctx:
                c.execute(
                    f"LOAD DATA LOCAL INFILE '{filename}' INTO TABLE "
                    "test_load_local FIELDS TERMINATED BY ','",
                    local_infile_source=source,
                )
            self.assertEqual(CR.CR_NET_PACKET_TOO_LARGE, ctx.exception.args[0])
            c.execute("SELECT 1")
            self.assertEqual((1,), c.fetchone())
        finally:
            c.execute("DROP TABLE test_load_local")
            c.close()


if __name__ == "__main__":
    import unittest

    unittest.main()
