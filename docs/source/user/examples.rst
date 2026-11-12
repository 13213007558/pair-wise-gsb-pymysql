.. _examples:

========
Examples
========

.. _CRUD:

CRUD
----

The following examples make use of a simple table

.. code:: sql

   CREATE TABLE `users` (
       `id` int(11) NOT NULL AUTO_INCREMENT,
       `email` varchar(255) COLLATE utf8_bin NOT NULL,
       `password` varchar(255) COLLATE utf8_bin NOT NULL,
       PRIMARY KEY (`id`)
   ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
   AUTO_INCREMENT=1 ;


.. code:: python

    import pymysql.cursors

    # Connect to the database
    connection = pymysql.connect(host='localhost',
                                 user='user',
                                 password='passwd',
                                 database='db',
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    with connection:
        with connection.cursor() as cursor:
            # Create a new record
            sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
            cursor.execute(sql, ('webmaster@python.org', 'very-secret'))

        # connection is not autocommit by default. So you must commit to save
        # your changes.
        connection.commit()

        with connection.cursor() as cursor:
            # Read a single record
            sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
            cursor.execute(sql, ('webmaster@python.org',))
            result = cursor.fetchone()
            print(result)


This example will print:

.. code:: python

    {'id': 1, 'password': 'very-secret'}


.. _load-data-local:

LOAD DATA LOCAL INFILE
----------------------

A server asking for a ``LOAD DATA LOCAL INFILE`` payload does not authorize
the client to open any local path. The application must approve a byte source
for each statement and pass it to ``execute()``:

.. code:: python

    import pymysql

    connection = pymysql.connect(host='localhost',
                                 user='user',
                                 password='passwd',
                                 database='db',
                                 local_infile=True)

    with connection:
        with open('/srv/imports/approved.csv', 'rb') as f:
            source = pymysql.LocalInfileSource(f, max_bytes=10 * 1024 * 1024)
            with connection.cursor() as cursor:
                cursor.execute(
                    "LOAD DATA LOCAL INFILE 'approved.csv' "
                    "INTO TABLE t FIELDS TERMINATED BY ','",
                    local_infile_source=source,
                )

The filename in the SQL statement is only what the server asked for; the
driver never resolves it. Only bytes from the approved source are sent, at
most ``max_bytes`` of them (also capped by the connection's
``local_infile_max_bytes`` / ``max_allowed_packet``). A statement without an
approved source, a second file request in one statement, a read error, or an
over-budget source is refused after the upload exchange is closed, so the
connection stays in sync. If the server's response cannot be consumed, the
connection is closed instead of letting a later query read it as a result.
