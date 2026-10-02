spark.recording.store
=====================

.. py:module:: spark.recording.store


Attributes
----------

.. autoapisummary::

   spark.recording.store.fcntl
   spark.recording.store.msvcrt
   spark.recording.store.SCHEMA_VERSION
   spark.recording.store.SCHEMA
   spark.recording.store.MAX_STEPS
   spark.recording.store.SCHEDULER_VARIABLES
   spark.recording.store.TEMPORARY


Functions
---------

.. autoapisummary::

   spark.recording.store.now
   spark.recording.store.timestamp
   spark.recording.store.to_json
   spark.recording.store.from_json
   spark.recording.store.write_json
   spark.recording.store.write_arrays
   spark.recording.store.remove_partial
   spark.recording.store.git_state
   spark.recording.store.environment
   spark.recording.store.new_run_dir
   spark.recording.store.move_run_dir
   spark.recording.store.is_temporary
   spark.recording.store.window_file
   spark.recording.store.window_files
   spark.recording.store.checkpoint_file
   spark.recording.store.checkpoint_steps
   spark.recording.store.read_info
   spark.recording.store.heartbeat_age
   spark.recording.store.process
   spark.recording.store.abandoned
   spark.recording.store.connect
   spark.recording.store.connect_read_only
   spark.recording.store.commit
   spark.recording.store.write_request
   spark.recording.store.pending_requests
   spark.recording.store.read_request
   spark.recording.store.blank_request
   spark.recording.store.lock
   spark.recording.store.unlock
   spark.recording.store.locked


Module Contents
---------------

.. py:data:: fcntl
   :value: None


.. py:data:: msvcrt
   :value: None


.. py:data:: SCHEMA_VERSION
   :value: 4


   Version of the tables of ``index.sqlite``, stored as its ``user_version``.

.. py:data:: SCHEMA
   :value: Multiline-String

   .. raw:: html

      <details><summary>Show Value</summary>

   .. code-block:: python

      """
      CREATE TABLE IF NOT EXISTS keys (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, tag TEXT);
      CREATE TABLE IF NOT EXISTS scalars (t INTEGER, key INTEGER, value REAL, wall REAL);
      CREATE INDEX IF NOT EXISTS scalars_by_key ON scalars (key, t, value);
      CREATE TABLE IF NOT EXISTS events (t INTEGER, kind TEXT, payload TEXT, wall REAL);
      CREATE TABLE IF NOT EXISTS tags (t INTEGER, key TEXT, value TEXT, wall REAL);
      CREATE TABLE IF NOT EXISTS windows (
          measurements TEXT, window INTEGER, t0 INTEGER, t1 INTEGER, spans INTEGER, groups INTEGER, file TEXT, raw TEXT, wall REAL
      );
      CREATE INDEX IF NOT EXISTS windows_by_number ON windows (measurements, window);
      CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, kind TEXT, payload TEXT, wall REAL, status TEXT, handled REAL);
      CREATE TABLE IF NOT EXISTS progress (key TEXT PRIMARY KEY, value INTEGER);
      CREATE TABLE IF NOT EXISTS spans (measurements TEXT, window INTEGER, t0 INTEGER, steps INTEGER);
      CREATE TABLE IF NOT EXISTS groups (measurements TEXT, window INTEGER, t0 INTEGER, steps INTEGER);
      """

   .. raw:: html

      </details>



   SQL creating the tables of ``index.sqlite``.

   A scalar names its series by the id of a row of ``keys``. ``spans`` holds one row per span a set
   of measurements recorded. ``groups`` holds one row per group written, from its first step
   recorded. A span or group whose ``window`` is -1 was listed but lost: its window was being
   filled when the process ended. The ``raw`` column of ``windows`` lists the raw streams of a
   window as a JSON list.

.. py:data:: MAX_STEPS
   :value: 4611686018427387904


   Most steps a request can ask the recorder of a run to record. The steps stay within int64.

.. py:data:: SCHEDULER_VARIABLES
   :value: ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_JOB_NAME',...


   Environment variables of cluster schedulers written to ``run.json`` when set.

.. py:data:: TEMPORARY
   :value: '.creating'


   Suffix of the directory of a run being created. The directory is ``.<name>.<id>.creating``, next
   to the runs.

.. py:function:: now()

   Returns the local time in ISO 8601, to the second, with its UTC offset.


.. py:function:: timestamp(text)

   Returns a time written by `now` as a POSIX timestamp.

   Times written on hosts of other time zones compare correctly. Raises a ValueError or a
   TypeError when ``text`` is not such a time.


.. py:function:: to_json(data, **kwargs)

   Encodes ``data`` as JSON, NumPy and JAX values included.

   Keyword arguments are passed to `json.dumps`.


.. py:function:: from_json(text, kind, default)

   Decodes ``text`` as JSON when it holds a ``kind``, such as a dict, and returns ``default``
   otherwise, or for an empty or invalid ``text``.


.. py:function:: write_json(path, data)

   Writes ``data`` to ``path`` as indented JSON.

   The file is written to a temporary name, flushed to disk and moved into place.


.. py:function:: write_arrays(path, arrays)

   Writes ``arrays`` to ``path`` as an uncompressed ``.npz`` file.

   Creates the parent directories. The file is written to a temporary name, flushed to disk and
   moved into place.


.. py:function:: remove_partial(directory)

   Removes the temporary files left under ``directory`` by writes a process did not finish.


.. py:function:: git_state(run_dir)

   Returns the commit, branch and uncommitted changes of the repository of the working
   directory.

   Uncommitted changes to tracked files are written to ``git.patch`` in the run.

   :param run_dir: Directory of the run, where ``git.patch`` is written.
   :type run_dir: pathlib.Path

   :returns: ``sha``, ``branch``, ``dirty``, ``patch`` (the name of the patch file, or None) and
             ``untracked`` (at most `SETTINGS.git_untracked` untracked files). None outside a repository,
             or when ``git`` cannot be run.
   :rtype: dict or None


.. py:function:: environment()

   Returns the versions, devices, host, command line and scheduler job of the process.


.. py:function:: new_run_dir(root, name, run_id = None)

   Creates the directory of a new run under a temporary name.

   `move_run_dir` moves it into place once the run is written. Temporary directories left under
   ``root`` for more than `SETTINGS.creation_expires_after` seconds are removed.

   :param root: Directory of the runs.
   :type root: str or path-like
   :param name: Name of the run, used without ``run_id``. Characters other than letters, digits, ``_``,
                ``.`` and ``-`` are replaced by ``-``. An empty name gives ``run``.
   :type name: str
   :param run_id: Name of the final directory. By default, ``<date>-<time>_<name>_<id>`` with a random
                  ``<id>``.
   :type run_id: str, optional

   :returns: * **temporary** (*pathlib.Path*) -- Absolute path of the directory created.
             * **final** (*pathlib.Path*) -- Absolute path the run is moved to.

   :raises ValueError: When ``run_id`` is not a string of letters, digits, ``_``, ``.`` and ``-`` starting with
       a letter or a digit.
   :raises FileExistsError: When the directory of ``run_id`` exists.


.. py:function:: move_run_dir(temporary, final)

   Moves the directory of a new run into place.

   :param temporary: Paths given by `new_run_dir`.
   :type temporary: pathlib.Path
   :param final: Paths given by `new_run_dir`.
   :type final: pathlib.Path

   :returns: ``final``.
   :rtype: pathlib.Path

   :raises FileExistsError: When another run took ``final`` meanwhile.


.. py:function:: is_temporary(path)

   Returns whether ``path`` is the directory of a run not moved into place yet.


.. py:function:: window_file(measurements, number)

   Returns the path of a window file within its run, ``windows/<measurements>/<number>.npz``.


.. py:function:: window_files(run_dir)

   Returns the window files of a run on disk, as ``(measurements, number)``.

   Listed or not in the index.


.. py:function:: checkpoint_file(run_dir, step)

   Returns the file of the checkpoint of a run at ``step``, ``checkpoints/<step>.spark``.


.. py:function:: checkpoint_steps(run_dir)

   Returns the steps of the checkpoints of a run, in increasing order.

   A checkpoint is written aside and moved in place once complete, so a file
   ``checkpoints/<step>.spark`` is a complete checkpoint.


.. py:function:: read_info(run_dir)

   Reads ``run.json`` of a run.

   :raises ValueError: When the file is not valid JSON or does not hold a JSON object.


.. py:function:: heartbeat_age(info)

   Returns the time since the last heartbeat of a run and the time between heartbeats.

   :param info: Contents of ``run.json``.
   :type info: dict

   :returns: * **age** (*float*) -- Seconds since the last heartbeat. Infinity when ``heartbeat`` cannot be read.
             * **period** (*float*) -- Seconds between heartbeats, 5 when ``heartbeat_every`` is missing. Zero when
               ``heartbeat`` cannot be read.


.. py:function:: process()

   Returns the host and process id of this process, as written to ``run.json``.


.. py:function:: abandoned(run_dir)

   Returns whether the process writing a run marked as running is gone.

   On POSIX, a process of this host is gone when it has ended. Any other process is gone when the
   heartbeat of the run is older than both `SETTINGS.abandoned_after` seconds and
   `SETTINGS.crashed_after` heartbeat periods. A run with another status, such as a finished run
   being resumed, or whose ``run.json`` cannot be read, is not abandoned.


.. py:function:: connect(run_dir)

   Opens the index of a run for writing, creating its tables.

   :raises ValueError: When the index was written by another version of the tables.

   .. rubric:: Notes

   The index uses a rollback journal, deleted after each commit. A finished run is then a
   single file, readable from a read-only location or another host. A commit waits for the
   readers reading at that moment. Pages are not written before the commit (``cache_spill``),
   and no statement but the commit waits for readers. Statements wait `SETTINGS.busy_timeout`
   seconds for a busy index.


.. py:function:: connect_read_only(run_dir)

   Opens the index of a run for reading.

   The journal of a commit left unfinished by a process that ended is rolled back first. That
   happens when the process is gone (`abandoned`), no recorder holds the lock of the run and
   its directory can be written. Otherwise the index is opened as immutable and read as it
   stands, with a warning, and the journal is left for `Recorder.resume` to roll back.

   :raises sqlite3.OperationalError: When the index is locked, or cannot be opened for a reason other than an unfinished
       commit.


.. py:function:: commit(connection, run_dir)

   Commits ``connection``, waiting as long as another process holds the index.

   Warns each time the busy timeout of the connection runs out, every `SETTINGS.busy_timeout`
   seconds for a connection of `connect`. A reader in the middle of a long query holds the index.


.. py:function:: write_request(run_dir, kind, payload)

   Writes a request to the recorder of a run as a file of ``requests/``.

   The recorder reads the file, lists the request in the index and removes the file. Only the
   recorder writes the index.

   :param run_dir: Directory of the run.
   :type run_dir: pathlib.Path
   :param kind: Kind of the request, such as ``'record'``.
   :type kind: str
   :param payload: Arguments of the request, written as JSON.
   :type payload: dict

   :returns: Id of the request, the time in nanoseconds followed by a random suffix. It orders the
             requests by the time they were written.
   :rtype: str


.. py:function:: pending_requests(run_dir)

   Returns the files of the requests not yet read by the recorder, oldest first.

   Empty when ``requests/`` is missing or cannot be read.


.. py:function:: read_request(path)

   Reads a request written by `write_request`.

   :returns: ``id``, from the file name, with ``kind``, ``payload`` and ``wall``. A field missing or
             of the wrong type reads as None, or as an empty payload.
   :rtype: dict

   :raises OSError: When the file cannot be read.


.. py:function:: blank_request(path)

   Returns the request of a file that cannot be read: its id, as `read_request` gives it, and no
   kind, payload or time.


.. py:function:: lock(run_dir, wait = 0.0)

   Takes the exclusive lock of a run.

   The recorder writing a run holds its lock until it closes. The lock is taken on
   ``run.lock``, created when missing.

   :param run_dir: Directory of the run.
   :type run_dir: pathlib.Path
   :param wait: Seconds to keep trying while another process holds the lock.
   :type wait: float, default 0.0

   :returns: The open lock file, for `unlock`. None when another process holds the lock. Where the
             file system has no locks, the file is returned unlocked, with a warning. Where the
             platform has no locks, the file is returned unlocked.
   :rtype: file or None


.. py:function:: unlock(file)

   Releases a lock taken by `lock` and closes its file.


.. py:function:: locked(run_dir)

   Returns whether a recorder holds the lock of a run.

   None when it cannot be told, as when the run has no lock file, the file cannot be opened or
   the platform has no locks.


