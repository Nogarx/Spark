#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

from __future__ import annotations
import typing as tp

import os
import re
import sys
import json
import time
import uuid
import errno
import shutil
import socket
import sqlite3
import pathlib
import datetime
import platform
import warnings
import subprocess
import numpy as np

from spark.recording.utils import is_name
from spark.recording.settings import SETTINGS

try:
    import fcntl
except ImportError:
    fcntl = None
try:
    import msvcrt
except ImportError:
    msvcrt = None

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

SCHEMA_VERSION = 4
"""
    Version of the tables of ``index.sqlite``, stored as its ``user_version``.
"""

SCHEMA = """
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
"""
    SQL creating the tables of ``index.sqlite``.

    A scalar names its series by the id of a row of ``keys``. ``spans`` holds one row per span a set
    of measurements recorded. ``groups`` holds one row per group written, from its first step
    recorded. A span or group whose ``window`` is -1 was listed but lost: its window was being
    filled when the process ended. The ``raw`` column of ``windows`` lists the raw streams of a
    window as a JSON list.
"""

MAX_STEPS = 2 ** 62
"""
    Most steps a request can ask the recorder of a run to record. The steps stay within int64.
"""

SCHEDULER_VARIABLES = (
    'SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_JOB_NAME', 'SLURM_JOB_PARTITION',
    'SLURM_JOB_NODELIST', 'SLURM_NNODES', 'SLURM_NTASKS', 'SLURM_PROCID', 'SLURM_RESTART_COUNT',
    'PBS_JOBID', 'PBS_ARRAY_INDEX', 'LSB_JOBID', 'LSB_JOBINDEX', 'JOB_ID', 'SGE_TASK_ID',
)
"""
    Environment variables of cluster schedulers written to ``run.json`` when set.
"""

TEMPORARY = '.creating'
"""
    Suffix of the directory of a run being created. The directory is ``.<name>.<id>.creating``, next
    to the runs.
"""

_HELD = (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK, errno.EDEADLK)
"""
    Error numbers of a lock held by another process.
"""

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def now() -> str:
    """
        Returns the local time in ISO 8601, to the second, with its UTC offset.
    """
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def timestamp(text: str) -> float:
    """
        Returns a time written by `now` as a POSIX timestamp.

        Times written on hosts of other time zones compare correctly. Raises a ValueError or a
        TypeError when ``text`` is not such a time.
    """
    return datetime.datetime.fromisoformat(text).timestamp()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _json_default(value: tp.Any) -> tp.Any:
    """
        Converts NumPy and JAX numbers and arrays to JSON numbers and lists.

        Any other value, or an array that cannot be converted, gives its string.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) or type(value).__module__.startswith(('jax', 'jaxlib')):
        try:
            return np.asarray(value).tolist()
        except Exception:
            pass
    return str(value)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def to_json(data: tp.Any, **kwargs: tp.Any) -> str:
    """
        Encodes ``data`` as JSON, NumPy and JAX values included.

        Keyword arguments are passed to `json.dumps`.
    """
    return json.dumps(data, default=_json_default, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def from_json(text: str | None, kind: type, default: tp.Any) -> tp.Any:
    """
        Decodes ``text`` as JSON when it holds a ``kind``, such as a dict, and returns ``default``
        otherwise, or for an empty or invalid ``text``.
    """
    try:
        data = json.loads(text) if text else default
    except ValueError:
        return default
    return data if isinstance(data, kind) else default

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _write(path: pathlib.Path, write: tp.Callable[[tp.IO], None]) -> None:
    """
        Writes ``path`` through ``write`` to a temporary file, flushed to disk and moved into place.

        The temporary file is ``.<name>.partial``, removed when writing fails.
    """
    temp = path.with_name(f'.{path.name}.partial')
    try:
        with open(temp, 'wb') as file:
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            temp.unlink()
        except OSError:
            pass
        raise
    _sync_directory(path.parent)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _sync_directory(directory: pathlib.Path) -> None:
    """
        Flushes the entries of a directory to disk, where the platform allows it.
    """
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def write_json(path: pathlib.Path, data: tp.Any) -> None:
    """
        Writes ``data`` to ``path`` as indented JSON.

        The file is written to a temporary name, flushed to disk and moved into place.
    """
    text = to_json(data, indent=2).encode()
    _write(path, lambda file: file.write(text))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def write_arrays(path: pathlib.Path, arrays: dict[str, np.ndarray]) -> None:
    """
        Writes ``arrays`` to ``path`` as an uncompressed ``.npz`` file.

        Creates the parent directories. The file is written to a temporary name, flushed to disk and
        moved into place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, lambda file: np.savez(file, **arrays))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def remove_partial(directory: pathlib.Path) -> None:
    """
        Removes the temporary files left under ``directory`` by writes a process did not finish.
    """
    for temp in directory.rglob('.*.partial'):
        try:
            temp.unlink()
        except OSError:
            pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def _git(cwd: pathlib.Path, *args: str) -> str | None:
    """
        Runs ``git`` with ``args`` in ``cwd`` and returns its output, or None when it fails.
    """
    try:
        result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, errors='replace', timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def git_state(run_dir: pathlib.Path) -> dict[str, tp.Any] | None:
    """
        Returns the commit, branch and uncommitted changes of the repository of the working
        directory.

        Uncommitted changes to tracked files are written to ``git.patch`` in the run.

        Parameters
        ----------
        run_dir : pathlib.Path
            Directory of the run, where ``git.patch`` is written.

        Returns
        -------
        dict or None
            ``sha``, ``branch``, ``dirty``, ``patch`` (the name of the patch file, or None) and
            ``untracked`` (at most `SETTINGS.git_untracked` untracked files). None outside a repository,
            or when ``git`` cannot be run.
    """
    cwd = pathlib.Path.cwd()
    sha = _git(cwd, 'rev-parse', 'HEAD')
    if sha is None:
        return None
    branch = _git(cwd, 'rev-parse', '--abbrev-ref', 'HEAD')
    diff = _git(cwd, 'diff', 'HEAD') or ''
    untracked = [line for line in (_git(cwd, 'ls-files', '--others', '--exclude-standard') or '').splitlines() if line]
    patch = None
    if diff:
        (run_dir / 'git.patch').write_text(diff)
        patch = 'git.patch'
    return {
        'sha': sha.strip(),
        'branch': branch.strip() if branch else None,
        'dirty': bool(diff),
        'patch': patch,
        'untracked': untracked[:SETTINGS.git_untracked],
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def environment() -> dict[str, tp.Any]:
    """
        Returns the versions, devices, host, command line and scheduler job of the process.
    """
    import jax
    versions = {'python': platform.python_version()}
    for name in ('jax', 'jaxlib', 'flax', 'numpy', 'optax'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', getattr(module, 'version', None))
        except ImportError:
            versions[name] = None
    try:
        from importlib.metadata import version
        versions['spark_snn'] = version('spark_snn')
    except Exception:
        versions['spark_snn'] = None
    return {
        'versions': versions,
        'devices': [f'{d.platform}:{getattr(d, "device_kind", "")}' for d in jax.devices()],
        'processes': jax.process_count(),
        'host': socket.gethostname(),
        'platform': platform.platform(),
        'argv': list(sys.argv),
        'cwd': str(pathlib.Path.cwd()),
        'xla_flags': os.environ.get('XLA_FLAGS'),
        'pid': os.getpid(),
        'scheduler': {k: os.environ[k] for k in SCHEDULER_VARIABLES if k in os.environ},
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def new_run_dir(root: str | os.PathLike, name: str, run_id: str | None = None) -> tuple[pathlib.Path, pathlib.Path]:
    """
        Creates the directory of a new run under a temporary name.

        `move_run_dir` moves it into place once the run is written. Temporary directories left under
        ``root`` for more than `SETTINGS.creation_expires_after` seconds are removed.

        Parameters
        ----------
        root : str or path-like
            Directory of the runs.
        name : str
            Name of the run, used without ``run_id``. Characters other than letters, digits, ``_``,
            ``.`` and ``-`` are replaced by ``-``. An empty name gives ``run``.
        run_id : str, optional
            Name of the final directory. By default, ``<date>-<time>_<name>_<id>`` with a random
            ``<id>``.

        Returns
        -------
        temporary : pathlib.Path
            Absolute path of the directory created.
        final : pathlib.Path
            Absolute path the run is moved to.

        Raises
        ------
        ValueError
            When ``run_id`` is not a string of letters, digits, ``_``, ``.`` and ``-`` starting with
            a letter or a digit.
        FileExistsError
            When the directory of ``run_id`` exists.
    """
    root = pathlib.Path(root).resolve()
    if run_id is not None:
        if not is_name(run_id):
            raise ValueError(f'Invalid run id "{run_id}". Expected letters, digits, "_", "." and "-", starting with a letter or a digit.')
        final = root / run_id
        if final.exists():
            raise FileExistsError(f'"{final}" exists.')
    else:
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        name = re.sub(r'[^A-Za-z0-9_.\-]', '-', str(name)) or 'run'
        final = root / f'{stamp}_{name}_{uuid.uuid4().hex[:8]}'
    temporary = root / f'.{final.name}.{uuid.uuid4().hex[:8]}{TEMPORARY}'
    temporary.mkdir(parents=True)
    # Runs whose creation a process ending interrupted; a run is created in seconds.
    for left in root.glob(f'.*{TEMPORARY}'):
        try:
            if left != temporary and time.time() - left.stat().st_mtime > SETTINGS.creation_expires_after:
                shutil.rmtree(left, ignore_errors=True)
        except OSError:
            pass
    return temporary, final

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def move_run_dir(temporary: pathlib.Path, final: pathlib.Path) -> pathlib.Path:
    """
        Moves the directory of a new run into place.

        Parameters
        ----------
        temporary, final : pathlib.Path
            Paths given by `new_run_dir`.

        Returns
        -------
        pathlib.Path
            ``final``.

        Raises
        ------
        FileExistsError
            When another run took ``final`` meanwhile.
    """
    try:
        if final.exists():
            raise FileExistsError(f'"{final}" exists.')
        os.rename(temporary, final)
    except OSError as error:
        if isinstance(error, FileExistsError) or error.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(f'"{final}" exists.') from None
        raise
    _sync_directory(final.parent)
    return final

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def is_temporary(path: pathlib.Path) -> bool:
    """
        Returns whether ``path`` is the directory of a run not moved into place yet.
    """
    return path.name.startswith('.') and path.name.endswith(TEMPORARY)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def window_file(measurements: str, number: int) -> pathlib.PurePosixPath:
    """
        Returns the path of a window file within its run, ``windows/<measurements>/<number>.npz``.
    """
    return pathlib.PurePosixPath('windows', measurements, f'{int(number):06d}.npz')

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def window_files(run_dir: pathlib.Path) -> set[tuple[str, int]]:
    """
        Returns the window files of a run on disk, as ``(measurements, number)``.

        Listed or not in the index.
    """
    root = run_dir / 'windows'
    if not root.is_dir():
        return set()
    return {
        (directory.name, int(file.stem))
        for directory in root.iterdir() for file in directory.glob('*.npz') if file.stem.isdigit()
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def checkpoint_file(run_dir: pathlib.Path, step: int) -> pathlib.Path:
    """
        Returns the file of the checkpoint of a run at ``step``, ``checkpoints/<step>.spark``.
    """
    return run_dir / 'checkpoints' / f'{int(step):012d}.spark'

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def checkpoint_steps(run_dir: pathlib.Path) -> list[int]:
    """
        Returns the steps of the checkpoints of a run, in increasing order.

        A checkpoint is written aside and moved in place once complete, so a file
        ``checkpoints/<step>.spark`` is a complete checkpoint.
    """
    root = run_dir / 'checkpoints'
    if not root.is_dir():
        return []
    return sorted(int(file.stem) for file in root.iterdir() if file.is_file() and file.suffix == '.spark' and file.stem.isdigit())

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def read_info(run_dir: pathlib.Path) -> dict[str, tp.Any]:
    """
        Reads ``run.json`` of a run.

        Raises
        ------
        ValueError
            When the file is not valid JSON or does not hold a JSON object.
    """
    info = json.loads((run_dir / 'run.json').read_text())
    if not isinstance(info, dict):
        raise ValueError(f'"{run_dir / "run.json"}" does not hold a JSON object.')
    return info

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def heartbeat_age(info: dict[str, tp.Any]) -> tuple[float, float]:
    """
        Returns the time since the last heartbeat of a run and the time between heartbeats.

        Parameters
        ----------
        info : dict
            Contents of ``run.json``.

        Returns
        -------
        age : float
            Seconds since the last heartbeat. Infinity when ``heartbeat`` cannot be read.
        period : float
            Seconds between heartbeats, 5 when ``heartbeat_every`` is missing. Zero when
            ``heartbeat`` cannot be read.
    """
    try:
        period = float(info.get('heartbeat_every', 5.0))
        return time.time() - timestamp(info['heartbeat']), period
    except (KeyError, TypeError, ValueError):
        return float('inf'), 0.0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def process() -> dict[str, tp.Any]:
    """
        Returns the host and process id of this process, as written to ``run.json``.
    """
    return {'host': socket.gethostname(), 'pid': os.getpid()}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def abandoned(run_dir: pathlib.Path) -> bool:
    """
        Returns whether the process writing a run marked as running is gone.

        On POSIX, a process of this host is gone when it has ended. Any other process is gone when the
        heartbeat of the run is older than both `SETTINGS.abandoned_after` seconds and
        `SETTINGS.crashed_after` heartbeat periods. A run with another status, such as a finished run
        being resumed, or whose ``run.json`` cannot be read, is not abandoned.
    """
    try:
        info = read_info(run_dir)
    except (OSError, ValueError):
        return False
    if info.get('status') != 'running':
        return False
    writer = info.get('process') or {}
    if os.name == 'posix' and writer.get('host') == socket.gethostname() and isinstance(writer.get('pid'), int):
        try:
            os.kill(writer['pid'], 0)                                   # signal 0: whether it exists
        except ProcessLookupError:
            return True
        except OSError:
            pass
        return False
    age, period = heartbeat_age(info)
    return age > max(SETTINGS.abandoned_after, SETTINGS.crashed_after * period)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def connect(run_dir: pathlib.Path) -> sqlite3.Connection:
    """
        Opens the index of a run for writing, creating its tables.

        Raises
        ------
        ValueError
            When the index was written by another version of the tables.

        Notes
        -----
        The index uses a rollback journal, deleted after each commit. A finished run is then a
        single file, readable from a read-only location or another host. A commit waits for the
        readers reading at that moment. Pages are not written before the commit (``cache_spill``),
        and no statement but the commit waits for readers. Statements wait `SETTINGS.busy_timeout`
        seconds for a busy index.
    """
    connection = sqlite3.connect(run_dir / 'index.sqlite', timeout=SETTINGS.busy_timeout)
    try:
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.execute('PRAGMA synchronous=FULL')
        connection.execute('PRAGMA cache_spill=OFF')
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            raise ValueError(f'The index of "{run_dir}" has version {version} of the tables; this version of Spark writes version {SCHEMA_VERSION}.')
        if version == 0:
            connection.executescript(SCHEMA)
            connection.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')
            commit(connection, run_dir)
        elif 'tag' not in {row[1] for row in connection.execute('PRAGMA table_info(keys)')}:
            # Written before the tags of the series were kept: resumed, they are kept from now on.
            connection.execute('ALTER TABLE keys ADD COLUMN tag TEXT')
            commit(connection, run_dir)
    except BaseException:
        connection.close()
        raise
    return connection

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def connect_read_only(run_dir: pathlib.Path) -> sqlite3.Connection:
    """
        Opens the index of a run for reading.

        The journal of a commit left unfinished by a process that ended is rolled back first. That
        happens when the process is gone (`abandoned`), no recorder holds the lock of the run and
        its directory can be written. Otherwise the index is opened as immutable and read as it
        stands, with a warning, and the journal is left for `Recorder.resume` to roll back.

        Raises
        ------
        sqlite3.OperationalError
            When the index is locked, or cannot be opened for a reason other than an unfinished
            commit.
    """
    path = run_dir / 'index.sqlite'
    uri = path.resolve().as_uri()
    for attempt in range(2):
        connection = sqlite3.connect(f'{uri}?mode=ro', uri=True, timeout=SETTINGS.read_timeout)
        try:
            connection.execute('PRAGMA user_version').fetchone()
            return connection
        except sqlite3.OperationalError as error:
            connection.close()
            if 'locked' in str(error) or 'readonly' not in str(error).replace('-', '').replace(' ', ''):
                raise
        writable = os.access(run_dir, os.W_OK) and os.access(path, os.W_OK)
        # A journal can look unfinished where locks do not reach across hosts, as when its writer is stalled in
        # the middle of a commit: it is rolled back only when that writer is known to be gone.
        if attempt or not writable or locked(run_dir) is not False or not abandoned(run_dir):
            break
        # Opening the index to write rolls back the journal left unfinished.
        connection = sqlite3.connect(path, timeout=SETTINGS.read_timeout)
        try:
            connection.execute('PRAGMA user_version').fetchone()
        finally:
            connection.close()
    warnings.warn(f'The index of "{run_dir.name}" is read as it stands: a commit left unfinished cannot be rolled back here.')
    return sqlite3.connect(f'{uri}?immutable=1', uri=True)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def commit(connection: sqlite3.Connection, run_dir: pathlib.Path) -> None:
    """
        Commits ``connection``, waiting as long as another process holds the index.

        Warns each time the busy timeout of the connection runs out, every `SETTINGS.busy_timeout`
        seconds for a connection of `connect`. A reader in the middle of a long query holds the index.
    """
    started = time.monotonic()
    while True:
        try:
            connection.commit()
            return
        except sqlite3.OperationalError as error:
            if 'locked' not in str(error) and 'busy' not in str(error):
                raise
            warnings.warn(
                f'The index of "{run_dir.name}" has been held by another process for {time.monotonic() - started:.0f} s; '
                f'the recorder waits for it. A reader that keeps a query open, as a cursor not read to its end, holds it.'
            )

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def write_request(run_dir: pathlib.Path, kind: str, payload: dict[str, tp.Any]) -> str:
    """
        Writes a request to the recorder of a run as a file of ``requests/``.

        The recorder reads the file, lists the request in the index and removes the file. Only the
        recorder writes the index.

        Parameters
        ----------
        run_dir : pathlib.Path
            Directory of the run.
        kind : str
            Kind of the request, such as ``'record'``.
        payload : dict
            Arguments of the request, written as JSON.

        Returns
        -------
        str
            Id of the request, the time in nanoseconds followed by a random suffix. It orders the
            requests by the time they were written.
    """
    request_id = f'{time.time_ns():020d}-{uuid.uuid4().hex[:8]}'
    directory = run_dir / 'requests'
    directory.mkdir(exist_ok=True)
    write_json(directory / f'{request_id}.json', {'id': request_id, 'kind': kind, 'payload': payload, 'wall': time.time()})
    return request_id

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def pending_requests(run_dir: pathlib.Path) -> list[pathlib.Path]:
    """
        Returns the files of the requests not yet read by the recorder, oldest first.

        Empty when ``requests/`` is missing or cannot be read.
    """
    directory = run_dir / 'requests'
    try:
        return sorted(p for p in directory.iterdir() if p.suffix == '.json' and not p.name.startswith('.'))
    except OSError:
        return []                                                       # none, or none readable for now

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def read_request(path: pathlib.Path) -> dict[str, tp.Any]:
    """
        Reads a request written by `write_request`.

        Returns
        -------
        dict
            ``id``, from the file name, with ``kind``, ``payload`` and ``wall``. A field missing or
            of the wrong type reads as None, or as an empty payload.

        Raises
        ------
        OSError
            When the file cannot be read.
    """
    data = from_json(path.read_text(), dict, {})
    payload = data.get('payload')
    return {
        'id': path.stem,
        'kind': data.get('kind') if isinstance(data.get('kind'), str) else None,
        'payload': payload if isinstance(payload, dict) else {},
        'wall': data.get('wall') if isinstance(data.get('wall'), (int, float)) else None,
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def blank_request(path: pathlib.Path) -> dict[str, tp.Any]:
    """
        Returns the request of a file that cannot be read: its id, as `read_request` gives it, and no
        kind, payload or time.
    """
    return {'id': path.stem, 'kind': None, 'payload': {}, 'wall': None}

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def lock(run_dir: pathlib.Path, wait: float = 0.0) -> tp.IO | None:
    """
        Takes the exclusive lock of a run.

        The recorder writing a run holds its lock until it closes. The lock is taken on
        ``run.lock``, created when missing.

        Parameters
        ----------
        run_dir : pathlib.Path
            Directory of the run.
        wait : float, default 0.0
            Seconds to keep trying while another process holds the lock.

        Returns
        -------
        file or None
            The open lock file, for `unlock`. None when another process holds the lock. Where the
            file system has no locks, the file is returned unlocked, with a warning. Where the
            platform has no locks, the file is returned unlocked.
    """
    file = open(run_dir / 'run.lock', 'a+b')
    deadline = time.monotonic() + wait
    while True:
        try:
            if fcntl is not None:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:
                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            return file
        except OSError as error:
            if error.errno not in _HELD:
                warnings.warn(
                    f'"{run_dir}" cannot be locked on this file system ({error}). Nothing stops two recorders from '
                    f'writing the run at once, and runs read as crashed only once their heartbeat stops.'
                )
                return file
            if time.monotonic() >= deadline:
                file.close()
                return None
            time.sleep(0.1)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def unlock(file: tp.IO) -> None:
    """
        Releases a lock taken by `lock` and closes its file.
    """
    try:
        if fcntl is not None:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    file.close()

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def locked(run_dir: pathlib.Path) -> bool | None:
    """
        Returns whether a recorder holds the lock of a run.

        None when it cannot be told, as when the run has no lock file, the file cannot be opened or
        the platform has no locks.
    """
    try:
        file = open(run_dir / 'run.lock', 'rb')
    except OSError:
        return None
    try:
        if fcntl is not None:
            fcntl.flock(file.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBRLCK, 1)
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            return None
        return False
    except OSError as error:
        return True if error.errno in _HELD else None
    finally:
        file.close()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
