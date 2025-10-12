import json
import logging
import pickle
import tempfile
from typing import Any

from simplekv.fs import KeyValueStore


log = logging.getLogger(__name__)


class BytesCacheEntry:
    def __init__(self, cache: KeyValueStore, key: str):
        self._cache = cache
        self._key = key
        self._have_data = key in cache
        log.debug(f"entry %s: %s in cache", key, "already" if self._have_data else "NOT")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None and not self._have_data:
            raise Exception(f"No data to write to cache for key {self._key}")

    def _validate_put(self, allow_replace: bool):
        if self._have_data and not allow_replace:
            raise Exception("Data already in cache")

    def has_data(self) -> bool:
        return self._have_data

    def get_bytes(self) -> bytes:
        return self._cache.get(self._key)

    def put_bytes(self, data: bytes, allow_replace: bool = False):
        self._validate_put(allow_replace)

        log.debug(f"entry %s: storing %d bytes in cache", self._entry._key, len(data))
        self._cache.put(self._key, data)
        self._have_data = True


class TemporaryFileForEntry:
    def __init__(self, entry: BytesCacheEntry, allow_replace: bool, mode="wb"):
        entry._validate_put(allow_replace)

        self._entry = entry
        self._temp_file = tempfile.NamedTemporaryFile(mode, delete=True, delete_on_close=False)

    def __enter__(self):
        return self._temp_file.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            log.debug(f"entry %s: storing %d bytes in cache", self._entry._key, self._temp_file.tell())
            self._temp_file.close()
            # note: this will *move* the file if possible, avoiding the need for another copy either in RAM or on disk
            self._entry._cache.put_file(self._entry._key, self._temp_file.name)
            self._entry._have_data = True
        else:
            # an error occurred, do not put the file in the cache
            pass

        self._temp_file.__exit__(exc_type, exc_value, traceback)


class CacheEntry(BytesCacheEntry):
    def __init__(self, cache: KeyValueStore, key: str, type: str):
        """
        Method specifies the serialization method to use.
        """
        assert type in ["bytes", "json", "numpy", "pickle", "str"], "Unsupported serialization method"

        super().__init__(cache, key + self._get_suffix(type))
        self._type = type

    def _get_suffix(self, type: str) -> str:
        match type:
            case "bytes":
                return ""
            case "json":
                return ".json"
            case "numpy":
                return ".npy"
            case "pickle":
                return ".pickle"
            case "str":
                return ".txt"

    def get(self) -> Any:
        with self._cache.open(self._key) as f:
            match self._type:
                case "bytes":
                    return f.read()
                case "json":
                    return json.load(f)
                case "numpy":
                    import numpy as np
                    return np.load(f)
                case "pickle":
                    return pickle.load(f)
                case "str":
                    return f.read().decode()

    def put(self, obj: Any, allow_replace: bool = False) -> None:
        match self._type:
            case "bytes":
                assert isinstance(obj, bytes)
                self.put_bytes(obj)
            case "json":
                with TemporaryFileForEntry(self, allow_replace, mode="wt") as f:
                    json.dump(obj, f)
            case "numpy":
                import numpy as np

                with TemporaryFileForEntry(self, allow_replace) as f:
                    np.save(f, obj)
            case "pickle":
                with TemporaryFileForEntry(self, allow_replace) as f:
                    pickle.dump(obj, f)
            case "str":
                assert isinstance(obj, str)
                self.put_bytes(obj.encode())


def make_key(*args, version: int = None) -> str:
    """
    Create a valid cache key from arbitrary arguments.
    The arguments must be hashable.
    An optional version number can be provided to invalidate old cache entries when there is a change in the computation.
    """
    if len(args) == 0:
        raise ValueError("At least one argument is required to make a cache key")

    if isinstance(args[0], str):
        key = args[0] + "_"
        args = args[1:]
    else:
        key = ""

    the_hash = hash(args)
    key += f"{the_hash:x}"

    if version is not None:
        key += f"_v{version}"

    log.debug("make_key: input %s, hash %s", args, key)
    return key
