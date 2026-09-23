"""Bind request namespaces to FlexKV's existing KVManager API."""

from contextvars import ContextVar
from functools import partial
from typing import Optional

from sglang.srt.mem_cache.utils import storage_namespace_seed


def cache_namespace(
    extra_key: Optional[str], cache_salt: Optional[str]
) -> Optional[list[str]]:
    seed = storage_namespace_seed(extra_key, cache_salt)
    return [seed] if seed is not None else None


class _NamespacedKVManager:
    def __init__(self, manager, namespace):
        self._manager = manager
        self._namespace = namespace

    def __getattr__(self, name):
        value = getattr(self._manager, name)
        if name in ("get_match", "put_match", "prefetch_async"):
            return partial(value, namespace=self._namespace.get())
        return value


class NamespacedFlexKVConnector:
    def __init__(self, connector):
        self._connector = connector
        self._namespace = ContextVar("flexkv_namespace", default=None)
        if connector.kv_manager is not None:
            connector.kv_manager = _NamespacedKVManager(
                connector.kv_manager, self._namespace
            )

    def __getattr__(self, name):
        return getattr(self._connector, name)

    def _call(self, name, namespace, *args, **kwargs):
        token = self._namespace.set(namespace)
        try:
            return getattr(self._connector, name)(*args, **kwargs)
        finally:
            self._namespace.reset(token)

    def lookup_kv(self, *args, namespace=None, **kwargs):
        return self._call("lookup_kv", namespace, *args, **kwargs)

    def store_kv(self, *args, namespace=None, **kwargs):
        return self._call("store_kv", namespace, *args, **kwargs)

    def prefetch_async(self, *args, namespace=None, **kwargs):
        return self._call("prefetch_async", namespace, *args, **kwargs)
