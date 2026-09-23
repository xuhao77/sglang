from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import pytest

from sglang.srt.mem_cache.storage.flexkv.namespace import (
    NamespacedFlexKVConnector,
    cache_namespace,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Connector:
    def __init__(self, manager):
        self.kv_manager = manager

    def lookup_kv(self, token_ids, token_mask):
        return self.kv_manager.get_match(token_ids=token_ids, token_mask=token_mask)

    def store_kv(self, token_ids, token_mask):
        return self.kv_manager.put_match(token_ids=token_ids, token_mask=token_mask)

    def prefetch_async(self, token_ids, token_mask):
        return self.kv_manager.prefetch_async(token_ids=token_ids)


@pytest.mark.parametrize(
    "connector_method, manager_method",
    [
        ("lookup_kv", "get_match"),
        ("store_kv", "put_match"),
        ("prefetch_async", "prefetch_async"),
    ],
)
def test_connector_scopes_each_native_manager_call(connector_method, manager_method):
    manager = Mock()
    raw_connector = _Connector(manager)
    connector = NamespacedFlexKVConnector(raw_connector)
    token_ids = [1, 2, 3, 4]
    token_mask = [True] * 4
    native_call = getattr(manager, manager_method)

    for namespace in (
        cache_namespace(None, "tenant-a"),
        None,
        cache_namespace(None, "tenant-b"),
    ):
        result = getattr(connector, connector_method)(
            token_ids, token_mask, namespace=namespace
        )
        assert result is native_call.return_value
        assert native_call.call_args.kwargs["namespace"] == namespace
        assert native_call.call_args.kwargs["token_ids"] is token_ids

    getattr(raw_connector, connector_method)(token_ids, token_mask)
    assert native_call.call_args.kwargs["namespace"] is None


def test_namespace_is_reset_after_connector_failure():
    manager = Mock()
    manager.get_match.side_effect = RuntimeError("lookup failed")
    raw_connector = _Connector(manager)
    connector = NamespacedFlexKVConnector(raw_connector)

    with pytest.raises(RuntimeError, match="lookup failed"):
        connector.lookup_kv([1], [True], namespace=["tenant"])

    manager.get_match.side_effect = None
    raw_connector.lookup_kv([1], [True])
    assert manager.get_match.call_args.kwargs["namespace"] is None


def test_concurrent_calls_do_not_share_namespace():
    manager = Mock()
    manager.get_match.side_effect = lambda **kwargs: kwargs["namespace"]
    raw_connector = _Connector(manager)
    connector = NamespacedFlexKVConnector(raw_connector)
    barrier = Barrier(2)

    def lookup(token_ids, token_mask):
        barrier.wait(timeout=5)
        return raw_connector.kv_manager.get_match(
            token_ids=token_ids, token_mask=token_mask
        )

    raw_connector.lookup_kv = lookup
    namespaces = [cache_namespace(None, "tenant-a"), cache_namespace(None, "tenant-b")]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(connector.lookup_kv, [1], [True], namespace=namespace)
            for namespace in namespaces
        ]
        assert [future.result(timeout=10) for future in futures] == namespaces


def test_namespace_encoding_preserves_identity_boundaries():
    identities = [
        (None, "a\0b"),
        ("a", "b"),
        ("a\0b", "c"),
        ("a", "b\0c"),
        ("租户", "盐"),
    ]
    namespaces = [cache_namespace(*identity) for identity in identities]

    assert cache_namespace(None, None) is None
    assert all(namespace is not None for namespace in namespaces)
    assert len({tuple(namespace) for namespace in namespaces}) == len(identities)
    assert cache_namespace(*identities[0]) == namespaces[0]


def test_non_matching_methods_are_forwarded():
    manager = Mock()
    raw_connector = _Connector(manager)
    raw_connector.shutdown = Mock()
    connector = NamespacedFlexKVConnector(raw_connector)

    connector.shutdown()
    connector.kv_manager.cancel([17])

    raw_connector.shutdown.assert_called_once_with()
    manager.cancel.assert_called_once_with([17])
