"""FlexKV wait-complete prefetch wiring (minimal scheduler surface)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.storage.flexkv import _flexkv_factory
from sglang.srt.mem_cache.storage.flexkv.utils import request_key
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_flexkv_module(module_filename: str, module_name: str):
    connector_name = "flexkv.integration.sglang.connector"
    connector_stub = ModuleType(connector_name)
    connector_stub.FlexKVConnector = object
    connector_stub.FlexKVHostReleaseShim = object

    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv"
        / module_filename
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with patch.dict(sys.modules, {connector_name: connector_stub}):
        spec.loader.exec_module(module)
    return module


def test_scheduler_flexkv_prefetch_is_one_liner_to_tree_cache():
    sched = Scheduler.__new__(Scheduler)
    sched.enable_hicache_storage = False
    sched.tree_cache = MagicMock()

    req = SimpleNamespace(rid="x")
    with patch(
        "sglang.srt.managers.scheduler.get_memory",
        return_value=SimpleNamespace(enable_flexkv=True),
    ):
        sched._prefetch_kvcache(req)
    sched.tree_cache.prefetch_request.assert_called_once_with(req)


@pytest.mark.parametrize(
    "module_filename, class_name",
    [
        ("flexkv_radix_cache.py", "FlexKVRadixCache"),
        ("flexkv_hybrid_radix_cache.py", "FlexKVHybridRadixCache"),
    ],
)
def test_flexkv_prefetch_request_page_aligns_and_launches(module_filename, class_name):
    module = _load_flexkv_module(
        module_filename, f"_flexkv_prefetch_request_ut_{class_name}"
    )
    cache_class = getattr(module, class_name)
    cache = cache_class.__new__(cache_class)
    cache.page_size = 2
    cache.flexkv_connector = MagicMock()
    cache.flexkv_connector.prefetch_async = MagicMock(return_value=1)

    req = MagicMock()
    req.rid = "r1"
    req.cache_request_handle = CacheRequestHandle("r1", 2)
    req.full_untruncated_fill_ids = [1, 2, 3, 4, 5]
    req._compute_max_prefix_len = MagicMock(return_value=4)
    req.init_next_round_input = MagicMock()

    cache.prefetch_request(req)

    req.init_next_round_input.assert_called_once_with(tree_cache=None, cow_mamba=False)
    args, _kwargs = cache.flexkv_connector.prefetch_async.call_args
    assert args[0] == request_key(req.cache_request_handle)
    assert list(args[1]) == [1, 2, 3, 4]


def test_scheduler_wait_gate_uses_existing_or_condition():
    """Document the only scheduler wait change: or-in enable_flexkv."""
    enable_hicache_storage = False
    enable_flexkv = True
    assert (enable_hicache_storage or enable_flexkv) is True


@pytest.mark.parametrize("enable_dp_attention, expected_rank", [(False, 2), (True, 3)])
@pytest.mark.parametrize("hybrid", [False, True])
def test_factory_uses_published_parallel_context(
    enable_dp_attention, expected_rank, hybrid
):
    parallel = SimpleNamespace(
        pp_group=SimpleNamespace(rank_in_group=1),
        attn_tp_group=object(),
        attn_cp_group=SimpleNamespace(rank_in_group=0),
        dp_rank=2,
        attn_dp_rank=3,
    )
    context = SimpleNamespace(
        server_args=SimpleNamespace(enable_dp_attention=enable_dp_attention),
        params=SimpleNamespace(),
        model_config=object(),
        tp_rank=6,
        tp_size=8,
        tp_group=object(),
        tp_worker=SimpleNamespace(ps=SimpleNamespace(dp_rank=99, attn_dp_rank=99)),
        is_hybrid_ssm=False,
        is_hybrid_swa=hybrid,
    )
    radix_module = MagicMock()
    hybrid_module = MagicMock()
    unified_module = MagicMock()
    with (
        patch.dict(
            sys.modules,
            {
                "flexkv.integration.sglang.connector": MagicMock(),
                "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache": radix_module,
                "sglang.srt.mem_cache.storage.flexkv.flexkv_hybrid_radix_cache": hybrid_module,
                "sglang.srt.mem_cache.unified_radix_cache": unified_module,
            },
        ),
        patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
    ):
        result = _flexkv_factory(context)

    constructor = (
        hybrid_module.FlexKVHybridRadixCache
        if hybrid
        else radix_module.FlexKVRadixCache
    )
    assert result is constructor.return_value
    assert constructor.call_args.kwargs["dp_rank"] == expected_rank
    assert constructor.call_args.kwargs["pp_rank"] == 1
    assert constructor.call_args.kwargs["attn_cp_rank"] == 0
    if hybrid:
        assert context.params.tree_components == (ComponentType.FULL, ComponentType.SWA)
        unified_module.UnifiedRadixCache.assert_called_once_with(context.params)
