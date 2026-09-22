from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle


def request_key(handle: CacheRequestHandle) -> str:
    return f"{handle.rid}:{handle.attempt_id}"
