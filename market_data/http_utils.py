"""HTTP 辅助：东方财富友好请求头 + 瞬时错误重试。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import time
from typing import TypeVar

import requests

T = TypeVar("T")

DEFAULT_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


@contextmanager
def eastmoney_friendly_requests_get():
    """为 AkShare 访问东方财富时补丁 requests.get（User-Agent / Referer / 超时）。"""
    orig = requests.get

    def get(*args: object, **kwargs: object):
        headers = dict(kwargs.pop("headers", None) or {})
        headers.setdefault(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        )
        headers.setdefault("Referer", "https://quote.eastmoney.com/")
        headers.setdefault("Accept", "*/*")
        kwargs["headers"] = headers
        to = kwargs.get("timeout", 15)
        if isinstance(to, (int, float)):
            kwargs["timeout"] = max(float(to), 45.0)
        elif isinstance(to, tuple) and len(to) == 2:
            kwargs["timeout"] = (max(float(to[0]), 45.0), max(float(to[1]), 90.0))
        return orig(*args, **kwargs)

    requests.get = get  # type: ignore[method-assign]
    try:
        yield
    finally:
        requests.get = orig  # type: ignore[method-assign]


def retry_transient(
    fn: Callable[[], T],
    *,
    retries: int = 5,
    retry_base_sleep_s: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = DEFAULT_TRANSIENT_EXCEPTIONS,
    on_retry: Callable[[BaseException, int], None] | None = None,
) -> T:
    """对瞬时网络错误做指数退避重试。"""
    last: BaseException | None = None
    for attempt in range(retries):
        try:
            return fn()
        except exceptions as e:
            last = e
            if attempt >= retries - 1:
                raise
            wait = retry_base_sleep_s * (2**attempt)
            if on_retry:
                on_retry(e, attempt)
            else:
                print(f"网络请求失败（{type(e).__name__}），{wait:.1f}s 后重试 ({attempt + 1}/{retries})…")
            time.sleep(wait)
    assert last is not None
    raise last
