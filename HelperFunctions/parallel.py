"""Ordered threading for independent ticker and event tasks."""
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral


def parallel_map(function, items, max_workers=None):
    if max_workers is not None and (
        not isinstance(max_workers, Integral) or isinstance(max_workers, bool)
        or max_workers < 1
    ):
        raise ValueError('max_workers must be a positive integer')
    items = list(items)
    if not items:
        return []
    if len(items) == 1 or max_workers == 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(function, items))
