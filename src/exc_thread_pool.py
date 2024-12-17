from concurrent.futures import Executor, Future, ThreadPoolExecutor
import contextlib
from asyncio.base_events import timeouts
import pytest
import signal
import threading
import pytest
import time
import queue
from collections import deque
from typing import Any, Callable, Coroutine
import asyncio

import uvloop


class LoopManager:
    def __init__(self, loop=None):
        if loop:
            self.loop = loop
        else:
            self.loop = asyncio.get_event_loop()
        self.tasks: set[asyncio.Task] = set()
        self._shutdown = asyncio.Event()

    def submit(self, coroutine: Coroutine) -> asyncio.Task:
        if self._shutdown.is_set():
            raise ValueError("Can't submit jobs to a shutdown loop.")
        task = self.loop.create_task(coroutine)
        task.add_done_callback(self._on_completion)
        self.tasks.add(task)
        return task

    def _on_completion(self, task):
        if task.exception():
            raise task.exception()
        self.tasks.remove(task)

    def shutdown(self, timeout: int | None = None):
        self._shutdown.set()
        for task in self.tasks:
            task.cancel()

        wait_task = asyncio.gather(*self.tasks, return_exceptions=True)
        try:
            self.loop.run_until_complete(wait_task)
        except Exception:
            pass

        self.loop.stop()
        timeouts = 3
        err = None
        while timeouts:
            try:
                self.loop.close()
            except RuntimeError as e:
                timeouts -= 1
                time.sleep(0.1)
        if not timeouts:
            raise RuntimeError("Failed to close loop")

    def add_signal_handlers(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.loop.add_signal_handler(sig, self.shutdown)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


@pytest.fixture
def event_loop_policy():
    return uvloop.EventLoopPolicy()


@pytest.mark.asyncio
async def test_loop_man():
    async def coro():
        return 1

    with LoopManager() as loop:
        result = await loop.submit(coro())
        assert result == 1


class PoolManager:
    def __init__(
        self,
        pool: Executor,
        error_callback: Callable[[Exception], Any] | None = None,
        timeout=None,
    ):
        self.pool = pool
        self.queue: deque[Future] = deque()
        self.timeout = timeout

    def submit(self, f: Callable, *args, **kwargs) -> None:
        future = self.pool.submit(f, *args, **kwargs)
        future.add_done_callback(self.check_for_err)
        self.queue.append(future)

    def check_for_err(self, task):
        if task.exception():
            raise task.exception()

    def __iter__(self):
        return self

    def __next__(self):
        if self.queue:
            vnext = self.queue.pop()
            if vnext.exception():
                raise vnext.exception()
            return vnext.result()
        else:
            raise StopIteration

    def shutdown(self):
        for v in self.queue:
            v.cancel()


def queue_iterator(q: queue.Queue, timeout=None):
    while (v := q.get(timeout=timeout)) is not None:
        yield v
    q.shutdown()


def test_ethreadpool():
    def exc_callback(e: Exception):
        raise e

    def task():
        raise ValueError()

    pool = ThreadPoolExecutor()
    man = PoolManager(pool, exc_callback)
    for i in range(6):
        man.submit(task)

    with pytest.raises(ValueError):
        for j, v in enumerate(man):
            print(j, v)
