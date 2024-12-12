from concurrent.futures import Executor, Future, ThreadPoolExecutor
import contextlib
import signal
import threading
from dspy import asyncify
import pytest
import time
import queue
from collections import deque
from typing import Any, Callable, Coroutine
import asyncio

import uvloop


def check_for_err(future: Future | asyncio.Task):
    if future.exception():
        raise future.exception()


class LoopManager:
    def __init__(self):
        self.loop = uvloop.new_event_loop()
        self.tasks: set[asyncio.Task] = set()
        self._shutdown = asyncio.Event()

    def submit(self, coroutine: Coroutine) -> asyncio.Task:
        if self._shutdown.is_set():
            raise ValueError("Can't submit jobs to a shutdown loop.")
        task = self.loop.create_task(self.wrapper(coroutine))
        task.add_done_callback(check_for_err)
        self.tasks.add(task)
        return task

    async def wrapper(self, coroutine: Coroutine) -> Any:
        try:
            return await coroutine
        finally:
            self.tasks.remove(coroutine)

    async def shutdown(self, timeout: int | None = None):
        self._shutdown.set()
        for task in self.tasks:
            task.cancel()

        await asyncio.wait_for(
            asyncio.gather(*self.tasks, return_exceptions=True), timeout=timeout
        )
        self.loop.stop()

    def add_signal_handlers(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.loop.add_signal_handler(
                sig, lambda s: self.loop.run_until_complete(self.shutdown())
            )

    @contextlib.asynccontextmanager
    async def run(self):
        self.add_signal_handlers()
        try:
            yield self
        finally:
            await self.shutdown()


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

    def __iter__(self):
        return self

    def __next__(self):
        return self.collect()

    def collect(self) -> Any:
        return self.queue.pop()

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
