"""One bounded producer for dataset batches; no process pool or scheduling service."""

import math
import platform
import queue
import threading
import time
import weakref


class Measurements:
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def observe(self, name, value):
        with self.lock:
            key = f"ewma_{name}"
            self.values[key] = .2*value + .8*self.values[key] if key in self.values else value
            count = f"{name}_samples"
            self.values[count] = self.values.get(count, 0)+1

    def update(self, **values):
        with self.lock:
            self.values.update(values)

    def maximum(self, name, value):
        with self.lock:
            self.values[name] = max(value, self.values.get(name, 0))

    def snapshot(self):
        with self.lock:
            values = dict(self.values)
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            values["process_peak_rss_mb"] = rss/(1024**2 if platform.system() == "Darwin" else 1024)
        except ImportError:
            values["process_peak_rss_mb"] = None
        return values


def training_step(metrics, milliseconds):
    value = float(milliseconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError("training step milliseconds must be finite and nonnegative")
    metrics.observe("training_step_ms", value)


def _payload_bytes(sample):
    labels = (sample.labels,) if hasattr(sample.labels, "nbytes") else sample.labels
    return sample.points.nbytes + sum(v.nbytes for v in labels if v is not None)


def _produce(source, state, metrics):
    try:
        while not state["stop"].is_set():
            if not state["slots"].acquire(timeout=.05):
                continue
            if state["stop"].is_set():
                break
            try:
                start = time.perf_counter()
                value = next(source)
                metrics.observe("generation_ms", (time.perf_counter()-start)*1000)
                size = _payload_bytes(value)
                if size > state["max_bytes"]:
                    raise MemoryError("producer exceeded its declared maximum batch payload")
                with state["lock"]:
                    state["bytes"] += size
                    state["queue"].put_nowait((value, size, None))
                    state["items"] += 1
                    metrics.maximum("queue_peak_bytes", state["bytes"])
                    metrics.maximum("queue_peak_batches", state["items"])
            except BaseException as error:
                state["queue"].put_nowait((None, 0, error))
                break
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()


def _close(state, thread):
    state["stop"].set()
    if thread is not None and thread is not threading.current_thread():
        thread.join()
    while True:
        try:
            state["queue"].get_nowait()
        except queue.Empty:
            break
    state["bytes"] = 0
    state["items"] = 0


class BatchStream:
    """Context-managed batches with an explicit bounded payload reservation.

    Producer query scratch uses its Engine budget. This separate budget covers
    prefetched NumPy payloads; batches handed to the consumer become its memory.
    Slots are reserved before generation, so a full queue has no extra pending
    batch. Closing waits for the current synchronous native call and drains it.
    """
    def __init__(self, source, depth, max_batch_bytes, budget_bytes, query_statistics=None):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0 or depth > 8:
            raise ValueError("prefetch must be an integer between 0 and 8")
        if depth*max_batch_bytes > budget_bytes:
            raise MemoryError("prefetched batch payloads exceed prefetch_budget_mb; reduce prefetch or chunk_size")
        self._source = iter(source)
        self._metrics = Measurements()
        self._query_statistics = query_statistics
        self._depth = depth
        self._closed = False
        self._state = {"queue": queue.Queue(maxsize=max(1, depth)), "slots": threading.Semaphore(depth),
                       "stop": threading.Event(), "lock": threading.Lock(), "bytes": 0, "items": 0,
                       "max_bytes": max_batch_bytes}
        self._thread = None
        if depth:
            self._thread = threading.Thread(target=_produce, args=(self._source, self._state, self._metrics), daemon=True)
            self._thread.start()
        self._finalizer = weakref.finalize(self, _close, self._state, self._thread)
        self._metrics.update(prefetch=depth, reserved_payload_bytes=depth*max_batch_bytes)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        start = time.perf_counter()
        try:
            if self._depth:
                value, size, error = self._state["queue"].get()
                if error is not None:
                    self._state["slots"].release()
                    raise error
                with self._state["lock"]:
                    self._state["bytes"] -= size
                    self._state["items"] -= 1
                self._state["slots"].release()
            else:
                value = next(self._source)
                self._metrics.observe("generation_ms", (time.perf_counter()-start)*1000)
            self._metrics.observe("wait_ms", (time.perf_counter()-start)*1000)
            return value
        except BaseException:
            self.close()
            raise

    def statistics(self):
        result = self._metrics.snapshot()
        with self._state["lock"]:
            result["queue_occupancy"] = self._state["items"]
            result["queued_payload_bytes"] = self._state["bytes"]
        result["query"] = None if self._query_statistics is None else self._query_statistics()
        return result

    def record_training_step(self, milliseconds):
        training_step(self._metrics, milliseconds)

    def close(self):
        if not self._closed:
            self._closed = True
            self._finalizer()
            if not self._depth:
                close = getattr(self._source, "close", None)
                if close is not None:
                    close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
