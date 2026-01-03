"""
Copyright (C) 2024 Michael Piazza

This file is part of Smart Notes.

Smart Notes is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Smart Notes is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Smart Notes.  If not, see <https://www.gnu.org/licenses/>.
"""

import asyncio
import json
import os
import time
from typing import Dict, Optional

from .logger import logger
from .utils import get_file_path


class AdaptiveRateLimiter:
    """
    Implements a rate-based limiter (leaky bucket) that controls Requests Per Minute (RPM).
    Instead of limiting concurrency, it enforces a minimum time interval between requests.

    It adapts the RPM based on success/failure:
    - Success: Additive Increase (RAMP UP)
    - Failure (429): Multiplicative Decrease (BACK OFF)
    """

    def __init__(
        self,
        initial_rpm: int = 5,
        min_rpm: int = 1,
        max_rpm: int = 60,
        manager=None,
        provider_name: str = "",
    ):
        self._rpm = float(initial_rpm)
        self._min_rpm = min_rpm
        self._max_rpm = max_rpm
        self._manager = manager
        self._provider_name = provider_name

        # Queue-based scheduling to allow unlimited concurrency but enforce rate limits
        # dynamically. This prevents "locking in" slow rates for future requests.
        self._queue = asyncio.Queue()
        self._rpm_changed = asyncio.Event()
        self._last_request_time = 0.0
        self._scheduler_task: Optional[asyncio.Task] = None

    async def acquire(self):
        """
        Waits until it's safe to make a request based on current RPM.
        """
        # Ensure scheduler is running
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(self._scheduler())

        # Create a future for this request and add to queue
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self._queue.put(fut)

        # Wait for our turn
        try:
            await fut
        except asyncio.CancelledError:
            # If we are cancelled, the scheduler might still be holding us or
            # about to release us. We can't easily remove from asyncio.Queue
            # (it's not a deque), so the scheduler handle cancellation
            # by checking fut.cancelled().
            raise

    async def _scheduler(self):
        while True:
            # Wait for next request
            fut = await self._queue.get()

            if fut.cancelled():
                self._queue.task_done()
                continue

            # Wait for rate limit slot
            while True:
                interval = 60.0 / self._rpm
                target = self._last_request_time + interval
                now = time.time()

                # If idle, we can go immediately (target < now).
                # Otherwise wait difference.
                wait = target - now

                if wait <= 0:
                    break

                try:
                    # Wait for the calculated time, but wake up if RPM changes
                    await asyncio.wait_for(self._rpm_changed.wait(), timeout=wait)
                    self._rpm_changed.clear()
                    # Loop again to recalculate with new RPM
                    if fut.cancelled():
                        break
                except asyncio.TimeoutError:
                    # Time reached naturally
                    break

            # If task was cancelled during wait, don't consume the slot
            if fut.cancelled():
                self._queue.task_done()
                continue

            # Dispatch
            self._last_request_time = time.time()
            if not fut.done():
                fut.set_result(None)
            self._queue.task_done()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def report_success(self):
        """Called when a request succeeds. Gently ramps up RPM."""
        if self._rpm < self._max_rpm:
            old_rpm = int(self._rpm)
            # Additive increase: +10.0 RPM per success for faster ramp-up
            self._rpm += 10.0

            # Cap at max
            if self._rpm > self._max_rpm:
                self._rpm = self._max_rpm

            new_rpm = int(self._rpm)

            # If RPM changed, notify scheduler to potentially wake up early
            if self._rpm != float(old_rpm):
                self._rpm_changed.set()

            if new_rpm > old_rpm:
                logger.debug(
                    f"RateLimiter: Ramping up {self._provider_name} to {new_rpm} RPM"
                )
                if self._manager:
                    self._manager.save_state()

    async def report_failure(self):
        """Called when a 429/failure occurs. Backs off RPM."""
        if self._rpm > self._min_rpm:
            old_rpm = int(self._rpm)
            # Multiplicative decrease
            self._rpm = max(self._min_rpm, self._rpm * 0.5)

            new_rpm = int(self._rpm)

            # Notify scheduler (though decreasing RPM usually means waiting longer,
            # so waking up just to wait longer is fine)
            self._rpm_changed.set()

            logger.warning(
                f"RateLimiter: 429/Congestion detected! Backing off {self._provider_name} to {new_rpm} RPM"
            )
            if self._manager:
                self._manager.save_state()

    async def report_timeout(self):
        """Called when a request times out. Currently does not back off RPM."""
        logger.warning(
            f"RateLimiter: Timeout detected for {self._provider_name}. Maintaining {int(self._rpm)} RPM"
        )


class RateLimitManager:
    _instance = None

    def __init__(self):
        self.limiters: Dict[str, AdaptiveRateLimiter] = {}
        self.state_file = "rate_limits.json"

        # Default RPM limits
        self.defaults = {
            # OpenAI Tier 5 Limits (High Concurrency)
            # Tier 5 allows ~10,000 RPM. We'll set a high initial to allow maximum throughput.
            "openai": {"initial": 10000, "max": 30000},
            "anthropic": {"initial": 50, "max": 200},
            "deepseek": {"initial": 50, "max": 200},
            # Google Services
            "google_text": {"initial": 5, "max": 25},
            "google_image": {"initial": 5, "max": 20},
            "google_tts": {"initial": 2, "max": 10},
            # Fallback
            "google": {"initial": 5, "max": 20},
            # TTS
            "elevenLabs": {"initial": 10, "max": 60},
            "azure": {"initial": 30, "max": 120},
            # Image
            "replicate": {"initial": 5, "max": 50},
        }

        self.loaded_state = self._load_state()

    def _load_state(self) -> Dict[str, float]:
        try:
            path = get_file_path(self.state_file)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load rate limits: {e}")
        return {}

    def save_state(self):
        try:
            path = get_file_path(self.state_file)
            state = {k: v._rpm for k, v in self.limiters.items()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save rate limits: {e}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RateLimitManager()
        return cls._instance

    def get_limiter(self, provider: str) -> AdaptiveRateLimiter:
        if provider not in self.limiters:
            config = self.defaults.get(provider, {"initial": 10, "max": 60})

            initial_rpm = config["initial"]

            # Load saved state
            if provider in self.loaded_state:
                saved_rpm = self.loaded_state[provider]
                # Sanity check saved value
                if (
                    1 <= saved_rpm <= config["max"] * 1.5
                ):  # Allow slight overage if config changed
                    initial_rpm = saved_rpm
                # Cap it at current max if strictly enforced
                initial_rpm = min(initial_rpm, config["max"])

            self.limiters[provider] = AdaptiveRateLimiter(
                initial_rpm=initial_rpm,
                max_rpm=config["max"],
                manager=self,
                provider_name=provider,
            )

        return self.limiters[provider]


# Global accessor
def get_rate_limiter(provider: str) -> AdaptiveRateLimiter:
    return RateLimitManager.get_instance().get_limiter(provider)
