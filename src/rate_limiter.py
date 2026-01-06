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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from .logger import logger
from .utils import get_file_path


class RateLimitExceededError(Exception):
    """Raised when rate limit wait time would exceed maximum allowed."""

    pass


@dataclass
class RateLimitDimension:
    """Tracks usage for one dimension (RPM, TPM, or RPD)."""

    limit: float  # Current limit (may be learned from API)
    max_limit: float  # Maximum allowed limit
    min_limit: float  # Minimum limit (floor)
    used: float = 0.0  # Usage in current window
    window_start: float = field(default_factory=time.time)  # Window start timestamp
    window_seconds: float = 60.0  # Window duration (60s for RPM/TPM, 86400 for RPD)
    learned_from_api: bool = False  # Whether limit was learned from response headers
    remaining_from_api: Optional[float] = None  # Remaining quota from last API response
    reset_at_from_api: Optional[float] = None  # Reset timestamp from last API response

    def is_exhausted(self, amount: float = 1.0) -> bool:
        """Check if adding `amount` would exceed the limit."""
        self._maybe_reset_window()
        # If we have API-reported remaining, use that for more accuracy
        if self.remaining_from_api is not None and self.learned_from_api:
            return amount > self.remaining_from_api
        return (self.used + amount) > self.limit

    def consume(self, amount: float = 1.0) -> None:
        """Record usage of `amount` units."""
        self._maybe_reset_window()
        self.used += amount
        # Also decrement remaining if we're tracking it
        if self.remaining_from_api is not None:
            self.remaining_from_api = max(0, self.remaining_from_api - amount)

    def time_until_available(self, amount: float = 1.0) -> float:
        """Returns seconds until `amount` units would be available."""
        self._maybe_reset_window()

        # If we have API-reported reset time, use it
        if (
            self.reset_at_from_api is not None
            and self.learned_from_api
            and self.remaining_from_api is not None
            and amount > self.remaining_from_api
        ):
            return max(0, self.reset_at_from_api - time.time())

        if not self.is_exhausted(amount):
            return 0.0

        # Time until window resets
        elapsed = time.time() - self.window_start
        return max(0, self.window_seconds - elapsed)

    def update_from_headers(
        self,
        limit: Optional[float],
        remaining: Optional[float],
        reset_at: Optional[float],
    ) -> None:
        """Update limits based on API response headers."""
        if limit is not None and limit > 0:
            # Clamp to our configured max
            self.limit = min(limit, self.max_limit)
            self.learned_from_api = True
        if remaining is not None:
            self.remaining_from_api = remaining
            # Sync our used counter with reality
            if self.limit > 0:
                self.used = self.limit - remaining
        if reset_at is not None:
            self.reset_at_from_api = reset_at

    def _maybe_reset_window(self) -> None:
        """Reset the window if it has expired."""
        now = time.time()
        elapsed = now - self.window_start

        # Check API-reported reset time first
        if self.reset_at_from_api is not None and now >= self.reset_at_from_api:
            self.used = 0.0
            self.window_start = now
            self.remaining_from_api = self.limit if self.learned_from_api else None
            self.reset_at_from_api = None
            return

        # Standard window reset
        if elapsed >= self.window_seconds:
            # For daily windows, align to UTC midnight
            if self.window_seconds >= 86400:
                self.window_start = self.get_utc_midnight_timestamp()
            else:
                self.window_start = now
            self.used = 0.0
            self.remaining_from_api = None
            self.reset_at_from_api = None

    @staticmethod
    def get_utc_midnight_timestamp() -> float:
        """Get timestamp of the most recent UTC midnight."""
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.timestamp()

    def apply_backoff(self) -> None:
        """Apply multiplicative decrease on failure (AIMD)."""
        old_limit = self.limit
        self.limit = max(self.min_limit, self.limit * 0.5)
        if self.limit != old_limit:
            logger.debug(
                f"Rate limit backoff: {old_limit:.0f} -> {self.limit:.0f} "
                f"(window: {self.window_seconds}s)"
            )

    def apply_rampup(self) -> None:
        """Apply additive increase on success (AIMD). Uses 2% multiplicative for smoother ramp."""
        if self.learned_from_api:
            # Don't ramp up if we know the real limit
            return
        old_limit = self.limit
        # Multiplicative increase of 2% is smoother than fixed +10
        self.limit = min(self.max_limit, self.limit * 1.02)
        if int(self.limit) > int(old_limit):
            logger.debug(
                f"Rate limit rampup: {old_limit:.0f} -> {self.limit:.0f} "
                f"(window: {self.window_seconds}s)"
            )


@dataclass
class ModelRateLimitConfig:
    """Configuration for a specific model's rate limits."""

    rpm: int = 60
    rpm_max: int = 1000
    tpm: int = 100000
    tpm_max: int = 1000000
    rpd: int = 10000
    rpd_max: int = 100000
    reset_hour_utc: int = 0  # Hour when daily limit resets (0 = midnight UTC)


# Default configurations per provider and model
# Format: "provider" or "provider:model" -> config
DEFAULT_RATE_LIMITS: dict[str, ModelRateLimitConfig] = {
    # OpenAI Tier 5 defaults (conservative starting points, will learn from headers)
    "openai": ModelRateLimitConfig(
        rpm=1000,
        rpm_max=15000,
        tpm=1000000,
        tpm_max=40000000,
        rpd=100000,
        rpd_max=10000000,
    ),
    "openai:gpt-5.2": ModelRateLimitConfig(
        rpm=1000,
        rpm_max=15000,
        tpm=1000000,
        tpm_max=40000000,
        rpd=100000,
        rpd_max=10000000,
    ),
    "openai:gpt-5.1": ModelRateLimitConfig(
        rpm=1000,
        rpm_max=15000,
        tpm=1000000,
        tpm_max=40000000,
        rpd=100000,
        rpd_max=10000000,
    ),
    "openai:gpt-5": ModelRateLimitConfig(
        rpm=1000,
        rpm_max=15000,
        tpm=1000000,
        tpm_max=40000000,
        rpd=100000,
        rpd_max=10000000,
    ),
    "openai:gpt-4o": ModelRateLimitConfig(
        rpm=500, rpm_max=10000, tpm=500000, tpm_max=30000000, rpd=50000, rpd_max=5000000
    ),
    "openai:gpt-4o-mini": ModelRateLimitConfig(
        rpm=1000,
        rpm_max=30000,
        tpm=1000000,
        tpm_max=150000000,
        rpd=100000,
        rpd_max=10000000,
    ),
    # Anthropic
    "anthropic": ModelRateLimitConfig(
        rpm=50, rpm_max=4000, tpm=100000, tpm_max=400000, rpd=10000, rpd_max=1000000
    ),
    # DeepSeek
    "deepseek": ModelRateLimitConfig(
        rpm=60, rpm_max=500, tpm=100000, tpm_max=10000000, rpd=10000, rpd_max=1000000
    ),
    # Google Gemini - Text models
    "google_text": ModelRateLimitConfig(
        rpm=15, rpm_max=60, tpm=500000, tpm_max=4000000, rpd=200, rpd_max=1500
    ),
    "google_text:gemini-3-pro": ModelRateLimitConfig(
        rpm=25, rpm_max=60, tpm=500000, tpm_max=1000000, rpd=200, rpd_max=250
    ),
    "google_text:gemini-3-pro-preview": ModelRateLimitConfig(
        rpm=25, rpm_max=60, tpm=500000, tpm_max=1000000, rpd=200, rpd_max=250
    ),
    "google_text:gemini-2.5-flash": ModelRateLimitConfig(
        rpm=1000, rpm_max=2000, tpm=250000, tpm_max=1000000, rpd=1000, rpd_max=10000
    ),
    "google_text:gemini-2.5-pro": ModelRateLimitConfig(
        rpm=150, rpm_max=360, tpm=1000000, tpm_max=4000000, rpd=1000, rpd_max=2500
    ),
    # Google Gemini - TTS models (very restrictive)
    "google_tts": ModelRateLimitConfig(
        rpm=5, rpm_max=10, tpm=5000, tpm_max=10000, rpd=50, rpd_max=100
    ),
    "google_tts:gemini-2.5-flash-tts": ModelRateLimitConfig(
        rpm=5, rpm_max=10, tpm=5000, tpm_max=10000, rpd=50, rpd_max=100
    ),
    "google_tts:gemini-2.5-flash-preview-tts": ModelRateLimitConfig(
        rpm=5, rpm_max=10, tpm=5000, tpm_max=10000, rpd=50, rpd_max=100
    ),
    "google_tts:gemini-2.5-pro-preview-tts": ModelRateLimitConfig(
        rpm=10, rpm_max=10, tpm=5000, tpm_max=10000, rpd=50, rpd_max=50
    ),
    # Google Gemini - Image models
    "google_image": ModelRateLimitConfig(
        rpm=15, rpm_max=20, tpm=50000, tpm_max=100000, rpd=200, rpd_max=250
    ),
    "google_image:gemini-3-pro-image-preview": ModelRateLimitConfig(
        rpm=18, rpm_max=20, tpm=50000, tpm_max=100000, rpd=200, rpd_max=250
    ),
    # ElevenLabs TTS
    "elevenLabs": ModelRateLimitConfig(
        rpm=10, rpm_max=100, tpm=100000, tpm_max=1000000, rpd=1000, rpd_max=10000
    ),
    # Azure TTS
    "azure": ModelRateLimitConfig(
        rpm=30, rpm_max=200, tpm=500000, tpm_max=5000000, rpd=10000, rpd_max=100000
    ),
    # Replicate (image generation)
    "replicate": ModelRateLimitConfig(
        rpm=10, rpm_max=100, tpm=100000, tpm_max=1000000, rpd=1000, rpd_max=10000
    ),
}


class MultiDimensionalRateLimiter:
    """
    Rate limiter that tracks RPM (Requests Per Minute), TPM (Tokens Per Minute),
    and RPD (Requests Per Day) simultaneously.

    Uses AIMD (Additive Increase, Multiplicative Decrease) to adapt limits,
    and can learn actual limits from API response headers.
    """

    def __init__(
        self,
        provider: str,
        model: str = "",
        config: Optional[ModelRateLimitConfig] = None,
        manager: Optional["RateLimitManager"] = None,
    ):
        self._provider = provider
        self._model = model
        self._manager = manager
        self._key = f"{provider}:{model}" if model else provider

        # Get config, falling back to provider default
        if config is None:
            config = DEFAULT_RATE_LIMITS.get(
                self._key, DEFAULT_RATE_LIMITS.get(provider, ModelRateLimitConfig())
            )

        # Initialize the three dimensions
        self.rpm = RateLimitDimension(
            limit=float(config.rpm),
            max_limit=float(config.rpm_max),
            min_limit=1.0,
            window_seconds=60.0,
        )
        self.tpm = RateLimitDimension(
            limit=float(config.tpm),
            max_limit=float(config.tpm_max),
            min_limit=100.0,
            window_seconds=60.0,
        )
        self.rpd = RateLimitDimension(
            limit=float(config.rpd),
            max_limit=float(config.rpd_max),
            min_limit=1.0,
            window_seconds=86400.0,  # 24 hours
        )

        # Queue-based scheduling
        self._queue: asyncio.Queue[tuple[asyncio.Future[None], int]] = asyncio.Queue()
        self._scheduler_task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()

        # Track estimated vs actual tokens for learning
        self._last_estimated_tokens = 0

    @property
    def key(self) -> str:
        return self._key

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """
        Wait until it's safe to make a request.
        Pass estimated_tokens for TPM tracking (will be corrected later via report_success).
        """
        # Ensure scheduler is running
        if self._scheduler_task is None or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(self._scheduler())

        self._last_estimated_tokens = estimated_tokens

        # Create a future and queue it
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        await self._queue.put((fut, estimated_tokens))

        # Wait for our turn
        try:
            await fut
        except asyncio.CancelledError:
            raise

    # Maximum time to wait for rate limits before giving up
    MAX_WAIT_SECONDS = 120.0  # 2 minutes max wait

    async def _scheduler(self) -> None:
        """Background task that processes the queue respecting all rate limits."""
        while True:
            item = await self._queue.get()
            fut, estimated_tokens = item

            if fut.cancelled():
                self._queue.task_done()
                continue

            # Track total wait time
            total_waited = 0.0
            exceeded_limit = False

            # Wait until all dimensions allow us to proceed
            while True:
                # Calculate wait time for each dimension
                rpm_wait = self.rpm.time_until_available(1)
                tpm_wait = (
                    self.tpm.time_until_available(estimated_tokens)
                    if estimated_tokens > 0
                    else 0
                )
                rpd_wait = self.rpd.time_until_available(1)

                max_wait = max(rpm_wait, tpm_wait, rpd_wait)

                if max_wait <= 0:
                    # Ready to proceed
                    break

                # Check if we've exceeded max wait time
                if total_waited + max_wait > self.MAX_WAIT_SECONDS:
                    dimension = (
                        "RPM"
                        if rpm_wait == max_wait
                        else ("TPM" if tpm_wait == max_wait else "RPD")
                    )
                    error_msg = (
                        f"Rate limit exceeded for {self._key}: "
                        f"would need to wait {max_wait:.1f}s for {dimension}. "
                        f"Try again later or delete rate_limits.json to reset."
                    )
                    logger.error(error_msg)
                    if not fut.done():
                        fut.set_exception(RateLimitExceededError(error_msg))
                    self._queue.task_done()
                    exceeded_limit = True
                    break

                # Log if we're waiting due to limits (only for significant waits)
                if max_wait > 5:
                    dimension = (
                        "RPM"
                        if rpm_wait == max_wait
                        else ("TPM" if tpm_wait == max_wait else "RPD")
                    )
                    logger.debug(
                        f"RateLimiter [{self._key}]: Waiting {max_wait:.1f}s for {dimension}"
                    )

                try:
                    sleep_time = min(max_wait, 1.0)  # Check every second
                    await asyncio.sleep(sleep_time)
                    total_waited += sleep_time
                except asyncio.CancelledError:
                    if not fut.done():
                        fut.cancel()
                    self._queue.task_done()
                    return

            # If we broke out due to max wait exceeded, skip to next item
            if exceeded_limit:
                continue

            # Check if cancelled while waiting
            if fut.cancelled():
                self._queue.task_done()
                continue

            # Consume from all dimensions
            async with self._lock:
                self.rpm.consume(1)
                if estimated_tokens > 0:
                    self.tpm.consume(estimated_tokens)
                self.rpd.consume(1)

            if not fut.done():
                fut.set_result(None)
            self._queue.task_done()

    async def __aenter__(self) -> "MultiDimensionalRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        pass

    async def report_success(
        self,
        actual_tokens: int = 0,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Called after a successful request.
        - actual_tokens: Real token count from response (corrects estimate)
        - headers: Response headers for learning actual limits
        """
        async with self._lock:
            # Correct TPM usage if we have actual token count from response body
            if actual_tokens > 0 and self._last_estimated_tokens > 0:
                diff = actual_tokens - self._last_estimated_tokens
                if diff != 0:
                    self.tpm.used = max(0, self.tpm.used + diff)
                logger.debug(
                    f"RateLimiter [{self._key}]: Token correction "
                    f"(estimated: {self._last_estimated_tokens}, actual: {actual_tokens})"
                )
                # If we're getting actual tokens from response, we're accurately tracking usage
                # No need to ramp up TPM - we know exactly what we're using
                self.tpm.learned_from_api = True

            # Parse headers to learn actual limits (OpenAI-style x-ratelimit headers)
            if headers:
                self._parse_rate_limit_headers(headers)

            # Apply ramp-up (only if not learned from API)
            self.rpm.apply_rampup()
            self.tpm.apply_rampup()
            # Don't ramp up RPD - it's usually fixed

            # Save state
            if self._manager:
                self._manager.save_state()

    async def report_failure(
        self,
        headers: Optional[dict[str, str]] = None,
        retry_after: Optional[float] = None,
        is_daily_limit: bool = False,
    ) -> None:
        """
        Called when a 429 or rate limit error occurs.

        Args:
            headers: Response headers (may contain rate limit info)
            retry_after: Retry-After header value in seconds
            is_daily_limit: Set True only if the error explicitly indicates daily quota exceeded
        """
        async with self._lock:
            # Parse headers even on failure - we might learn the real limits
            if headers:
                self._parse_rate_limit_headers(headers)

            # If we got a Retry-After header, use it
            if retry_after and retry_after > 0:
                # Set the reset time for RPM (most likely the bottleneck)
                self.rpm.reset_at_from_api = time.time() + retry_after

            # Apply backoff to RPM and TPM (the typical cause of 429 errors)
            self.rpm.apply_backoff()
            self.tpm.apply_backoff()

            # Only apply RPD backoff if we know it's specifically a daily limit error
            # Standard 429 errors are almost always RPM/TPM related
            if is_daily_limit:
                self.rpd.apply_backoff()

            logger.warning(
                f"RateLimiter [{self._key}]: 429/Failure. "
                f"RPM: {self.rpm.limit:.0f}, TPM: {self.tpm.limit:.0f}, RPD: {self.rpd.limit:.0f}"
            )

            if self._manager:
                self._manager.save_state()

    async def report_timeout(self) -> None:
        """Called when a request times out. Does not back off aggressively."""
        logger.warning(
            f"RateLimiter [{self._key}]: Timeout. Maintaining current limits."
        )

    def _parse_rate_limit_headers(self, headers: dict[str, str]) -> None:
        """Parse rate limit information from response headers."""
        # OpenAI-style headers
        rpm_limit = self._parse_header_float(headers.get("x-ratelimit-limit-requests"))
        rpm_remaining = self._parse_header_float(
            headers.get("x-ratelimit-remaining-requests")
        )
        rpm_reset = self._parse_reset_header(headers.get("x-ratelimit-reset-requests"))

        tpm_limit = self._parse_header_float(headers.get("x-ratelimit-limit-tokens"))
        tpm_remaining = self._parse_header_float(
            headers.get("x-ratelimit-remaining-tokens")
        )
        tpm_reset = self._parse_reset_header(headers.get("x-ratelimit-reset-tokens"))

        # Update dimensions
        if rpm_limit or rpm_remaining is not None or rpm_reset:
            self.rpm.update_from_headers(rpm_limit, rpm_remaining, rpm_reset)
            if rpm_limit:
                logger.debug(
                    f"RateLimiter [{self._key}]: Learned RPM limit from API: {rpm_limit}"
                )

        if tpm_limit or tpm_remaining is not None or tpm_reset:
            self.tpm.update_from_headers(tpm_limit, tpm_remaining, tpm_reset)
            if tpm_limit:
                logger.debug(
                    f"RateLimiter [{self._key}]: Learned TPM limit from API: {tpm_limit}"
                )

        # Also check Retry-After for immediate wait time
        retry_after = self._parse_header_float(headers.get("retry-after"))
        if retry_after:
            self.rpm.reset_at_from_api = time.time() + retry_after

    @staticmethod
    def _parse_header_float(value: Optional[str]) -> Optional[float]:
        """Parse a header value as a float."""
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_reset_header(value: Optional[str]) -> Optional[float]:
        """Parse a reset time header (could be seconds or ISO timestamp)."""
        if value is None:
            return None
        try:
            # Try as seconds first
            seconds = float(value.rstrip("s").rstrip("ms"))
            if "ms" in value:
                seconds /= 1000
            return time.time() + seconds
        except ValueError:
            pass
        try:
            # Try as ISO timestamp
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return None

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "rpm_limit": self.rpm.limit,
            "rpm_learned": self.rpm.learned_from_api,
            "tpm_limit": self.tpm.limit,
            "tpm_learned": self.tpm.learned_from_api,
            "rpd_limit": self.rpd.limit,
            "rpd_used": self.rpd.used,
            "rpd_window_start": self.rpd.window_start,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from persistence."""
        # Get the original config defaults for recovery
        config = DEFAULT_RATE_LIMITS.get(
            self._key, DEFAULT_RATE_LIMITS.get(self._provider, ModelRateLimitConfig())
        )

        if "rpm_limit" in state:
            loaded_limit = min(state["rpm_limit"], self.rpm.max_limit)
            # If backed off to near min_limit and not learned from API, reset to default
            if loaded_limit <= self.rpm.min_limit * 2 and not state.get(
                "rpm_learned", False
            ):
                logger.debug(
                    f"RateLimiter [{self._key}]: Resetting RPM from {loaded_limit:.0f} to default {config.rpm}"
                )
                self.rpm.limit = float(config.rpm)
            else:
                self.rpm.limit = loaded_limit
            self.rpm.learned_from_api = state.get("rpm_learned", False)

        if "tpm_limit" in state:
            loaded_limit = min(state["tpm_limit"], self.tpm.max_limit)
            # Reset if backed off too much and not learned
            if loaded_limit <= self.tpm.min_limit * 2 and not state.get(
                "tpm_learned", False
            ):
                logger.debug(
                    f"RateLimiter [{self._key}]: Resetting TPM from {loaded_limit:.0f} to default {config.tpm}"
                )
                self.tpm.limit = float(config.tpm)
            else:
                self.tpm.limit = loaded_limit
            self.tpm.learned_from_api = state.get("tpm_learned", False)

        if "rpd_limit" in state:
            loaded_limit = min(state["rpd_limit"], self.rpd.max_limit)
            # Always reset RPD to default on new session if it was backed off
            # Daily limits should reset each day anyway
            if loaded_limit < config.rpd:
                logger.debug(
                    f"RateLimiter [{self._key}]: Resetting RPD from {loaded_limit:.0f} to default {config.rpd}"
                )
                self.rpd.limit = float(config.rpd)
            else:
                self.rpd.limit = loaded_limit

        # Restore daily usage if within the same day
        if "rpd_window_start" in state:
            saved_start = state["rpd_window_start"]
            current_midnight = RateLimitDimension.get_utc_midnight_timestamp()
            # If saved window start is from today, restore usage
            if saved_start >= current_midnight:
                self.rpd.window_start = saved_start
                self.rpd.used = state.get("rpd_used", 0)
            else:
                # New day - reset usage
                self.rpd.used = 0
                self.rpd.window_start = current_midnight


class RateLimitManager:
    """
    Manages rate limiters for all provider/model combinations.
    Handles persistence of learned limits.
    """

    _instance: Optional["RateLimitManager"] = None

    def __init__(self) -> None:
        self.limiters: dict[str, MultiDimensionalRateLimiter] = {}
        self.state_file = "rate_limits.json"
        self._loaded_state = self._load_state()
        self._save_lock = (
            asyncio.Lock() if asyncio.get_event_loop().is_running() else None
        )

    @classmethod
    def get_instance(cls) -> "RateLimitManager":
        if cls._instance is None:
            cls._instance = RateLimitManager()
        return cls._instance

    def get_limiter(
        self, provider: str, model: str = ""
    ) -> MultiDimensionalRateLimiter:
        """Get or create a rate limiter for a provider/model combination."""
        key = f"{provider}:{model}" if model else provider

        if key not in self.limiters:
            # Try to get model-specific config, fall back to provider
            config = DEFAULT_RATE_LIMITS.get(
                key, DEFAULT_RATE_LIMITS.get(provider, ModelRateLimitConfig())
            )

            limiter = MultiDimensionalRateLimiter(
                provider=provider,
                model=model,
                config=config,
                manager=self,
            )

            # Load saved state
            if key in self._loaded_state:
                limiter.load_state(self._loaded_state[key])

            self.limiters[key] = limiter
            logger.debug(
                f"Created rate limiter [{key}]: "
                f"RPM={limiter.rpm.limit:.0f}, "
                f"TPM={limiter.tpm.limit:.0f}, "
                f"RPD={limiter.rpd.limit:.0f}"
            )

        return self.limiters[key]

    def _load_state(self) -> dict[str, dict[str, Any]]:
        """Load saved state from disk."""
        try:
            path = get_file_path(self.state_file)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    # Handle old format (just rpm values)
                    if data and isinstance(next(iter(data.values())), (int, float)):
                        # Migrate old format
                        return {k: {"rpm_limit": v} for k, v in data.items()}
                    return data
        except Exception as e:
            logger.error(f"Failed to load rate limits: {e}")
        return {}

    def save_state(self) -> None:
        """Save current state to disk."""
        try:
            path = get_file_path(self.state_file)
            state = {key: limiter.get_state() for key, limiter in self.limiters.items()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rate limits: {e}")

    def get_all_limits_summary(self) -> dict[str, dict[str, float]]:
        """Get a summary of all current rate limits for reporting."""
        return {
            key: {
                "rpm": limiter.rpm.limit,
                "rpm_used": limiter.rpm.used,
                "tpm": limiter.tpm.limit,
                "tpm_used": limiter.tpm.used,
                "rpd": limiter.rpd.limit,
                "rpd_used": limiter.rpd.used,
            }
            for key, limiter in self.limiters.items()
        }


def get_rate_limiter(provider: str, model: str = "") -> MultiDimensionalRateLimiter:
    """Global accessor for rate limiters."""
    return RateLimitManager.get_instance().get_limiter(provider, model)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.
    Uses a rough heuristic: ~4 characters per token for English.
    This is intentionally conservative (overestimates).
    """
    if not text:
        return 0
    # Rough estimate: 1 token ≈ 4 characters for English
    # Use 3.5 to be slightly conservative (overestimate tokens)
    return max(1, int(len(text) / 3.5))


def extract_rate_limit_headers(headers: Any) -> dict[str, str]:
    """
    Extract rate limit related headers from API response.

    Handles OpenAI-style headers:
    - x-ratelimit-limit-requests / x-ratelimit-remaining-requests
    - x-ratelimit-limit-tokens / x-ratelimit-remaining-tokens
    - x-ratelimit-reset-requests / x-ratelimit-reset-tokens
    - retry-after
    """
    rate_limit_headers: dict[str, str] = {}
    for key in headers:
        key_lower = key.lower()
        if "ratelimit" in key_lower or "retry-after" in key_lower:
            rate_limit_headers[key_lower] = headers[key]
    return rate_limit_headers


def parse_retry_after(headers: Any) -> Optional[float]:
    """Parse Retry-After header value to seconds."""
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return None


def extract_google_token_usage(response_bytes: bytes, fallback: int = 0) -> int:
    """
    Extract token usage from Google API response body.

    Google returns usageMetadata in the response body, not in headers:
    { "usageMetadata": { "promptTokenCount": X, "candidatesTokenCount": Y, "totalTokenCount": Z } }
    """
    try:
        data = json.loads(response_bytes)
        usage = data.get("usageMetadata", {})
        # Prefer totalTokenCount if available, otherwise sum prompt + candidates
        total = usage.get("totalTokenCount")
        if total is not None:
            return int(total)
        prompt_tokens = usage.get("promptTokenCount", 0)
        candidates_tokens = usage.get("candidatesTokenCount", 0)
        if prompt_tokens or candidates_tokens:
            return prompt_tokens + candidates_tokens
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    return fallback


def extract_openai_token_usage(response: dict) -> int:
    """
    Extract token usage from OpenAI-style API response.

    OpenAI and compatible APIs return:
    { "usage": { "prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z } }
    """
    try:
        usage = response.get("usage", {})
        return usage.get("total_tokens", 0) or (
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
    except (KeyError, TypeError, AttributeError):
        return 0
