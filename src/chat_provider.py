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
import time
from typing import Optional

import aiohttp

from .config import config
from .constants import (
    CHAT_CLIENT_TIMEOUT_SEC,
    DEFAULT_TEMPERATURE,
    MAX_RETRIES,
    REASONING_CLIENT_TIMEOUT_SEC,
    RETRY_BASE_SECONDS,
)
from .logger import logger
from .models import (
    ChatModels,
    ChatProviders,
    CustomProvider,
    OpenAIReasoningEffort,
    openai_reasoning_efforts_for_model,
)
from .rate_limiter import (
    estimate_tokens,
    extract_rate_limit_headers,
    get_rate_limiter,
    parse_retry_after,
)

OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
GOOGLE_ENDPOINT_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class ChatProvider:
    async def async_get_chat_response(
        self,
        prompt: str,
        model: ChatModels,
        provider: ChatProviders,
        note_id: int,  # Kept for compatibility but unused
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: Optional[OpenAIReasoningEffort] = None,
        retry_count: int = 0,
    ) -> str:
        # Check custom providers
        custom_provider = next(
            (p for p in (config.custom_providers or []) if p["name"] == provider), None
        )
        if custom_provider:
            return await self._get_openai_compatible_response(
                prompt,
                model,
                temperature,
                reasoning_effort,
                retry_count,
                custom_provider,
            )

        if provider == "openai":
            return await self._get_openai_response(
                prompt, model, temperature, reasoning_effort, retry_count
            )
        elif provider == "anthropic":
            return await self._get_anthropic_response(
                prompt, model, temperature, retry_count
            )
        elif provider == "deepseek":
            return await self._get_deepseek_response(
                prompt, model, temperature, retry_count
            )
        elif provider == "google":
            return await self._get_google_response(
                prompt, model, temperature, retry_count
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _get_openai_compatible_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        reasoning_effort: Optional[OpenAIReasoningEffort],
        retry_count: int,
        provider_config: CustomProvider,
    ) -> str:
        api_key = provider_config["api_key"]
        base_url = provider_config["base_url"].rstrip("/")

        # Intelligent URL guessing
        if not base_url.endswith("/chat/completions"):
            # If it ends in /v1, just append chat/completions
            if base_url.endswith("/v1"):
                url = f"{base_url}/chat/completions"
            else:
                # Otherwise assume it's a base URL and append full path
                url = f"{base_url}/v1/chat/completions"
        else:
            url = base_url

        logger.debug(
            f"Custom Provider {provider_config['name']}: hitting {url} model: {model} retries {retry_count}"
        )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        # Pass reasoning effort if set? Most compatible providers probably ignore it or use temperature.
        if reasoning_effort and reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort
            # Some might error if both are sent, similar to OpenAI logic
            if "temperature" in payload:
                del payload["temperature"]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Check for reasoning models to allow longer timeouts
        timeout = CHAT_CLIENT_TIMEOUT_SEC
        lower_model = model.lower()
        if (
            "o1-" in lower_model
            or "o3-" in lower_model
            or "gemini-3" in lower_model
            or "reasoning" in lower_model
            or "thinking" in lower_model
        ):
            timeout = REASONING_CLIENT_TIMEOUT_SEC
            logger.debug(
                f"Custom Provider: Detected reasoning model {model}, using extended timeout {timeout}s"
            )

        return await self._execute_request(
            url=url,
            headers=headers,
            json_payload=payload,
            timeout_sec=timeout,
            retry_count=retry_count,
            provider=provider_config["name"],
            prompt=prompt,
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

    async def _get_openai_response(
        self,
        prompt: str,
        model: ChatModels,
        temperature: float,
        reasoning_effort: Optional[OpenAIReasoningEffort],
        retry_count: int,
    ) -> str:
        api_key = config.openai_api_key
        if not api_key:
            raise Exception("OpenAI API key not found. Please set it in the settings.")

        # Check reasoning
        is_reasoning = model.lower().startswith("o") or model in (
            "gpt-5",
            "gpt-5.1",
            "gpt-5.2",
        )

        logger.debug(
            f"OpenAI: hitting {OPENAI_ENDPOINT} model: {model} retries {retry_count} for prompt: {prompt}"
        )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        timeout_val = CHAT_CLIENT_TIMEOUT_SEC
        if is_reasoning:
            efforts = openai_reasoning_efforts_for_model(model)

            # Use provided effort, or default if None/Invalid
            current_effort = reasoning_effort
            if (
                not current_effort or current_effort not in efforts
            ) and "none" in efforts:
                # If "none" is valid, it effectively means "use temperature" in my config UI logic,
                # but API-wise, O1 models *require* reasoning_effort or default?
                # Actually O1 models DON'T support temperature usually.
                # If user selected "none" in UI for an O1 model, what should happen?
                # The UI initializes to valid effort.
                # If passed effort is None, default to "medium" or first available.
                current_effort = "medium" if "medium" in efforts else efforts[0]

            if current_effort == "none":
                # If explicit "none" passed (meaning turn off reasoning if possible?),
                # but this block is `if is_reasoning`.
                # For models that support BOTH (like gpt-5.2 in the list), if effort is "none", use temperature?
                # My UI logic disables temperature if effort != "none".
                # So if effort IS "none", we should use temperature.
                pass

            if current_effort and current_effort != "none":
                payload["reasoning_effort"] = current_effort
                # Reasoning models usually don't support temperature
                # But check model specs. Assuming mutually exclusive here based on UI.
            else:
                payload["temperature"] = temperature

            if current_effort in ("high", "xhigh"):
                timeout_val = REASONING_CLIENT_TIMEOUT_SEC
        else:
            payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        return await self._execute_request(
            url=config.openai_endpoint or OPENAI_ENDPOINT,
            headers=headers,
            json_payload=payload,
            timeout_sec=timeout_val,
            retry_count=retry_count,
            provider="openai",
            prompt=prompt,
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

    async def _get_anthropic_response(
        self,
        prompt: str,
        model: ChatModels,
        temperature: float,
        retry_count: int,
    ) -> str:
        api_key = config.anthropic_api_key
        if not api_key:
            raise Exception(
                "Anthropic API key not found. Please set it in the settings."
            )

        logger.debug(
            f"Anthropic: hitting {ANTHROPIC_ENDPOINT} model: {model} retries {retry_count}"
        )

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,  # Anthropic requires max_tokens
            "temperature": temperature,
        }

        return await self._execute_request(
            url=ANTHROPIC_ENDPOINT,
            headers=headers,
            json_payload=payload,
            timeout_sec=CHAT_CLIENT_TIMEOUT_SEC,
            retry_count=retry_count,
            provider="anthropic",
            prompt=prompt,
            model=model,
            temperature=temperature,
            reasoning_effort=None,
        )

    async def _get_deepseek_response(
        self,
        prompt: str,
        model: ChatModels,
        temperature: float,
        retry_count: int,
    ) -> str:
        api_key = config.deepseek_api_key
        if not api_key:
            raise Exception(
                "DeepSeek API key not found. Please set it in the settings."
            )

        logger.debug(
            f"DeepSeek: hitting {DEEPSEEK_ENDPOINT} model: {model} retries {retry_count}"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        return await self._execute_request(
            url=DEEPSEEK_ENDPOINT,
            headers=headers,
            json_payload=payload,
            timeout_sec=CHAT_CLIENT_TIMEOUT_SEC,
            retry_count=retry_count,
            provider="deepseek",
            prompt=prompt,
            model=model,
            temperature=temperature,
            reasoning_effort=None,
        )

    async def _get_google_response(
        self,
        prompt: str,
        model: ChatModels,
        temperature: float,
        retry_count: int,
    ) -> str:
        api_key = config.google_api_key
        if not api_key:
            raise Exception("Google API key not found. Please set it in the settings.")

        logger.debug(
            f"Google: hitting {GOOGLE_ENDPOINT_BASE} model: {model} retries {retry_count}"
        )

        url = f"{GOOGLE_ENDPOINT_BASE}/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}

        # Gemini 3 defaults to dynamic thinking (high) if not specified.
        # Use default temperature 1.0 as recommended for Gemini 3, unless user explicitly changes it?
        # User provided code:
        # "For Gemini 3, we strongly recommend keeping the temperature parameter at its default value of 1.0."
        # "Gemini 3 series models use dynamic thinking by default... If thinking_level is not specified, Gemini 3 will default to high."

        # We will pass the temperature if it's not 1.0 (default in Anki Smart Notes might be 1.0)
        # But wait, existing default is 1.0 in constants.py.
        # If I pass it, is it bad? The docs say "Changing the temperature (setting it below 1.0) may lead to unexpected behavior"
        # I'll stick to passing it for now as Smart Notes allows configuration.

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }

        return await self._execute_request(
            url=url,
            headers=headers,
            json_payload=payload,
            timeout_sec=CHAT_CLIENT_TIMEOUT_SEC,
            retry_count=retry_count,
            provider="google_text",  # Use specific text quota key
            prompt=prompt,
            model=model,
            temperature=temperature,
            reasoning_effort=None,
        )

    async def fetch_models(self, provider_config: CustomProvider) -> list[str]:
        api_key = provider_config["api_key"]
        base_url = provider_config["base_url"]

        # Try to derive models endpoint from chat completion endpoint
        url = base_url
        if "/chat/completions" in url:
            url = url.replace("/chat/completions", "/models")
        else:
            # Fallback: assume base_url is root or v1, try appending /models
            url = f"{url}models" if url.endswith("/") else f"{url}/models"

        logger.debug(f"Fetching models for {provider_config['name']} from {url}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as session,
                session.get(url, headers=headers) as response,
            ):
                response.raise_for_status()
                data = await response.json()
                # Standard OpenAI response: { "data": [ { "id": "model-id", ... } ] }
                return sorted([model["id"] for model in data.get("data", [])])
        except Exception as e:
            logger.error(f"Failed to fetch models from {provider_config['name']}: {e}")
            raise e

    async def _execute_request(
        self,
        url: str,
        headers: dict,
        json_payload: dict,
        timeout_sec: int,
        retry_count: int,
        provider: str,
        prompt: str,
        model: ChatModels,
        temperature: float,
        reasoning_effort: Optional[OpenAIReasoningEffort],
    ) -> str:
        # Get per-model rate limiter
        limiter = get_rate_limiter(provider, model)

        # Estimate tokens for TPM tracking
        estimated_tokens = estimate_tokens(prompt)

        try:
            # Acquire rate limit slot with token estimate
            acquire_start = time.time()
            await limiter.acquire(estimated_tokens)
            acquire_time = time.time() - acquire_start
            if acquire_time > 0.5:  # Only log if acquire took significant time
                logger.debug(
                    f"[{provider}] Rate limiter acquire took {acquire_time:.1f}s"
                )

            start_time = time.time()
            logger.debug(f"[{provider}] Sending HTTP request to {url}...")

            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout_sec)
                ) as session,
                session.post(url, headers=headers, json=json_payload) as response,
            ):
                elapsed = time.time() - start_time
                logger.debug(
                    f"[{provider}] Got HTTP response: status={response.status} "
                    f"in {elapsed:.1f}s"
                )

                # Extract headers for rate limit learning
                response_headers = extract_rate_limit_headers(response.headers)

                if response.status == 429:
                    logger.debug(f"Got a 429 from {provider}")

                    # Parse Retry-After header
                    retry_after = parse_retry_after(response.headers)
                    await limiter.report_failure(response_headers, retry_after)

                    if retry_count < MAX_RETRIES:
                        wait_time = retry_after or (2**retry_count) * RETRY_BASE_SECONDS
                        logger.debug(
                            f"Retry: {retry_count} Waiting {wait_time} seconds before retrying"
                        )
                        await asyncio.sleep(wait_time)
                        # Recursively call the public method to retry logic
                        return await self.async_get_chat_response(
                            prompt,
                            model,
                            provider.replace("_text", "")
                            if provider == "google_text"
                            else provider,  # type: ignore - Pass base provider name if needed for logic, but internally execute uses key
                            -1,
                            temperature,
                            reasoning_effort,
                            retry_count + 1,
                        )

                response.raise_for_status()

                resp = await response.json()

                # Check for error in response body (Google sometimes returns 200 with error)
                if "error" in resp:
                    error_msg = resp.get("error", {})
                    if isinstance(error_msg, dict):
                        error_text = error_msg.get("message", str(error_msg))
                        error_code = error_msg.get("code", "unknown")
                        error_status = error_msg.get("status", "")
                    else:
                        error_text = str(error_msg)
                        error_code = "unknown"
                        error_status = ""
                    logger.error(
                        f"{provider} returned error in body: [{error_code}] {error_status}: {error_text}"
                    )
                    raise Exception(f"{provider} API error: {error_text}")

                # Extract actual token usage from response
                actual_tokens = self._extract_token_usage(resp, provider)

                # Report success with actual tokens and headers for learning
                await limiter.report_success(actual_tokens, response_headers)

                try:
                    if "anthropic" in provider:
                        msg = resp["content"][0]["text"]
                    elif "google" in provider:
                        # { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
                        msg = resp["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        msg = resp["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    logger.error(
                        f"{provider} returned unexpected response format: {resp}"
                    )
                    raise Exception(
                        f"{provider} returned unexpected response: {e}"
                    ) from e

                logger.debug(f"Got response from {provider}: {msg}")
                return msg

        except asyncio.TimeoutError:
            logger.warning(f"{provider} request timed out")
            # Timeouts are also a sign of congestion, but we don't back off RPM for them
            # as it might just be a slow provider/model.
            await limiter.report_timeout()

            if retry_count < MAX_RETRIES:
                wait_time = (2**retry_count) * RETRY_BASE_SECONDS
                await asyncio.sleep(wait_time)
                return await self.async_get_chat_response(
                    prompt,
                    model,
                    provider.replace("_text", "")
                    if provider == "google_text"
                    else provider,  # type: ignore
                    -1,
                    temperature,
                    reasoning_effort,
                    retry_count + 1,
                )
            raise

        except aiohttp.ClientResponseError as e:
            # Log non-429 errors
            if e.status != 429:
                logger.warning(f"{provider} returned status {e.status}: {e.message}")

            # Only report failure to rate limiter for rate-limit related errors
            # 429 = Too Many Requests, 5xx = Server errors (might indicate overload)
            # Don't penalize rate limits for client errors like 400, 401, 403, 404
            if e.status == 429 or e.status >= 500:
                await limiter.report_failure()
            raise

        except Exception:
            # Any other exception? Maybe report failure if it's network related?
            # For safety, let's treat generic exceptions as potentially overload related
            # if we can distinguish them. But for now, just 429 and Timeout.
            raise

    def _extract_token_usage(self, response: dict, provider: str) -> int:
        """Extract total token usage from API response."""
        try:
            if "anthropic" in provider:
                # Anthropic: { "usage": { "input_tokens": X, "output_tokens": Y } }
                usage = response.get("usage", {})
                return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            elif "google" in provider:
                # Google: { "usageMetadata": { "promptTokenCount": X, "candidatesTokenCount": Y } }
                usage = response.get("usageMetadata", {})
                return usage.get("promptTokenCount", 0) + usage.get(
                    "candidatesTokenCount", 0
                )
            else:
                # OpenAI-style: { "usage": { "prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z } }
                usage = response.get("usage", {})
                return usage.get("total_tokens", 0) or (
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                )
        except Exception:
            return 0


chat_provider = ChatProvider()
