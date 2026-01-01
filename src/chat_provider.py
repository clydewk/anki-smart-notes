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
        base_url = provider_config["base_url"]
        
        # Ensure base_url ends with v1/chat/completions if not present?
        # Standard OpenAI compatible usually requires the full URL or base? 
        # The user input "Base URL" usually implies the root. 
        # But for OpenAI it is https://api.openai.com/v1/chat/completions.
        # If user puts "http://localhost:11434/v1", we might need to append /chat/completions?
        # Let's assume user provides the FULL endpoint URL for simplicity and maximum flexibility,
        # OR we try to be smart.
        # Let's assume it IS the endpoint for chat completions.
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

        return await self._execute_request(
            url=url,
            headers=headers,
            json_payload=payload,
            timeout_sec=CHAT_CLIENT_TIMEOUT_SEC,
            retry_count=retry_count,
            provider=provider_config["name"],
            prompt=prompt,
            model=model,  # type: ignore
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
            if not current_effort or current_effort not in efforts:
                if "none" in efforts:
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
            raise Exception(
                "Google API key not found. Please set it in the settings."
            )
        
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
        # I'll stick to passing it if it matches the config, but users might lower it.
        # Let's pass it for now as Smart Notes allows configuration.
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature
            }
        }

        return await self._execute_request(
            url=url,
            headers=headers,
            json_payload=payload,
            timeout_sec=CHAT_CLIENT_TIMEOUT_SEC,
            retry_count=retry_count,
            provider="google",
            prompt=prompt,
            model=model,
            temperature=temperature,
            reasoning_effort=None,
        )

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
        try:
            async with (
                aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout_sec)
                ) as session,
                session.post(url, headers=headers, json=json_payload) as response,
            ):
                if response.status == 429:
                    logger.debug(f"Got a 429 from {provider}")
                    if retry_count < MAX_RETRIES:
                        wait_time = (2**retry_count) * RETRY_BASE_SECONDS
                        logger.debug(
                            f"Retry: {retry_count} Waiting {wait_time} seconds before retrying"
                        )
                        await asyncio.sleep(wait_time)
                        # Recursively call the public method to retry logic
                        return await self.async_get_chat_response(
                            prompt,
                            model,
                            provider,  # type: ignore
                            -1,
                            temperature,
                            reasoning_effort,
                            retry_count + 1,
                        )

                response.raise_for_status()
                resp = await response.json()

                if provider == "anthropic":
                    msg = resp["content"][0]["text"]
                elif provider == "google":
                    # { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
                    msg = resp["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    msg = resp["choices"][0]["message"]["content"]

                logger.debug(f"Got response from {provider}: {msg}")
                return msg

        except asyncio.TimeoutError:
            logger.warning(f"{provider} request timed out")
            if retry_count < MAX_RETRIES:
                wait_time = (2**retry_count) * RETRY_BASE_SECONDS
                await asyncio.sleep(wait_time)
                return await self.async_get_chat_response(
                    prompt,
                    model,
                    provider,  # type: ignore
                    -1,
                    temperature,
                    reasoning_effort,
                    retry_count + 1,
                )
            raise


chat_provider = ChatProvider()
