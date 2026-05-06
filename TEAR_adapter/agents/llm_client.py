from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional


class StructuredLLMClient:
    def __init__(self, endpoint: Optional[str] = None, timeout_seconds: int = 120):
        self.endpoint = endpoint or os.environ.get("LOCAL_LLM_ENDPOINT")
        self.timeout_seconds = timeout_seconds

    def available(self) -> bool:
        return bool(self.endpoint)

    def chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> dict:
        if not self.endpoint:
            raise RuntimeError("LOCAL_LLM_ENDPOINT is not configured.")
        prompt = self._build_instruction_prompt(system_prompt, user_prompt)
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.05,
        }
        request = urllib.request.Request(
            url=f"{self.endpoint.rstrip('/')}/v1/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                content = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error}") from error
        text = content["choices"][0]["text"]
        return self._extract_json("{" + text if text.strip() and not text.lstrip().startswith("{") else text)

    @staticmethod
    def _build_instruction_prompt(system_prompt: str, user_prompt: str) -> str:
        return (
            "<s>[INST] <<SYS>>\n"
            f"{system_prompt}\n"
            "Output exactly one minified JSON object. No prose. No markdown. No examples.\n"
            "<</SYS>>\n"
            f"{user_prompt}\n"
            "Return only JSON. The first character must be { and the last character must be }.\n"
            "[/INST]{"
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        start = text.find("{")
        if start == -1:
            partial = StructuredLLMClient._extract_partial_object(text)
            if partial is not None:
                return partial
            raise ValueError(f"Expected JSON object in LLM response, received: {text}")
        depth = 0
        end = -1
        in_string = False
        escape = False
        for index, char in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == "\"":
                    in_string = False
                continue
            if char == "\"":
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end == -1:
            partial = StructuredLLMClient._extract_partial_object(text[start:])
            if partial is not None:
                return partial
            raise ValueError(f"Expected balanced JSON object in LLM response, received: {text}")
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            partial = StructuredLLMClient._extract_partial_object(text[start:end + 1])
            if partial is not None:
                return partial
            raise

    @staticmethod
    def _extract_partial_object(text: str) -> dict | None:
        if '"feature_updates"' in text:
            result: dict = {}
            feature_updates = StructuredLLMClient._extract_json_array_after_key(text, "feature_updates")
            if feature_updates is not None:
                result["feature_updates"] = feature_updates
            ordering_pairs = StructuredLLMClient._extract_json_array_after_key(text, "ordering_pairs")
            if ordering_pairs is not None:
                result["ordering_pairs"] = ordering_pairs
            unit_risk_scale = StructuredLLMClient._extract_numeric_after_key(text, "unit_risk_scale")
            confidence = StructuredLLMClient._extract_numeric_after_key(text, "confidence")
            if unit_risk_scale is not None:
                result["unit_risk_scale"] = unit_risk_scale
            if confidence is not None:
                result["confidence"] = confidence
            return result if "feature_updates" in result else None
        if '"feature_roles"' in text:
            result = {}
            feature_roles = StructuredLLMClient._extract_json_array_after_key(text, "feature_roles")
            if feature_roles is not None:
                result["feature_roles"] = feature_roles
            ordering_pairs = StructuredLLMClient._extract_json_array_after_key(text, "ordering_pairs")
            if ordering_pairs is not None:
                result["ordering_pairs"] = ordering_pairs
            return result if "feature_roles" in result else None
        return None

    @staticmethod
    def _extract_json_array_after_key(text: str, key: str):
        token = f'"{key}"'
        key_index = text.find(token)
        if key_index == -1:
            return None
        array_start = text.find("[", key_index)
        if array_start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        array_end = -1
        for index, char in enumerate(text[array_start:], start=array_start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == "\"":
                    in_string = False
                continue
            if char == "\"":
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    array_end = index
                    break
        if array_end == -1:
            return None
        try:
            return json.loads(text[array_start:array_end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_numeric_after_key(text: str, key: str) -> float | None:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if not match:
            return None
        return float(match.group(1))
