from __future__ import annotations

import ast
import json
import re

from .models import PlanAnswer

REQUIRED_KEYS = {
    "applicable_packets",
    "final_schedule",
    "moved_tasks",
    "confidence",
}
CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _candidate_payloads(response: str) -> list[str]:
    stripped = response.strip()
    candidates: list[str] = [stripped] if stripped else []
    candidates.extend(match.group(1).strip() for match in CODE_BLOCK_RE.finditer(response))
    return [candidate for candidate in candidates if candidate]


def _load_object(candidate: str) -> object | None:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[index:])
            return payload
        except json.JSONDecodeError:
            continue
    return None


def _coerce_list(value: object) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[0] == "[" and stripped[-1] == "]":
            parsed = _load_object(stripped)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        parts = [part.strip() for part in re.split(r"[\n,]", stripped)]
        if len(parts) == 1:
            parts = [part.strip() for part in stripped.split()]
        return [part for part in parts if part]
    return [str(value).strip()] if str(value).strip() else []


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match:
            return int(match.group(0))
    return default


def _coerce_plan_answer(payload: dict[str, object]) -> PlanAnswer | None:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    if not REQUIRED_KEYS.issubset(lowered):
        return None
    return PlanAnswer(
        applicable_packets=_coerce_list(lowered["applicable_packets"]),
        final_schedule=_coerce_list(lowered["final_schedule"]),
        moved_tasks=_coerce_int(lowered["moved_tasks"]),
        confidence=max(0, min(100, _coerce_int(lowered["confidence"]))),
    )


def parse_plan_answer_response(response: str) -> PlanAnswer | None:
    for candidate in _candidate_payloads(response):
        loaded = _load_object(candidate)
        if isinstance(loaded, dict):
            answer = _coerce_plan_answer(loaded)
            if answer is not None:
                return answer

    extracted_payload: dict[str, object] = {}
    for key in REQUIRED_KEYS:
        pattern = re.compile(
            rf"{key}\s*[:=]\s*(.+?)(?=\n\s*[A-Za-z_]+\s*[:=]|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(response)
        if match:
            extracted_payload[key] = match.group(1).strip()
    if extracted_payload:
        return _coerce_plan_answer(extracted_payload)
    return None
