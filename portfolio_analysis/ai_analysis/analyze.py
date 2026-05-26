"""
The single batched-call entry point for executing LLM commentary generation.
Includes caching, schema parsing, and verification routines.
"""

import json
import hashlib
from typing import Dict, List, Any
from google.genai import types

from portfolio_analysis.ai_analysis.schema import ProfileSnapshot, ItemComment
from portfolio_analysis.ai_analysis.config import get_client, DEFAULT_MODEL, DEFAULT_MAX_TOKENS
from portfolio_analysis.ai_analysis.prompts import SYSTEM_PROMPT

# In-memory cache for stateless commentaries
_COMMENTARY_CACHE: Dict[str, List[ItemComment]] = {}

def _compute_snapshot_hash(snapshot: ProfileSnapshot) -> str:
    """Compute a stable hash for a ProfileSnapshot to use as a cache key."""
    serialized_items = []
    for item in sorted(snapshot.items, key=lambda x: x.item_id):
        serialized_items.append({
            "item_id": item.item_id,
            "item_type": item.item_type,
            "title": item.title,
            "data": item.data,
            "phase": item.phase
        })
    
    payload = {
        "as_of_date": snapshot.as_of_date,
        "base_currency": snapshot.base_currency,
        "items": serialized_items,
        "portfolio_summary": snapshot.portfolio_summary
    }
    
    stable_json = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(stable_json.encode('utf-8')).hexdigest()

def clear_commentary_cache() -> None:
    """Utility to clear the global memory cache."""
    global _COMMENTARY_CACHE
    _COMMENTARY_CACHE.clear()

def analyze_profile(snapshot: ProfileSnapshot) -> List[ItemComment]:
    """One batched LLM call. Returns one ItemComment per input item.

    Triggered by the frontend's 'AI Analysis' button. Stateless.
    """
    if not snapshot.items:
        return []

    # 1. Cache lookup
    cache_key = _compute_snapshot_hash(snapshot)
    if cache_key in _COMMENTARY_CACHE:
        return _COMMENTARY_CACHE[cache_key]

    # 2. Build prompt context
    items_payload = []
    for item in snapshot.items:
        if not item.data:
            items_payload.append({
                "item_id": item.item_id,
                "item_type": item.item_type,
                "title": item.title,
                "data": {},
                "status_hint": "insufficient data"
            })
        else:
            items_payload.append({
                "item_id": item.item_id,
                "item_type": item.item_type,
                "title": item.title,
                "data": item.data
            })

    user_prompt = {
        "portfolio_summary": snapshot.portfolio_summary,
        "items": items_payload
    }
    
    user_content = (
        f"Analyze the following portfolio snapshot and return one comment per item in the specified JSON format.\n\n"
        f"SNAPSHOT:\n{json.dumps(user_prompt, indent=2)}"
    )

    # 3. Call Gemini API
    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=DEFAULT_MAX_TOKENS,
        temperature=0.1,  # Low temperature to avoid numeric hallucinations
    )

    def attempt_call(extra_instruction: str = "") -> str:
        contents = user_content
        if extra_instruction:
            contents += f"\n\nCRITICAL FIX: {extra_instruction}"
        
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=contents,
            config=config,
        )
        return response.text or ""

    # Parse JSON helper
    def parse_json(text: str) -> List[dict]:
        clean_text = text.strip()
        # Defensive check and cleanup for markdown JSON fences
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()
        return json.loads(clean_text)

    # Execute and attempt JSON parsing with a single fallback retry
    response_text = ""
    try:
        response_text = attempt_call()
        parsed_comments = parse_json(response_text)
    except Exception:
        # Retry once defensively on parse failure
        try:
            retry_text = attempt_call(
                "Your previous response was not valid JSON. Please return ONLY a valid, raw JSON list of objects, without markdown formatting."
            )
            parsed_comments = parse_json(retry_text)
        except Exception:
            # Complete failure: synthesize error comments for the UI instead of throwing
            error_comments = [
                ItemComment(item_id=item.item_id, comment="Error: Failed to generate or parse AI analysis.", status="error")
                for item in snapshot.items
            ]
            return error_comments

    # 4. Map parsed comments
    comment_map: Dict[str, str] = {}
    if isinstance(parsed_comments, list):
        for item_data in parsed_comments:
            if isinstance(item_data, dict) and "item_id" in item_data and "comment" in item_data:
                comment_map[item_data["item_id"]] = item_data["comment"]

    # 5. Verify every input item gets exactly one comment
    output_comments = []
    for item in snapshot.items:
        item_id = item.item_id
        if not item.data:
            output_comments.append(ItemComment(
                item_id=item_id,
                comment=comment_map.get(item_id, "Insufficient data to perform analysis for this item."),
                status="skipped_no_data"
            ))
        elif item_id in comment_map:
            output_comments.append(ItemComment(
                item_id=item_id,
                comment=comment_map[item_id],
                status="ok"
            ))
        else:
            # Handle missing outputs gracefully via synthesized errors
            output_comments.append(ItemComment(
                item_id=item_id,
                comment="Error: No commentary returned for this item.",
                status="error"
            ))

    # 6. Cache successful responses
    if all(c.status != "error" for c in output_comments):
        _COMMENTARY_CACHE[cache_key] = output_comments

    return output_comments
