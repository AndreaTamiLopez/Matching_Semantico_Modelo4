import os
import json
import hashlib
from typing import Any, Dict, List, Optional


def cache_key(project_text: str, candidate_policy_ids: List[Any], model_name: str, top_k_llm: int) -> str:
    h = hashlib.sha256()
    h.update((model_name or "").encode("utf-8"))
    h.update(str(top_k_llm).encode("utf-8"))
    h.update((project_text or "").encode("utf-8"))
    h.update(("|".join(map(str, candidate_policy_ids))).encode("utf-8"))
    return h.hexdigest()


def cache_load(cache_dir: str, key: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def cache_save(cache_dir: str, key: str, obj: Dict[str, Any]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
