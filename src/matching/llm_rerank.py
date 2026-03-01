import time
import requests
from typing import Any, Dict, List, Tuple

from .utils import extract_json_loose, validate_llm_schema


def ollama_is_up(url: str = "http://localhost:11434/api/tags", timeout: int = 3) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def ollama_rerank(
    politica_candidates: List[Tuple[int, str, float]],
    project_text: str,
    top_k_llm: int,
    ollama_url: str,
    model_name: str,
    temperature: float,
    timeout_sec: int,
    max_retries: int = 2
) -> Dict[str, Any]:
    candidates_txt = "\n".join([
        f"{i}. (bi={bi_score:.3f}) {pol_txt}"
        for i, (_, pol_txt, bi_score) in enumerate(politica_candidates)
    ])

    prompt = f"""
Selecciona las {top_k_llm} políticas más alineadas con el proyecto.
Criterios: mismo objetivo, población, instrumento y sector. Prefiere la más específica.

PROYECTO:
{project_text}

POLÍTICAS CANDIDATAS:
{candidates_txt}

Devuelve SOLO JSON:
{{
  "selections": [
    {{"candidate_index": 0, "score": 0.0, "reason": "1 frase"}}
  ]
}}
""".strip()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(ollama_url, json=payload, timeout=timeout_sec)
            if r.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
            obj = extract_json_loose(r.json().get("response", ""))
            validate_llm_schema(obj)
            return obj
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 6))

    raise RuntimeError(f"LLM rerank falló. Último error: {last_err}")
