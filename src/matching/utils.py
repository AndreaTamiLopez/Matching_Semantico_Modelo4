import os
import re
import json
import numpy as np
from typing import Any, Dict, List


def enable_hf_ssl_fix() -> str:
    """
    Si HuggingFace falla con SSLCertVerificationError, ejecuta esto ANTES
    de cargar SentenceTransformer.
    """
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    return certifi.where()


def clean_text(s: Any) -> str:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def format_for_model(texts: List[str], mode: str, model_name: str) -> List[str]:
    ml = (model_name or "").lower()
    if ("e5" in ml) or ("bge" in ml):
        prefix = "query: " if mode == "query" else "passage: "
        return [prefix + t for t in texts]
    return texts


def extract_json_loose(text: str) -> Dict[str, Any]:
    """
    Extrae JSON aunque el modelo meta texto extra.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No se encontró JSON en respuesta del LLM.")
    return json.loads(m.group(0).strip())


def validate_llm_schema(obj: Dict[str, Any]) -> None:
    """
    Espera:
      {"selections":[{"candidate_index":int,"score":float 0-1,"reason":str}, ...]}
    """
    if "selections" not in obj or not isinstance(obj["selections"], list) or len(obj["selections"]) == 0:
        raise ValueError("LLM: selections inválido/vacío.")

    for it in obj["selections"]:
        for k in ["candidate_index", "score", "reason"]:
            if k not in it:
                raise ValueError(f"LLM: falta {k}.")
        sc = float(it["score"])
        if sc < 0 or sc > 1:
            raise ValueError("LLM: score fuera de [0,1].")
