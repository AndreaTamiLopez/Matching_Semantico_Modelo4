import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

from .utils import clean_text
from .llm_rerank import ollama_is_up, ollama_rerank
from .cache import cache_key, cache_load, cache_save
from .embeddings import build_embeddings, knn_retrieve


def match_proyecto_to_politicas_optimizado(
    df_politicas: pd.DataFrame,
    df_proyectos: pd.DataFrame,
    col_text_politica: str,
    col_text_proyecto: str,
    col_id_proyecto: str,

    # opcional: id estable para cache (si no se pasa, usa el índice)
    col_id_politica: Optional[str] = None,

    # salida
    top_k: int = 10,

    # embeddings
    embed_model: str = "BAAI/bge-m3",
    batch_size: int = 16,
    top_n_candidates: int = 200,
    min_bi_score: float = 0.25,

    # gating
    confident_top1_threshold: float = 0.42,
    confident_margin_threshold: float = 0.06,

    # LLM
    use_llm_rerank: bool = True,
    ollama_url: str = "http://localhost:11434/api/generate",
    llm_model: str = "deepseek-r1:7b",
    llm_timeout_sec: int = 240,
    llm_temperature: float = 0.0,
    llm_candidates_cap: int = 40,
    top_k_llm: int = 5,

    # cache
    cache_dir: str = "./ollama_rerank_cache",

    # score final
    w_llm: float = 0.65,
    w_bi: float = 0.35,

    # logging
    verbose_every: int = 200
) -> pd.DataFrame:

    # Validaciones
    if col_text_politica not in df_politicas.columns:
        raise ValueError(f"df_politicas no tiene columna: {col_text_politica}")
    for c in [col_text_proyecto, col_id_proyecto]:
        if c not in df_proyectos.columns:
            raise ValueError(f"df_proyectos no tiene columna: {c}")
    if col_id_politica is not None and col_id_politica not in df_politicas.columns:
        raise ValueError(f"df_politicas no tiene col_id_politica: {col_id_politica}")

    if use_llm_rerank and not ollama_is_up():
        print("⚠️ Ollama no accesible. Continuo SOLO con embeddings.")
        use_llm_rerank = False

    # Textos
    pol_texts_raw = df_politicas[col_text_politica].fillna("").astype(str).map(clean_text).tolist()
    proy_texts_raw = df_proyectos[col_text_proyecto].fillna("").astype(str).map(clean_text).tolist()

    # Device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    # Modelo embeddings
    bi = SentenceTransformer(embed_model, device=device)

    E_pol = build_embeddings(pol_texts_raw, bi, batch_size=batch_size, model_name=embed_model, mode="query")
    E_proy = build_embeddings(proy_texts_raw, bi, batch_size=batch_size, model_name=embed_model, mode="passage")

    # Recuperación
    top_n_candidates = max(top_k, int(top_n_candidates))
    _, indices, bi_scores = knn_retrieve(E_pol, E_proy, top_n_candidates=top_n_candidates)

    out_rows: List[Dict[str, Any]] = []
    llm_calls = 0
    llm_used = 0

    for i in range(len(df_proyectos)):
        if verbose_every and (i % verbose_every == 0) and i > 0:
            print(f"[{i}/{len(df_proyectos)}] LLM calls={llm_calls}, LLM used={llm_used}")

        base = df_proyectos.iloc[i].to_dict()
        project_text = proy_texts_raw[i]

        # candidatos por embeddings
        candidates: List[Tuple[int, str, float]] = []
        for r in range(min(top_n_candidates, indices.shape[1])):
            j = int(indices[i, r])
            sc = float(bi_scores[i, r])
            if sc < float(min_bi_score):
                continue
            candidates.append((j, pol_texts_raw[j], sc))

        if not candidates:
            continue

        # gating
        top1 = float(candidates[0][2])
        top2 = float(candidates[1][2]) if len(candidates) > 1 else 0.0
        margin = float(top1 - top2)
        is_confident = (top1 >= confident_top1_threshold) and (margin >= confident_margin_threshold)

        selections = []
        llm_failed = False
        llm_fail_reason = ""

        # LLM solo si difícil
        if use_llm_rerank and (not is_confident):
            llm_used += 1
            candidates_for_llm = candidates[: min(len(candidates), int(llm_candidates_cap))]

            # ids estables para cache
            if col_id_politica is None:
                candidate_ids = [j for (j, _, _) in candidates_for_llm]
            else:
                candidate_ids = [df_politicas.iloc[j][col_id_politica] for (j, _, _) in candidates_for_llm]

            key = cache_key(project_text, candidate_ids, llm_model, min(top_k_llm, top_k))
            cached = cache_load(cache_dir, key)

            if cached is not None:
                selections = cached.get("selections", []) or []
            else:
                llm_calls += 1
                try:
                    llm_json = ollama_rerank(
                        politica_candidates=candidates_for_llm,
                        project_text=project_text,
                        top_k_llm=min(top_k_llm, top_k),
                        ollama_url=ollama_url,
                        model_name=llm_model,
                        temperature=llm_temperature,
                        timeout_sec=llm_timeout_sec
                    )
                    cache_save(cache_dir, key, llm_json)
                    selections = llm_json.get("selections", []) or []
                except Exception as e:
                    llm_failed = True
                    llm_fail_reason = f"{type(e).__name__}: {str(e)[:120]}"
                    selections = []

        # construir top_k
        used_policy = set()
        rank = 0

        # 1) picks LLM
        if selections:
            candidates_for_llm = candidates[: min(len(candidates), int(llm_candidates_cap))]
            for sel in selections:
                try:
                    cand_idx = int(sel["candidate_index"])
                    s_llm = float(sel["score"])
                    r_llm = str(sel["reason"])
                except Exception:
                    continue
                if cand_idx < 0 or cand_idx >= len(candidates_for_llm):
                    continue

                pol_j, _, s_bi = candidates_for_llm[cand_idx]
                if pol_j in used_policy:
                    continue

                used_policy.add(pol_j)
                rank += 1

                final_score = w_llm * s_llm + w_bi * float(s_bi)

                out_rows.append({
                    **base,
                    "matched_politica_text": df_politicas.iloc[pol_j][col_text_politica],
                    "bi_similarity_score": float(s_bi),
                    "llm_score": float(s_llm),
                    "llm_reason": r_llm,
                    "final_score": float(final_score),
                    "rank": rank,
                    "used_llm": True,
                    "llm_failed": False,
                    "confident_by_embeddings": False,
                    "top1_bi": float(top1),
                    "margin_top1_top2": float(margin),
                    "device_used": device
                })

                if rank >= top_k:
                    break

        # 2) completar con embeddings
        if rank < top_k:
            fallback = sorted(candidates, key=lambda x: x[2], reverse=True)
            for (pol_j, _, s_bi) in fallback:
                if rank >= top_k:
                    break
                if pol_j in used_policy:
                    continue

                used_policy.add(pol_j)
                rank += 1

                reason = "accepted_by_embeddings" if is_confident else "fallback_by_embeddings"
                if llm_failed:
                    reason = f"fallback_by_embeddings (llm_failed: {llm_fail_reason})"

                out_rows.append({
                    **base,
                    "matched_politica_text": df_politicas.iloc[pol_j][col_text_politica],
                    "bi_similarity_score": float(s_bi),
                    "llm_score": np.nan,
                    "llm_reason": reason,
                    "final_score": float(s_bi),
                    "rank": rank,
                    "used_llm": (use_llm_rerank and (not is_confident)),
                    "llm_failed": bool(llm_failed),
                    "confident_by_embeddings": bool(is_confident),
                    "top1_bi": float(top1),
                    "margin_top1_top2": float(margin),
                    "device_used": device
                })

    df_out = pd.DataFrame(out_rows)
    df_out = df_out.sort_values([col_id_proyecto, "rank"], ascending=[True, True]).reset_index(drop=True)

    print(f"✅ Terminado. Proyectos={len(df_proyectos)} | LLM calls={llm_calls} | LLM used={llm_used} | cache_dir={cache_dir}")
    return df_out
