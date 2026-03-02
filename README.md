# LLM Gated Semantic Matching

Sistema híbrido de emparejamiento semántico entre descripciones de proyectos y catálogos de políticas públicas utilizando embeddings con re-ranking opcional mediante LLM.

---

## 📌 Descripción General

Este proyecto implementa un pipeline optimizado que asigna, para cada proyecto, las políticas más alineadas conceptualmente.

El sistema combina:

- Recuperación por embeddings (Bi-Encoder)
- Mecanismo de gating por confianza
- Re-ranking opcional con LLM local (Ollama)
- Fallback automático si el LLM no está disponible
- Cache persistente en disco
- Exportación automática a Excel

---

## 🧠 Funcionamiento

### Flujo del pipeline

```
Proyecto
   ↓
Generación de embeddings
   ↓
Recuperación Top-N (similitud coseno)
   ↓
Evaluación de confianza
```

Si la confianza es alta:

→ Se aceptan directamente los resultados de embeddings

Si la confianza es baja:

→ Se llama al LLM para re-ranking  
→ Se combinan los scores

---

## 📊 Fórmula del Score Final

```
final_score = w_llm * llm_score + w_bi * bi_score
```

Si el LLM está desactivado o no disponible:

```
final_score = bi_score
```

---

## 📁 Estructura del Proyecto

```
src/matching/
├── utils.py
├── embeddings.py
├── llm_rerank.py
├── cache.py
├── matcher.py
└── export.py

scripts/
└── run_matching.py

data/
├── raw/
└── outputs/
```

---

## ⚙️ Instalación

### Crear entorno virtual

**Windows**
```
python -m venv .venv
.venv\Scripts\activate
```

**Mac o Linux**
```
python -m venv .venv
source .venv/bin/activate
```

### Instalar dependencias

```
pip install -r requirements.txt
```

---

## ▶️ Ejecución

Desde la raíz del proyecto:

**Windows**
```
set PYTHONPATH=src
python scripts\run_matching.py
```

**Mac o Linux**
```
PYTHONPATH=src python scripts/run_matching.py
```

---

## 📂 Output

El archivo de salida se genera en:

```
data/outputs/matching_topk_modelo3.xlsx
```

---

## 🔧 Parámetros Clave

### Embeddings

- embed_model
- batch_size
- top_n_candidates
- min_bi_score

### Gating de Confianza

- confident_top1_threshold
- confident_margin_threshold

### LLM

- use_llm_rerank
- llm_model
- llm_candidates_cap
- top_k_llm
- llm_timeout_sec

### Pesos del Score

- w_llm
- w_bi

---

## 🤖 Configuración Opcional de LLM (Ollama)

Descargar modelo:

```
ollama pull deepseek-r1:7b
```

Verificar servidor:

```
curl http://localhost:11434/api/tags
```

---

## 📑 Columnas del Resultado

El DataFrame final incluye:

- matched_politica_text
- bi_similarity_score
- llm_score
- final_score
- rank
- used_llm
- confident_by_embeddings
- top1_bi
- margin_top1_top2
- device_used

---

## 🛠 Stack Tecnológico

- Python
- SentenceTransformers
- PyTorch
- scikit-learn
- Ollama (opcional)
- Pandas
- XlsxWriter
