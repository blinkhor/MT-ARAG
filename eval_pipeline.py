import os
import sys
from pathlib import Path
import json
import pandas as pd
from difflib import SequenceMatcher
import numpy as np
from openai import OpenAI as OAClient

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

# ========== 路径与依赖 ==========
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import RAGConfig
from improved_main import MusicTherapyRAG

# ========== Step 0: 初始化 RAG 系统 ==========
# 不要把密钥硬编码到仓库里；优先从环境变量读取
OPENAI_API_KEY = "sk-MBnSXBiuwScGzqk60438F3513f5f4f719fD066CdE9Ee9010"
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# if not OPENAI_API_KEY:
#     raise RuntimeError("OPENAI_API_KEY 未设置，请在环境变量中配置。")
_EMBED_MODEL = "text-embedding-3-small"
_client = OAClient(api_key=OPENAI_API_KEY, base_url="https://api.vveai.com/v1")

config = RAGConfig()
rag_system = MusicTherapyRAG(openai_api_key=OPENAI_API_KEY, config=config)
rag_system.initialize_system()

# ========== Step 1: 加载评估数据集 ==========
DATASET_PATH = Path(__file__).resolve().parent / "eval_dataset.json"
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# ========== Step 2: 定义评估器与工具函数 ==========
# 说明：如果你的代理网关仅支持特定模型名，请按需调整 model / api_base
llm = OpenAI(model="gpt-4", api_base="https://api.vveai.com/v1")

def _embed_texts(texts):
    texts = [t if (t and t.strip()) else " " for t in texts]
    resp = _client.embeddings.create(model=_EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)

def _cosine_sim_matrix(A, B):
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T

def semantic_any_overlap(sys_ctx_list, human_ctx_list, query_text, thresh=0.60):
    """
    语义“任意命中”评估（不做贪心匹配）
    返回三个值：
      - precision：检索出的文档中与“人工相关文档集”任一条相似度 ≥ 阈值 的比例
      - recall   ：人工相关文档集中，被至少一条检索文档以 ≥ 阈值 命中的比例
      - relevancy：检索文档与问题(query)的语义相似度“平均分”（所有 system 段对 query 的余弦相似度取均值）
    """
    n_sys = len(sys_ctx_list or [])
    n_hum = len(human_ctx_list or [])

    if n_sys == 0:
        # 无检索文档时，三个指标都给 0.0（也可按需返回 None）
        return 0.0, 0.0, 0.0
    if n_hum == 0:
        # 无人工标注时，Precision/Recall 无法衡量覆盖，置 0；Relevancy 仍可按 query 计算
        A = _embed_texts(sys_ctx_list)
        q = _embed_texts([query_text])
        rel = float((_cosine_sim_matrix(A, q)[:, 0]).mean())
        return 0.0, 0.0, rel

    # 嵌入 system / human / query
    A = _embed_texts(sys_ctx_list)       # (n_sys, d)
    B = _embed_texts(human_ctx_list)     # (n_hum, d)
    q = _embed_texts([query_text])       # (1, d)

    # --- Precision / Recall（任意命中逻辑）---
    S = _cosine_sim_matrix(A, B)         # (n_sys, n_hum)
    sys_hits = (S.max(axis=1) >= thresh).sum()
    precision = sys_hits / n_sys

    hum_hits = (S.max(axis=0) >= thresh).sum()
    recall = hum_hits / n_hum

    # --- Relevancy：系统上下文对 Query 的平均相似度 ---
    rel = float((_cosine_sim_matrix(A, q)[:, 0]).mean())

    return precision, recall, rel


evaluators = {
    "relevancy": RelevancyEvaluator(llm=llm),
    "faithfulness": FaithfulnessEvaluator(llm=llm),
}

# ========== Step 3: 跑评估 ==========
results = []
for item in dataset:
    query = item["query"]
    ground_truth = item.get("ground_truth", "")
    human_contexts = item.get("relevant_contexts", []) or []

    # === 调用 RAG 系统 ===
    response = rag_system.query_engine_lit.query(query)
    predicted_answer = str(response)

    # === 提取系统检索到的上下文（容错：可能没有 source_nodes）===
    try:
        system_contexts = [n.node.get_content() for n in (response.source_nodes or [])]
    except Exception:
        system_contexts = []

    row = {
        "query": query,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "system_contexts": system_contexts,
        "human_contexts": human_contexts,
    }

    # --- 软匹配的 Precision & Recall ---
    try:
        prec, rec, rel = semantic_any_overlap(
            system_contexts,
            human_contexts,
            query_text=query,
            thresh=0.60
        )
        row["contextual_precision_score"] = prec
        row["contextual_recall_score"] = rec
        row["contextual_relevancy_score"] = rel
    except Exception as e:
        row["contextual_precision_score"] = None
        row["contextual_recall_score"] = None
        row["contextual_relevancy_score"] = None
        row["contextual_match_error"] = str(e)

    # --- LLM 评估器（注意：要用属性 res.score / res.feedback；relevancy 也需 contexts）---
    for name, evaluator in evaluators.items():
        try:
            if name == "relevancy":
                res = evaluator.evaluate(
                    query=query,
                    response=predicted_answer,
                    contexts=system_contexts,  # 关键：补 contexts
                )
            elif name == "faithfulness":
                res = evaluator.evaluate(
                    query=query,
                    response=predicted_answer,
                    contexts=system_contexts,
                )
            else:
                res = None

            if res is not None:
                # llama_index.core.evaluation.EvaluationResult
                row[name + "_score"] = getattr(res, "score", None)
                row[name + "_feedback"] = getattr(res, "feedback", None)
            else:
                row[name + "_score"] = None
                row[name + "_feedback"] = "Not implemented"
        except Exception as e:
            row[name + "_score"] = None
            row[name + "_feedback"] = str(e)

    results.append(row)

# ========== Step 4: 保存结果 ==========
df = pd.DataFrame(results)
df.to_csv("eval_results.csv", index=False, encoding="utf-8-sig")
print("✅ Evaluation finished. Results saved to eval_results.csv")
