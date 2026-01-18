import os
import sys
from pathlib import Path
import json
import pandas as pd
from difflib import SequenceMatcher

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
OPENAI_API_KEY = sk-MBnSXBiuwScGzqk60438F3513f5f4f719fD066CdE9Ee9010
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
# if not OPENAI_API_KEY:
#     raise RuntimeError("OPENAI_API_KEY 未设置，请在环境变量中配置。")

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

def _norm_text(t: str) -> str:
    """简单归一化：去首尾空格"""
    return (t or "").strip()

def _similarity(a: str, b: str) -> float:
    """基于 difflib 的软匹配相似度（0~1）"""
    a, b = _norm_text(a), _norm_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def _greedy_soft_overlap(sys_ctx_list, human_ctx_list, thresh=0.70):
    """
    在 system_contexts 与 human_contexts 之间做贪心匹配，
    每个 human 只可被匹配一次；返回匹配计数、精度、召回。
    """
    if not sys_ctx_list:
        return 0, 0.0, 0.0
    used = set()
    match_cnt = 0
    for s in sys_ctx_list:
        best_i, best_sim = -1, 0.0
        for i, h in enumerate(human_ctx_list or []):
            if i in used:
                continue
            sim = _similarity(s, h)
            if sim > best_sim:
                best_sim, best_i = sim, i
        if best_sim >= thresh and best_i >= 0:
            used.add(best_i)
            match_cnt += 1
    precision = match_cnt / max(1, len(sys_ctx_list))
    recall = match_cnt / max(1, len(human_ctx_list or []))
    return match_cnt, precision, recall

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
        _, prec, rec = _greedy_soft_overlap(system_contexts, human_contexts, thresh=0.70)
        row["contextual_precision_score"] = prec
        row["contextual_recall_score"] = rec
    except Exception as e:
        row["contextual_precision_score"] = None
        row["contextual_recall_score"] = None
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
