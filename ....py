import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

golden_df = pd.read_csv("./golden_qa.csv", encoding="GBK")

from rag_qa import rag_qa
from rag_qa import load_retriever

retriever = load_retriever(persist_dir="./chroma_db", top_k=3)

import csv
import time
from tqdm import tqdm

results = []
for idx, row in tqdm(golden_df.iterrows(), total=len(golden_df)):
    question = row["question"]
    expected = row["expected_answer"]
    source_ids = [s.strip() for s in str(row["source_chunks"]).split(",") if s.strip() != "N/A"]
    category = row["category"]

    # 调用RAG
    start = time.time()
    answer, retriever_docs = rag_qa(question, retriever)
    elapsed = time.time() - start

    # 提取检索到的文档ID（假设每个文档有 metadata["id"]）
    retrieved_ids = [doc.metadata.get("chunk_id", 'unknown') for doc in retriever_docs]

    # 计算 Hit Rate（至少一个正确片段命中）
    hit = 1 if any(f"chunk_{rid}" in source_ids for rid in retrieved_ids) else 0

    results.append({
        "id": row["id"],
        "question": question,
        "expected_answer": expected,
        "category": category,
        "retriever_ids": ";".join(map(str, retrieved_ids)),
        "source_ids": ";".join(source_ids),
        "hit": hit,
        "answer": answer,
        "time": round(elapsed, 2)
    })

df_result = pd.DataFrame(results)
df_result.to_csv("rag_test_raw.csv", index=False, encoding="utf-8-sig")