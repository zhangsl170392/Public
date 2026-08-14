import csv
import json
import time
from datetime import datetime
from lib2to3.fixes.fix_input import context

from scipy.stats.contingency import expected_freq

# 假设已有rag_qa函数
from rag_qa import rag_qa, load_retriever

def run_rag_evaluation(golden_csv_path, output_csv_path, top_k=3, persist_dir="./croma_db"):
    """对黄金问答对执行RAG评估，记录检索和生成结"""
    retriever = load_retriever(top_k=top_k)
    results = []
    with open(golden_csv_path, 'r', encoding="GBK") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row['question']
            expected_answer = row['expected_answer']
            expected_chunk_id = row.get('source_chunks', '')

            # 调用RAG系统，获取检索结果和生成答案
            # 假设 rag_qa 返回 (answer, retrieved_chunks, context)
            answers, docs = rag_qa(
                question,
                retriever
            )
            retrieved_chunk_ids = [str(doc.metadata.get('chunk_id', '')) for doc in docs]
            results.append({
                'question': question,
                'expected_answer': expected_answer,
                'expected_chunk_id': expected_chunk_id,
                'retrieved_chunk_ids': '|'.join(retrieved_chunk_ids),
                'generated_answers': answers,
                'context_used': docs,
                'timestamp': datetime.now().isoformat()
            })

            time.sleep(0.5)

    # 保存原始结果
    with open(output_csv_path, 'w', encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ 评估完成！共测试 {len(results)} 个问题，结果已保存至 {output_csv_path}")
    return results

def calculate_hit_rate(results, k=3):
    """计算Hit Rate@K"""
    hits = 0
    for r in results:
        expected_chunk_id = set(r.get('expected_chunk_id', '').split(','))
        retrieved_ids = r['retrieved_chunk_ids'].split('|')[:k]
        retrieved_ids = set(["chunk_" + r for r in retrieved_ids])
        if expected_chunk_id & retrieved_ids:
            hits += 1
    return hits / len(results) * 100


if __name__ == "__main__":
    results = run_rag_evaluation("golden_qa.csv", "rag_test_results.csv", top_k=3)
    hit_rate = calculate_hit_rate(results, k=3)
    print(f"Hit Rage@3: {hit_rate:.2f}%")

