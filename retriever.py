import os
import ssl
import certifi
import pandas as pd


# 1. 设置环境变量（辅助作用）
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['CURL_CA_BUNDLE'] = certifi.where()

# 2. 保存原始函数
_original_create_default_context = ssl.create_default_context

# 3. 定义新的函数，强制使用 certifi 的证书文件，从而绕过 Windows 存储
def _patched_create_default_context(*args, **kwargs):
    # 如果调用者没有指定任何证书源，则强制指定 certifi 的证书文件
    if 'cafile' not in kwargs and 'capath' not in kwargs and 'cadata' not in kwargs:
        kwargs['cafile'] = certifi.where()
    return _original_create_default_context(*args, **kwargs)

# 4. 替换全局函数
ssl.create_default_context = _patched_create_default_context

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

class DocumentRetriever:
    def __init__(self, persist_dir="./chroma_db", model="text-embedding-v1"):
        self.embeddings = DashScopeEmbeddings(model=model)
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
            relevance_score_fn= self._distance_to_relevance
        )
        # 为 Chroma 注入自定义的相关性分数函数
        # 将底层返回的距离（越小越相似）转换为 [0,1] 的相似度（越大越相似）
        # self.vectorstore._relevance_score_fn = self._distance_to_relevance

    @staticmethod
    def _distance_to_relevance(score: float) -> float:
        """
                将距离转换为 [0,1] 区间的相似度分数。
                适用于余弦距离（0~2）或欧氏距离（>=0），距离越小 -> 相似度越高。
                使用 1/(1+distance) 保证单调递减且输出范围 (0,1]。
        """
        if score < 0:
            distance = -score
        else:
            distance = score  # 如果已经是正距离
        return 1.0 / (1.0 + distance)

    def basic_search(self, query, k=3):
        """普通检索，返回文档列表"""
        return self.vectorstore.similarity_search(query, k=k)

    def search_with_score(self, query, k=3):
        """带分数的检索"""
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def threshold_search(self, query, k=3, score_threshold=0.6):
        """带相关性阈值的检索（需要归一化分数）"""
        docs = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, score_threshold=score_threshold
        )
        return docs

    def get_retriever(self, search_type="similarity", k=3, score_threshold=None):
        """返回一个标准 Retriever 对象，供 LCEL 使用"""
        search_kwargs = {"k": k}
        if search_type == "similarity_score_threshold" and score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

if __name__ == "__main__":
    retriever = DocumentRetriever()
    query = "今天天气怎么样？"
    print("=== 带分数检索 ===")
    results = retriever.search_with_score(query, k=2)
    data = []
    for doc, score in results:
        print(f"[距离: {score:.4f}] {doc.page_content[:100]}...]")
        data.append({
            'query': query,
            'content': doc,
            'score': score
        })
    df = pd.DataFrame(data)
    df.to_csv("retriever.csv", index=False, encoding='utf-8')
    print(f"查询结果已保存至retriever.csv。")

    print("\n=== 使用阈值检索器 ===")
    chain_retriever = retriever.get_retriever(
        search_type="similarity_score_threshold",
        k=2,
        score_threshold=0.65
    )
    threshold_docs = chain_retriever.invoke(query)
    for doc in threshold_docs:
        print(doc.page_content[:100])