import os
import ssl
import certifi

import fix_ssl

import json
import os
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
import shutil


DASH_SCOPE_API_KEY = os.environ.get("DASH_SCOPE_API_KEY")
EMBEDDING_MODEL = "qwen3.7-text-embedding"

# 基础目录，所有知识库的持久化子目录将放在这里
BASE_PERSIST_DIR = "./chroma_dbs"
os.makedirs(BASE_PERSIST_DIR, exist_ok=True)

# ---------- 知识库管理器 ----------
class KnowledgeBase:
    """
    管理一个独立的知识库，每个知识库有独立的子目录，持久化在 BASE_PERSIST_DIR/name 下
    """
    def __init__(self, name: str, embedding=None):
        self.name = name
        self.persist_dir = os.path.join(BASE_PERSIST_DIR, name)
        os.makedirs(self.persist_dir, exist_ok=True)

        # 共用或单独指定embedding，不同知识库可以共用同一个embedding
        if embedding is None:
            self.embedding = DashScopeEmbeddings(
                model=EMBEDDING_MODEL,
                dashscope_api_key=DASH_SCOPE_API_KEY
            )
        else:
            self.embedding = embedding
        self.vectorstore = None # 延迟加载

    def create_from_documents(self, documents: list[Document], batch_size=20):
        """用文档列表创建（覆盖已有的同名知识库）"""
        # 1.如果想“覆盖重建”，可以先把目录删掉（可选）
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)

        # 2.创建一个空向量库（仅初始化集合，不添加文档）
        self.vectorstore = Chroma(
            # documents=documents,
            embedding_function=self.embedding,
            persist_directory=self.persist_dir
        )

        # 3.分批添加文档
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i: i+batch_size]
            print(f"📦 添加第 {i//batch_size + 1} 批（{len(batch)} 个块）...")
            self.vectorstore.add_documents(batch)

        # 4.持久化到磁盘
        self.vectorstore.persist()
        print(f"✅ 知识库 '{self.name}' 创建完成，保存于 {self.persist_dir}")

    def load(self):
        """加载已存在的知识库（用于后续查询或追加）"""
        if self.vectorstore is None:
            # 检查目录是否存在且非空（至少有一个文件）
            if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embedding
                )
                print(f"✅ 知识库 '{self.name}' 已加载")
            else:
                print(f"⚠️ 知识库 '{self.name}' 目录为空，请先创建")
        return self.vectorstore is not None

    def add_documents(self, documents: list[Document]):
        """追加文档到现有知识库"""
        if self.vectorstore is None:
            self.load()
        if self.vectorstore is not None:
            self.vectorstore.add_documents(documents)
            print(f"📝 向知识库 '{self.name}' 添加了 {len(documents)} 个文档")
        else:
            raise RuntimeError(f"知识库 '{self.name}' 未加载或不存在")

    def search(self, query: str, k: int = 3):
        """相似性搜索并打印结果"""
        if self.vectorstore is None:
            self.load()
        if self.vectorstore is None:
            print(f"❌ 知识库 '{self.name}' 不可用")
            return []
        print(f"\n🔍 查询 [{self.name}]：{query}")
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        for i, (doc, score) in enumerate(results, 1):
            print(f"  结果 {i}:")
            print(f"    chunk_id: {doc.metadata.get('chunk_id', 'N/A')}")
            print(f"    内容预览: {doc.page_content[:150]}...")
            print(f"    相似度分数: {score:.4f}")
            print(f"    元数据: {doc.metadata}\n")
        return results

# ------------------- 辅助函数 -------------------
def load_documents_from_json(json_path: str) -> list[Document]:
    """从 JSON 分块文件加载 Document 列表"""
    with open(json_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    documents = []
    for item in chunks_data:
        content = item.get("content", "")
        metadata = item.get("metadata", {})
        chunk_id = item.get("chunk_id", "")
        metadata['chunk_id'] = chunk_id
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    print(f"从 {json_path} 加载了 {len(documents)} 个文档块")
    return documents

# ===================== 主程序 =====================
if __name__ == '__main__':
    # ---------- 配置：列出所有需要创建知识库的 JSON 文件 ----------
    # 每个 JSON 对应一个知识库，知识库名称 = 文件名（不含扩展名）
    json_files = [
        "knowledge_bases/sport/chunks.json", # 技术文档
        "knowledge_bases/technology/chunks.json",  # 体育文档
    ]

    # ---------- 依次创建每个知识库 ----------
    kb_instances = {}
    for json_file in json_files:
        # 从文件名提取知识库名称（去掉 .json）
        kb_name = os.path.basename(os.path.dirname(json_file))
        # 加载文档
        docs = load_documents_from_json(json_file)
        # 创建知识库实例
        kb = KnowledgeBase(kb_name)
        kb.create_from_documents(docs)
        kb_instances[kb_name] = kb


'''
# 1. 加载之前保存的分块 JSON 文件
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks_data = json.load(f)

# 2. 转换为 LangChain Document 对象列表
documents = []
for item in chunks_data:
    # chunks.json 中每个元素应包含 "content" 和 "metadata"
    content = item.get("content", "")
    metadata = item.get("metadata", {})
    chunk_id = item.get("chunk_id", "")
    metadata['chunk_id'] = chunk_id
    doc = Document(page_content=content, metadata=metadata)

    documents.append(doc)

print(f"加载了 {len(documents)} 个文档块")

# 3. 初始化 DashScope Embeddings
#    确保环境变量 DASHSCOPE_API_KEY 已设置，或者直接传入 api_key
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=DASH_SCOPE_API_KEY
)

# 4. 创建 Chroma 向量数据库并持久化
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 5. 显式持久化（Chroma 4.x 之后可能自动持久化，但调用 persist 更安全)
#vectorstore.persist()
print("✅ 向量数据库已创建，保存在 ./chroma_db 目录")

query = "什么是限流策略"
results = vectorstore.similarity_search(query, k=1)

print(f"查询：{query}\n")
for i, doc in enumerate(results):
    print(f"结果 {i+1}: ")
    print(doc.page_content[:200])
    print(f"元数据：{doc.metadata}\n")

def search_knowledge(query, k=3):
    print(f"查询： {query}\n")
    results = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results):
        print(f"结果：{i+1}: ")
        original_id = doc.metadata.get("chunk_id", "未保存ID")
        print(f"id: {original_id}")
        print(doc.page_content[:200])
        print(f"元数据：{doc.metadata}\n")
        print(f"相似度份数：{score}")

# search_knowledge(query)
'''