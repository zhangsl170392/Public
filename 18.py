from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os

# ---------- 配置 ----------
KNOWLEDGE_BASE_ROOT = "./knowledge_bases"
SUPPORTED_EXTENSIONS = (".txt", ".md")  # 支持的文件扩展名

# 分块参数
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# ---------- 工具函数：根据文件扩展名获取加载器 ----------
def get_loader(file_path):
    """返回合适的文档加载器"""
    if file_path.endswith(".txt"):
        return TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".md"):
        return UnstructuredMarkdownLoader(file_path, encoding="utf-8")
    else:
        return None

# ---------- 处理单个知识库目录 ---------
def process_knowledge_base(kb_path):
    """
    加载 kb_path 下的所有文档，分块后保存 chunks.json 到该目录
    """
    print(f"\n处理知识库：{kb_path}")

    # 1. 收集该目录下所有支持的文档文件
    all_docs = []
    for root, dirs, files in os.walk(kb_path):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                file_path = os.path.join(root, file)
                loader = get_loader(file_path)
                if loader is None:
                    continue
                try:
                    docs = loader.load()
                    # 为每个文档添加来源元数据（便于追溯）
                    for doc in docs:
                        doc.metadata["source"] = file_path
                    all_docs.extend(docs)
                    print(f" 加载了文件：{file_path}，共 {len(docs)} 个文档对象")
                except Exception as e:
                    print(f" 加载文件失败 {file_path}: {e}")
    if not all_docs:
        print("  ⚠️ 知识库 {kb_path} 中未找到支持的文档，跳过")
        return

    # 2. 初始化分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=SEPARATORS
    )

    # 3. 执行分割
    chunks = splitter.split_documents(all_docs)
    print(f"  知识库 {os.path.basename(kb_path)} 共生成 {len(chunks)} 个块")

    # 4. 准备输出数据
    output = []
    for idx, chunk in enumerate(chunks):
        output.append({
            "chunk_id": idx,
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "length": len(chunk.page_content)
        })

    # 5. 保存到知识库目录下
    output_path = os.path.join(kb_path, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 分块已保存至 {output_path}")

# ---------- 主程序 ----------
if __name__ == "__main__":
    # 检查根目录是否存在
    if not os.path.exists(KNOWLEDGE_BASE_ROOT):
        print(f"根目录 {KNOWLEDGE_BASE_ROOT} 不存在，请检查路径")
        exit(1)

    # 获取所有子目录（每个子目录作为一个独立知识库）
    kb_dirs = [d for d in os.listdir(KNOWLEDGE_BASE_ROOT)
               if os.path.isdir(os.path.join(KNOWLEDGE_BASE_ROOT, d))]

    if not kb_dirs:
        print(f"在 {KNOWLEDGE_BASE_ROOT} 下未找到任何知识库子目录")
        exit(1)

    print(f"发现 {len(kb_dirs)} 个知识库：{kb_dirs}")

    # 逐个处理
    for kb_name in kb_dirs:
        kb_path = os.path.join(KNOWLEDGE_BASE_ROOT, kb_name)
        process_knowledge_base(kb_path)
    print("\n所有知识库处理完成！")
'''
# 步骤1：加载文档
loader = TextLoader("knowledge_bases/technology/knowledge_base.txt", encoding="utf-8")
docs = loader.load()
print(f"加载了{len(docs)}个文档对象")

# 步骤2：初始化分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,     # 每个块的最大字符数（不是 token 数）
    chunk_overlap=50,   # 块与块之间重叠的字符数，避免关键信息被截断
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)

# 步骤3：执行分割
chunks = splitter.split_documents(docs)

print(f"文档被分为{len(chunks)}个块")
for i, chunk in enumerate(chunks[:3]):  # 打印前3个块预览
    print(f"\n---块{i+1}---")
    print(chunk.page_content[:100])
    print(f"元数据：{chunk.metadata}")

output = []
for idx, chunk in enumerate(chunks):
    output.append({
        "chunk_id": idx,
        "content": chunk.page_content,
        "metadata": chunk.metadata,
        "length": len(chunk.page_content)

    })
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("分块已保存到 chunks.json")
'''

