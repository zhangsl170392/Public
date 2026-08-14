'''
from langchain_community.document_loaders import TextLoader

# 加载本地文本文件
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

print(f"加载到 {len(documents)}")
print(documents[0].page_content[:150])
print("元数据：", documents[0].metadata)
'''
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)

chunks = splitter.split_documents(documents)

print(f"原始文档被切分为 {len(chunks)}个块")
for i, chunk in enumerate(chunks, 1):
    print(f"\n--- 块 {i} ---")
    print(chunk.page_content)
    print(f"元数据：{chunk.metadata}")