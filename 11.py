
all_docs = vectorstore.get()  # 获取所有存储的文档
for i, doc in enumerate(all_docs['documents'][:5]):   # 打印前5个文档内容
    print(f"Doc {i}: {doc[:100]}...")
    print(f"   元数据: {all_docs['metadatas'][i]}\n")