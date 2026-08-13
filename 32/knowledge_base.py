import os
import hashlib
import tempfile
#from typing import List

import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document

from rag_qa import PERSIST_DIR

COLLECTION_NAME = "dynamic_kb"
EMBEDDING_MODEL = "qwen3.7-text-embedding"

# 初始化嵌入模型（全局单例，避免重复加载）
@st.cache_resource
def get_embeddings():
    return DashScopeEmbeddings(model=EMBEDDING_MODEL)

# 初始化或加载向量库（全局单例）
@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    # 如果持久化目录不存在，Chroma会自动创建
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

# --- 文档处理函数 ---
def load_document(uploaded_file) -> list[Document]:
    """
    从 Streamlit 上传的文件对象加载文档。
    支持 .txt 和 .pdf 格式。
    """
    # 将上传的文件保存为临时文件，因为 LangChain 的加载器需要文件路径
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # 根据文件扩展名选择加载器[reference:6]
        if uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_file_path, encoding="utf-8")
        elif uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_path)
        else:
            raise ValueError(f"不支持的文件类型: {uploaded_file.name}")

        documents = loader.load()
        # 为每个文档块添加来源信息
        for doc in documents:
            doc.metadata['source'] = uploaded_file.name
        return documents
    finally:
        # 清理临时文件
        os.unlink(tmp_file_path)

def split_documents(documents: list[Document], chunk_size=500, chunk_overlap=50) -> list[Document]:
    """
    将文档分割成更小的块。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def get_document_hash(text: str) -> str:
    """计算文本的 MD5 哈希值，用于去重。"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def add_to_knowledge_base(uploaded_file):
    """
    处理上传的文件：加载、分割、去重，并添加到向量库。
    """
    vectorstore = get_vectorstore()
    embeddings = get_embeddings()

    # 1. 加载文档
    with st.spinner(f'正在加载文档: {uploaded_file.name}...'):
        raw_docs = load_document(uploaded_file)

    # 2. 分割文档
    with st.spinner('正在分割文档...'):
        chunks = split_documents(raw_docs)

    if not chunks:
        st.warning("文档为空或无法分割，请检查文件内容。")
        return

    # 3. 去重与添加
    new_chunks = []
    for chunk in chunks:
        # 使用内容哈希作为唯一ID
        # chunk_id = get_document_hash(chunk.page_content)
        new_chunks.append(chunk)

    if new_chunks:
        with st.spinner(f'正在将 {len(new_chunks)} 个文档块添加到知识库...'):
            vectorstore.add_documents(new_chunks)
        st.success(f"✅ 成功添加 {len(new_chunks)} 个文档块到知识库！")
    else:
        st.info("ℹ️ 所有文档块均已存在于知识库中，无需重复添加。")

