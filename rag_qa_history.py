import os
import certifi
import ssl

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

import dashscope
from http import HTTPStatus
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# ====================== 1.配置区 =====================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

# 向量库持久化目录
PERSIST_DIR = "./chroma_db"
# 检索返回最相关的文档块数量
TOP_k = 3
# 使用的模型
MODEL_NAME = "qwen-turbo"

# ====================== 2.加载向量库与检索器 =================
def load_retriever(persist_dir=PERSIST_DIR, top_k=TOP_k):
    """
    从本地加载持久化的Chroma向量库，并返回检索器对象
    """
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"persist_dir: {persist_dir} is not exist")

    # 初始化嵌入模型
    embeddings = DashScopeEmbeddings(model="text-embedding-v1")

    # 加载向量库
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # 转换为检索器
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )

    return retriever

# =================== 3.定义RAG提示词模版 ===================
RAG_PROMPT_TEMPLATE = """
你是一个基于知识库的专业助手。请严格根据以下【参考内容】回答用户的问题。

**重要规则**：
1. 如果【参考内容】中有直接相关的信息，请基于此信息作答。
2. 如果【参考内容】中没有答案，或信息不足以回答问题，请直接回复：“根据现有资料无法回答该问题。”
3. 不要编造【参考内容】中没有的信息。
"""

# ===================== 4.核心：RAG问答函数 ==================
def rag_qa(question, retriever, history=None, model=MODEL_NAME):
    """
    输入用户问题，检索相关文档，构建提示词，结合历史对话，调用大模型生成回答。
    :param question: 当前用户问题（纯文本）
    :param retriever: 检索器对象
    :param history: 之前的对话历史（列表，元素为 {"role": "user"/"assistant", "content": 纯文本}）
    :param model: 模型名称
    :return：（answer, source_documents）
    """

    if history is None:
        history = []

    # 4.1 检索相关文档块
    try:
        docs = retriever.invoke(question)
    except Exception as e:
        return f"检索失败，请检查向量库是否正常。错误信息：{e}", []

    # 如果没有检索到任何文档块
    if not docs:
        return "向量库中未找到与您问题相关的文档。", []

    # 4.2 拼接参考内容（将多个文档块合并为一个上下文）
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    # 4.3 构建最终的提示词
    #full_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    current_user_msg = {
        "role": "user",
        "content": f"【参考内容】\n{context}\n\n问题：{question}"
    }
    messages = [{"role": "system", "content": RAG_PROMPT_TEMPLATE}] + history + [current_user_msg]

    # 4.4 调用大模型
    try:
        response = dashscope.Generation.call(
            model=model,
            messages=messages
        )

        # 4.5 解析响应
        if response.status_code == HTTPStatus.OK:
            answers = response.output.text
            return answers, docs
        else:
            return f"模型调用失败，状态码：{response.status_code}，信息：{response.message}", docs
    except Exception as e:
        return f"请求异常：{str(e)}", docs

# =============== 5. 测试入口 ===============
if __name__ == "__main__":
    # 加载检索器
    print("⏳ 正在加载向量库...")
    try:
        retriever = load_retriever()
        print("✅ 向量库加载成功！")
    except Exception as e:
        print(f"❌ {e}")
        exit(1)
    # 初始化对话历史（只存储纯问题与回答，不含检索内容）
    conversation_history = []

    # 进入交互问答循环
    print("\n🤖 RAG智能问答已启动（输入 'exit' 退出）")
    while True:
        user_input = input("\n你：")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 再见！")
            break
        if not user_input.strip():
            continue

        # 调用RAG问答
        answer, source_docs = rag_qa(user_input, retriever)

        # 打印回答
        print(f"\nAI：{answer}")

        # （可选）打印引用的文档来源，方便调试和验证
        print("\n📚 参考来源：")
        for i, doc in enumerate(source_docs, 1):
            # 如果文档有元数据（如页码、文件名），可以一并打印
            source_info = doc.metadata.get("source", "未知来源")
            preview = doc.page_content[:50].replace("\n", " ") + "..."
            print(f" [{i}] {source_info} -> {preview}")

        # 将本轮纯问题和回答追加到历史（不包含检索内容）
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": answer})

