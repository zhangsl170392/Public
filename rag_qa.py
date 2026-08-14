import fix_ssl

import os
import dashscope
from dashscope import MultiModalConversation
from http import HTTPStatus
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# ====================== 1.配置区 =====================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
dashscope.api_key = DASHSCOPE_API_KEY

# 向量库持久化目录
PERSIST_DIR = "./chroma_dbs"

# 可用知识库列表
AVAILABLE_KB = ["technology", "sport"]

# 检索返回最相关的文档块数量
TOP_k = 3
# 使用的模型
MODEL_NAME = "qwen3.7-flash"

# ====================== 2.加载向量库与检索器 =================
def load_retriever(kb_name="tech", top_k=TOP_k):
    """
    从本地加载持久化的Chroma向量库，并返回检索器对象
    """
    persist_dir = os.path.join(PERSIST_DIR, kb_name)
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"知识库 '{kb_name}' 不存在于路径：{persist_dir}")

    # 初始化嵌入模型
    embeddings = DashScopeEmbeddings(model="qwen3.7-text-embedding")

    # 加载向量库
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # 转换为检索器
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )

    #return retriever
    # 配合返回相似度分数
    return vectorstore

# =================== 3.定义RAG提示词模版 ===================
RAG_PROMPT_TEMPLATE = """
你是一个基于知识库的专业助手。请严格根据以下【参考内容】回答用户的问题。

**重要规则**：
1. 如果【参考内容】中有直接相关的信息，请基于此信息作答。
2. 如果【参考内容】中没有答案，或信息不足以回答问题，请直接回复：“根据现有资料无法回答该问题。”
3. 不要编造【参考内容】中没有的信息。
4. 请使用Markdown格式组织回答，例如使用 **加粗** 强调关键点，使用 - 或 1. 列出要点，使用 ## 划分章节，使内容清晰易读。

【参考内容】
{context}

问题：{question}
回答：
"""

# ===================== 4.核心：RAG问答函数 ==================
def rag_qa(question, retriever, model=MODEL_NAME):
    """
    输入用户问题，检索相关文档，构建提示词，调用大模型生成回答。
    返回：（answer, source_documents）
    """
    # 4.1 检索相关文档块
    try:
        #docs = retriever.invoke(question)
        docs = retriever.similarity_search_with_score(question, k=TOP_k)
    except Exception as e:
        # return f"检索失败，请检查向量库是否正常。错误信息：{e}", []
        # 流式输出
        error_msg = f"检索失败，请检查向量库是否正常。错误信息：{e}"
        def error_gen():
            yield error_msg
        return error_gen(), []

    # 如果没有检索到任何文档块
    if not docs:
        # return "向量库中未找到与您问题相关的文档。", []
        # 流式输出
        def no_doc_gen():
            yield "向量库中未找到与您问题相关的文档。"
        return no_doc_gen(), []

    # 4.2 拼接参考内容（将多个文档块合并为一个上下文）
    context = "\n\n---\n\n".join([doc.page_content for doc, score in docs])

    # 4.3 构建最终的提示词
    full_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # 将文档块封装成字典列表
    source_dict = [
        {
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
        }
        for doc, score in docs
    ]

    # 4.4 调用大模型
    try:
        response = dashscope.MultiModalConversation.call(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            stream=True
        )
        '''
        # 4.5 解析响应
        if response.status_code == HTTPStatus.OK:
                answers = response.output.text
                return answers, source_dict
            else:
                return f"模型调用失败，状态码：{response.status_code}，信息：{response.message}", source_dict
        '''
        # 4.5 解析响应——流式
        def text_generator():
            previous = ""
            for res in response:

                if res.status_code == HTTPStatus.OK:
                    choices = res.output.choices
                    if choices and len(choices) > 0:
                        content = choices[0].message.content
                        if content and isinstance(content, list):
                            # 提取所有text字段并拼接
                            text_parts = [item.get('text', '') for item in content if 'text' in item]
                            full_text = ''.join(text_parts)
                            if full_text:
                                yield full_text
                        elif isinstance(content, str):
                            # 兼容旧版本或特殊情况
                            yield content
                else:
                    yield f"模型调用失败，状态码：{res.status_code}，信息：{res.message}"
                    break

        return text_generator(), source_dict

    except Exception as e:
        error_msg = f"请求异常：{str(e)}"
        def error_gen():
            yield error_msg
        return error_gen(), source_dict

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

