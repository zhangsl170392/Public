import streamlit as st
import os

from knowledge_base import get_vectorstore, add_to_knowledge_base

# 调试：先检查 st.secrets 是否存在该键
try:
    api_key = st.secrets["DASHSCOPE_API_KEY"]
    st.write(f"🔑 从 secrets 读取到的 Key 前缀：{api_key[:5]}...，长度：{len(api_key)}")
    # 去除可能的空白字符
    api_key = api_key.strip()
    os.environ["DASHSCOPE_API_KEY"] = api_key
    st.success("✅ 环境变量设置成功")
except KeyError as e:
    st.error(f"❌ 未在 secrets 中找到键 'DASHSCOPE_API_KEY'，请检查大小写。错误：{e}")
except Exception as e:
    st.error(f"❌ 读取 secrets 时出错：{e}")

# --- 页面配置 ---
st.set_page_config(page_title="📚 动态知识库问答", layout="wide")
st.title("📚 动态知识库问答系统")

# --- 初始化 ---
# 初始化向量库（确保在应用启动时加载或创建）
if 'vectorstore_ready' not in st.session_state:
    with st.spinner('正在初始化知识库...'):
        get_vectorstore()
    st.session_state.vectorstore_ready = True

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 侧边栏：文件上传 ---
with st.sidebar:
    st.header("📤 上传文档")
    uploaded_file = st.file_uploader(
        "选择要上传的文档 (.txt 或 .pdf)",
        type=["txt", "pdf"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        # 当有文件上传时，调用添加逻辑
        # 使用一个按钮来触发添加，避免每次重新运行都处理
        if st.button("添加到知识库", type="primary"):
            add_to_knowledge_base(uploaded_file)
            # 上传成功后，清除文件上传器的状态，以便上传下一个文件
            # 注意：这需要一些技巧，简单的做法是使用 st.experimental_rerun()
            # 或者重新设置 session_state 中的 key。
            # 这里我们只是提示用户，并让文件上传器保持当前文件。
            # 用户可以再次上传新文件。
    st.divider()
    st.caption("💡 提示：上传的文档将被分割并向量化，然后用于增强问答。")

# --- 主区域：聊天界面 ---
# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入你的问题"):
    # 将用户问题添加到历史并显示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用RAG问答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                vectorstore = get_vectorstore()
                # 执行检索
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(prompt)

                # 构建上下文
                context = "\n\n".join([doc.page_content for doc in docs])

                # 调用大模型 (这里省略了具体的LLM调用代码，你可以集成之前的rag_qa函数)
                # 为了演示，我们简单地返回检索到的文档片段
                # 在实际应用中，你应该调用你的LLM生成回答
                response = f"我检索到了以下相关信息（共{len(docs)}个片段）：\n\n"
                for i, doc in enumerate(docs):
                    response += f"**来源 {i+1}**: {doc.metadata.get('source', '未知')}\n"
                    response += f"{doc.page_content[:200]}...\n\n"

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"发生错误：{e}")
