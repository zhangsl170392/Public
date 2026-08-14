import streamlit as st
import time
import json
import os
from datetime import datetime
import uuid

from rag_qa import load_retriever, rag_qa, AVAILABLE_KB

# ----------- 页面配置 ------------
st.set_page_config(
    page_title="个人知识库问答",
    page_icon="📚",
    layout="wide",
    #initial_sidebar_state="expanded",
    initial_sidebar_state="collapsed"

)

# ---------- 自定义CSS（可选，用于微调） ----------
st.markdown("""
<style>
    .chat-message-user {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 8px 12px;
        margin: 4px 0;
    }
    .chat-message-assistant {
        background-color: #e8f0fe;
        border-radius: 10px;
        padding: 8px 12px;
        margin: 4px 0;
    }
    .timestamp {
        font-size: 0.7rem;
        color: #888;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore(kb_name):
    """根据知识库名称加载对应的向量库（缓存）"""
    try:
        return load_retriever(kb_name)
    except Exception as e:
        st.error(f"加载知识库 '{kb_name}' 失败：{e}")
        return None

# ---------- 初始化会话状态 ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    # 添加一条欢迎消息（带时间戳）
    welcome_id = uuid.uuid4().hex
    st.session_state["messages"].append({
        "id": welcome_id,
        "role": "assistant",
        "content": "你好！我是基于你知识库的问答助手。请问有什么可以帮助你的？",
        "timestamp": datetime.now().isoformat()
    })

if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}

if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")

if "current_kb" not in st.session_state:
    st.session_state["current_kb"] = AVAILABLE_KB[0] if AVAILABLE_KB else "tech"

# 点赞/点踩函数
def handle_feedback(msg_id: str, value: str):
    """处理反馈点击，value 为 'like' 或 'dislike'"""
    st.session_state.feedback[msg_id] = value

def render_feedback_buttons(msg_id: str):
    """
    为指定消息 ID 渲染点赞/点踩按钮。
    会根据 st.session_state.feedback 显示当前状态。
    """
    current_feedback = st.session_state.feedback.get(msg_id)
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if current_feedback == "like":
            st.button("👍", key=f"like_{msg_id}", disabled=True,
                      help="您已点赞", type="primary")
        else:
            st.button("👍", key=f"like_{msg_id}",
                      on_click=handle_feedback, args=(msg_id, "like"),
                      help="点赞（有用）")
    with col2:
        if current_feedback == "dislike":
            st.button("👎", key=f"dislike_{msg_id}", disabled=True,
                      help="您已点踩", type="primary")
        else:
            st.button("👎", key=f"dislike_{msg_id}",
                      on_click=handle_feedback, args=(msg_id, "dislike"),
                      help="点踩（无用）")
    with col3:
        if current_feedback:
            st.caption(f"✅ 已反馈：{'👍' if current_feedback == 'like' else '👎'}")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("📖 知识库选择")
    # 下拉框
    selected_kb = st.selectbox(
        "选择知识库",
        options=AVAILABLE_KB,
        index=AVAILABLE_KB.index(st.session_state.current_kb) if st.session_state.current_kb in AVAILABLE_KB else 0,
        key="kb_selector"
    )
    # 如果选择变化，更新会话状态（并触发重绘）
    if selected_kb != st.session_state.current_kb:
        st.session_state.current_kb = selected_kb
        # 由于缓存键是 kb_name，切换后会自动加载新知识库
        st.rerun()  #强制刷新界面

    st.info("当前知识库：`knowledge_base.txt`")
    st.caption("向量数据库：ChromaDB")
    st.divider()

    # 对话统计
    st.subheader("📊 对话统计")
    total_msgs = len(st.session_state["messages"])
    user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    assistant_msgs = sum(1 for m in st.session_state.messages if m["role"] == "assistant")
    st.metric("总消息数", total_msgs)
    col1, col2 = st.columns(2)
    col1.metric("👤 用户", user_msgs)
    col2.metric("🤖 助手", assistant_msgs)
    st.divider()

    # 导出对话历史
    st.subheader("💾 导出对话")
    if st.button("📥 导出为 JSON"):
        # 创建导出目录
        os.makedirs("conversations", exist_ok=True)
        filename = f"conversations/conversation_{st.session_state['conversation_id']}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
        st.success(f"✅ 已导出到 `{filename}`")
        # 提供下载按钮
        with open(filename, "r", encoding="utf-8") as f:
            data = f.read()
        st.download_button(
            label="⬇️ 点击下载",
            data = data,
            file_name=f"conversation_{st.session_state.conversation_id}.json",
            mime="application/json"
        )

    st.divider()

    #  清空对话（带二次确认）
    st.subheader("🗑️ 管理对话")
    with st.popover("清空对话历史", type="primary", use_container_width=True):
        st.warning("⚠️ 确定要清空所有对话吗？此操作不可撤销。")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ 确认清空"):
                st.session_state.messages = []
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "对话已清空。请问有什么可以帮助你的？",
                    "timestamp": datetime.now().isoformat()
                })
                st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.rerun()
        with col_no:
            if st.button("❌ 取消"):
                st.info("已取消清空操作")
        st.divider()

        st.caption("💡 提示：回答基于检索到的文档片段生成")
        st.caption(f"会话ID：`{st.session_state.conversation_id}`")

# ---------- 主界面标题 ----------
st.title("📚 个人知识库问答系统")
st.markdown("基于你的知识库文档回答问题，并显示参考来源。")

# ---------- 显示历史消息 ----------
for idx, msg in enumerate(st.session_state.messages):

    role = msg["role"]
    avatar = "🧑‍💻" if role == "user" else "🤖"
    msg_id = msg.get("id")
    with st.chat_message(role, avatar=avatar):
        # 显示消息内容
        st.markdown(msg["content"])
        # 显示时间戳（如果有）
        if "timestamp" in msg:
            ts = datetime.fromisoformat(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            st.caption(f"🕒 {ts}")
        # 如果是助手消息，显示反馈按钮
        if role == "assistant" and msg_id:

            # 显示反馈按钮
            render_feedback_buttons(msg_id)

        # 如果消息包含来源，显示在折叠面板中
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 查看参考来源"):
                for i, src in enumerate(msg["sources"]):
                    st.caption(f"**来源 {i+1}**")
                    # 显示相似度分数（如果存在）
                    if "score" in src:
                        score = src["score"]
                        st.caption(f"📊 相似度 (L2距离，越小越相关): `{score:.4f}`")
                    content = src.get("content", "")
                    if len(content) > 300:
                        content = content[:300] + "..."
                    st.text(content)
                    if src.get("metadata"):
                        st.caption(f"元数据：{src['metadata']}")
                    st.divider()

# ---------- 用户输入 ----------
if prompt := st.chat_input("请输入你的问题..."):
    # 1. 添加用户消息到历史（带时间戳）
    user_msgs = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(user_msgs)

    # 2. 显示用户消息
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
        st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 3. 调用RAG并显示回答
    with st.chat_message("assistant", avatar="🤖"):
        # 显示加载状态
        with st.spinner("🔍 正在检索知识库并生成回答..."):
            try:
                # 获取当前知识库的向量库（缓存）
                vectorstore = get_vectorstore(st.session_state.current_kb)
                if vectorstore is None:
                    st.error("无法加载当前知识库，请检查。")
                else:
                    # 调用rag_qa，传入vectorstore
                    stream_gen, sources = rag_qa(prompt, vectorstore)
                    # 流式显示回答
                    full_response = st.write_stream(stream_gen)

                    # 显示时间戳
                    st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    # 如果有来源，显示在折叠面板中
                    if sources:
                        with st.expander("📄 查看参考来源"):
                            for i, src in enumerate(sources):
                                st.caption(f"**来源 {i + 1}**")
                                if "score" in src:
                                    score = src["score"]
                                    st.caption(f"📊 相似度 (L2距离，越小越相关): `{score:.4f}`")
                                content = src.get("content", "")
                                if len(content) > 300:
                                    content = content[:300] + "..."
                                st.text(content)
                                if src.get("metadata"):
                                    st.caption(f"元数据：{src['metadata']}")
                                st.divider()

                    # 保存助手消息到历史（含来源和时间戳）
                    welcome_id = uuid.uuid4().hex
                    assistant_msgs = {
                        "id": welcome_id,
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.messages.append(assistant_msgs)
                    # 显示反馈按钮
                    render_feedback_buttons(welcome_id)

            except Exception as e:
                st.error(f"❌ 发生错误：{str(e)}")
                # 错误信息也存入历史
                error_msg = {
                    "role": "assistant",
                    "content": f"抱歉，处理你的问题时出错了：{str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(error_msg)
                st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                '''
                # 调用RAG问答函数（现在返回分数）
                retriever = load_retriever()
                #answer, sources = rag_qa(prompt, retriever)
                stream_gen, sources = rag_qa(prompt, retriever)
                '''
                # 显示回答
                # st.markdown(answer)
