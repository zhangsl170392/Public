import dashscope
from http import HTTPStatus
import json
from datetime import datetime
import uuid     # 用于生成唯一会话ID
import time     # 用于计算响应时间
import os       # 用于文件路径管理

MAX_HISTORY = 20    # 保留最近20轮对话

# 日志配置
LOG_DIR = "conversation_logs"   # 日志文件夹名称
os.makedirs(LOG_DIR, exist_ok=True)     # 自动创建文件夹

# ========== 对话日志类 ==========
class ConversationLogger:
    """对话日志管理器"""
    def __init__(self, model_name="qwen-turbo"):
        """初始化一个新的会话"""
        self.session_id = self._generate_session_id()
        self.model_name = model_name
        self.created_at = datetime.now()
        self.messages = []
        self.round_counter = 0
        self.total_tokens = 0

    def _generate_session_id(self):
        """生成唯一的会话ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]   # 取UUID前8位
        return f"{timestamp}_{unique_id}"

    def add_user_message(self, content):
        """记录用户信息"""
        self.round_counter += 1
        message = {
            "round": self.round_counter,
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "token_count": len(content)
        }
        self.messages.append(message)
        return self.round_counter

    def add_assistant_message(self, content, response_time_ms, token_count=None):
        """记录AI回复"""
        message = {
            "round": self.round_counter,
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "respnse_time_ms": response_time_ms,
            "token_count":  token_count or len(content)
        }
        self.messages.append(message)

        # 累计token
        if token_count:
            self.total_tokens += token_count

    def save_to_file(self):
        """保存会话到JSON文件"""
        # 计算会话统计
        user_msgs = [m for m in self.messages if m['role'] == 'user']
        assistant_msgs = [m for m in self.messages if m['role'] == 'assistant']

        # 计算平均响应时间
        response_times = [m.get('respnse_time_ms', 0) for m in assistant_msgs if "response_time_ms" in m]
        avg_response_time = sum(response_times) // len(response_times) if response_times else 0

        session_data = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "model": self.model_name,
            "total_rounds": self.round_counter,
            "total_tokens": self.total_tokens,
            "messages": self.messages,
            "summary": {
                "avg_response_time_ms": avg_response_time,
                "total_tokens": self.total_tokens,
                "user_messages_count": len(user_msgs),
                "assistant_messages_count": len(assistant_msgs)
            }
        }

        # 保存文件
        filename = os.path.join(LOG_DIR, self.session_id + ".json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        return filename


def call_llm_with_metrics(messages):
    """
    调用大模型API，返回AI的回复和性能指标

    Returns:
        tuple: (reply_content, response_time_ms, token_usage)
    """
    try:
        start_time = time.time()

        response = dashscope.Generation.call(
            model = 'qwen-turbo',
            messages=messages
        )
        response_time_ms = int((time.time() - start_time) * 1000)

        if response.status_code == HTTPStatus.OK:
            # 兼容两种响应格式
            if hasattr(response.output, 'text') and response.output.text:
                reply = response.output.text
            elif response.output.choices and len(response.output.choices) > 0:
                reply = response.output.choices[0].messages['content']
            else:
                reply = "⚠️ 无法解析AI回复"

            # 提取token用量（如果有）
            token_usage = None
            if hasattr(response, 'usage'):
                token_usage = {
                    "input_tokens": response.usage.get('input_tokens', 0),
                    "output_tokens": response.usage.get('output_tokens', 0),
                    "total_tokens": response.usage.get('total_tokens', 0),
                }
            return reply, response_time_ms, token_usage
        else:
            return f"❌ API错误：{response.message}", response_time_ms, None
    except Exception as e:
        response_time_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
        return f"❌ 网络错误：{str(e)}", response_time_ms, None


def save_conversation(messages):
    """保存对话记录到JSON文件"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"conversation_{timestamp}.json"

    # 过滤掉system消息（可选），只保存user和assistant的对话
    conversation_log = []
    for msg in messages:
        if msg['role'] != 'system':
            conversation_log.append(msg)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversation_log, f, ensure_ascii=False, indent=2)

    print(f"💾 对话已保存到：{filename}")

# 添加颜色代码
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def main():
    # 初始化对话历史
    messages = [
        {'role': 'system',
         'content': "你是一个专业的测试助手，擅长软件测试、质量保障和AI应用开发。回答要专业、准确、有帮助。"}
    ]

    # 初始化日志记录器
    logger = ConversationLogger(model_name='qwen-turbo')

    print("=" * 60)
    print("🤖 智能对话机器人已启动")
    print(f"📋 会话ID: {logger.session_id}")
    print("💡 提示：")
    print("   - 输入 'exit' 退出对话")
    print("   - 输入 'clear' 清空记忆")
    print("   - 输入 'save' 保存当前对话")
    print("   - 输入 'stats' 查看当前会话统计")
    print("=" * 60)

    # 正常对话
    while True:
        user_input = input("\n👤 你：").strip()

        # 处理退出命令
        if user_input.lower() == "exit":
            print("\n👋 再见！")
            filename = logger.save_to_file()
            print(f"💾 会话已自动保存到：{filename}")
            break

        # 处理清空记忆命令
        if user_input.lower() == "clear":
            messages = [messages[0]]
            print("🧹 对话记忆已清空，重新开始对话！")
            continue
        # 保存会话内容
        if user_input.lower() == "save":
            filename = logger.save_to_file()
            print(f"💾 对话已保存到：{filename}")
            continue
        if user_input.lower() == "stats":
            print("\n📊 当前会话统计：")
            print(f"   - 对话轮数：{logger.round_counter}")
            print(f"   - 总Token数：{logger.total_tokens}")
            print(f"   - 消息数量：{len(logger.messages)}")
            continue

        # 跳过空输入
        if not user_input:
            print("⚠️ 请输入有效内容")
            continue

        # 将用户信息加入历史
        logger.add_user_message(user_input)
        messages.append({"role": 'user', 'content': user_input})

        # 显示当前轮次（方便观察）
        print("🤔 AI思考中...")
        reply, response_time_ms, token_usage = call_llm_with_metrics(messages)

        token_count = token_usage.get('total_tokens') if token_usage else None
        logger.add_assistant_message(reply, response_time_ms, token_count)
        messages.append({"role": 'assistant', 'content': reply})

        # 更新API调用的消息历史
        # messages.append({"role": "assistant", "content": reply})

        # 调用大模型
        # reply = call_llm(messages)

        # 将AI回复加入历史
        # messages.append({"role": "assistant", "content": reply})

        # 在添加新消息后检查长度
        #if len(messages) > MAX_HISTORY * 2 + 1:
            # 保留system消息和最近的MAX_HISTORY轮对话
        #    messages = [messages[0]] + messages[-(MAX_HISTORY * 2):]
        #    print(f"⚠️ 对话历史过长，已自动截断（保留最近{MAX_HISTORY}轮）")
        # 打印AI回复
        print(f"{Colors.GREEN}🤖 AI：{reply}{Colors.RESET}")
        print(f"⏱️  响应时间：{response_time_ms}ms", end="")
        if token_usage:
            print(f" | 📊 Token：{token_usage.get('total_tokens', 0)} (输入:{token_usage.get('input_tokens', 0)}, 输出:{token_usage.get('output_tokens', 0)})")
        else:
            print()     # 换行

if __name__ == "__main__":
    main()