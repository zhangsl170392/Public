from http import HTTPStatus


from dashscope import Generation
import requests
import json

from pandas.core.computation import expr


def simple_call_with_error_handling():
    try:
        response = dashscope.Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "Hello World"}],
        )

        # 检查返回的状态码
        if response.status_code == HTTPStatus.OK:
            return response.output.text
        else:
            print(f"API返回错误：{response.code} - {response.message}")
            return None
    except Exception as e:
        print(f"发生异常：{type(e).__name__} - {e}")
        print(f"详细堆栈：{traceback.format_exc()}")
        return None



def call_with_detailed_exceptions(messages):
    try:
        response = Generation.call(
            model="qwen-turbo",
            messages=messages,
            timeout=30  # 设置超时时间
        )

        #  检查HTTP状态码
        if response.status_code == HTTPStatus.OK:
            return response.output.text
        # 处理各种API错误
        elif response.status_code == HTTPStatus.BAD_REQUEST:
            print(f"❌ 请求参数错误: {response.code} - {response.message}")
            return None
        elif response.status_code == HTTPStatus.UNAUTHORIZED:
            print(f"❌ API Key无效，请检查配置")
            return None
        elif response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            print(f"⚠️ 触发限频: {response.message}")
            return "RATE_LIMIT"
        elif response.status_code >= 500:
            print(f"⚠️ 服务端错误: {response.code} - {response.message}")
            return "SERVER_ERROR"
        else:
            print(f"❌ 未知错误: {response.code} - {response.message}")
            return None

    except requests.exceptions.Timeout:
        print(f"⏰ 请求超时，服务响应太慢")
        return "TIMEOUT"
    except requests.exceptions.ConnectionError:
        print(f"🔌 网络连接错误，请检查网络")
        return "CONNECTION_ERROR"
    except Exception as e:
        print(f"❌ 未知异常: {type(e).__name__}: {e}")
        return None


"""
def call_with_retry(messages, max_retries=3, base_delay=1.0):
    
    带指数退避重试的API调用

    Args:
        messages: 消息列表
        max_retries: 最大重试次数
        base_delay: 基础等待时间（秒）

    Returns:
        AI回复内容，或None（失败时）
    

    for attempt in range(max_retries):
        try:
            print("📡 [尝试 {attempt + 1}/{max_retries + 1}] 调用API...")
            response = Generation.call(
                model="qwen-turbo",
                messages=messages,
                timeout=60
            )

            # 成功
            if response.status_code == HTTPStatus.OK:
                print(f"✅ 调用成功")
                return response.output.text
            # 判断是否需要重试
            is_retriable = False
            retry_reason = ""

            # 限频：需要等待更长时间
            if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                is_retriable = True
                retry_reason = "触发限频"
                wait_time = 60

            # 服务端错误（5xx）- 可重试
            elif response.status_code >= 500:
                is_retriable = True
                retry_reason = f"服务端错误({response.status_code})"
                wait_time = min(base_delay * (2 ** attempt), 30)    # 最大30秒

            # 其它状态码不可重试
            else:
                print(f"❌ 不可重试的错误: {response.code} - {response.message}")
                return None

            # 执行重试逻辑
            if is_retriable and attempt < max_retries:
                print(f"⚠️ {retry_reason}，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                print(f"❌ 重试次数耗尽，放弃调用")
                return None

        # 网络异常处理
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                wait_time = min(base_delay * (2 ** attempt), 30)
                print(f"⏰ 请求超时，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"❌ 重试次数耗尽，请求超时")
                return None

        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                wait_time = min(base_delay * (2 ** attempt), 30)
                print(f"🔌 连接错误，等待 {wait_time} 秒后重试...")
                return None
        except Exception as e:
            print(f"❌ 未预期的异常: {type(e).__name__}: {e}")
            return None
    return None

# 测试
messages = ["你好！"]
result = call_with_detailed_exceptions(messages)
if result:
    print(f"AI回复：{result}")
"""

import dashscope
from http import HTTPStatus
import requests
import time
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 重试配置
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1    # 秒
RATE_LIMIT_WAIT = 60    # 限频等待时间（秒）
MAX_RETRY_DELAY = 30    # 最大重试延迟（秒）

# 超时配置
REQUEST_TIMEOUT = 60    # 秒

def call_with_retry(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    带智能重试的API调用
    支持：超时重试、网络错误重试、限频处理、服务端错误重试
    :param messages:
    :return:
    """

    for attempt in range(MAX_RETRIES + 1):
        try:
            print(f"  📡 [尝试 {attempt + 1}/{MAX_RETRIES + 1}]...", end=" ", flush=True)

            response = dashscope.Generation.call(
                model="qwen-turbo",
                messages=messages,
                timeout=REQUEST_TIMEOUT,
                result_format="message"
            )

            # 成功
            if response.status_code == HTTPStatus.OK:
                print("✅")

                return response.output.choices[0].message.content

            # ========== 处理限频 ==========
            if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                print(f"⚠️ 限频")
                if attempt < MAX_RETRIES:
                    print(f"     等待 {RATE_LIMIT_WAIT} 秒后重试...")
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                else:
                    print(f"  ❌ 重试次数耗尽")
                    return None

            # ========== 处理服务端错误 ==========
            if response.status_code >= 500:
                print(f"⚠️ 服务端错误({response.status_code})")
                if attempt < MAX_RETRIES:
                    delay = min(BASE_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    print(f"     等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"  ❌ 重试次数耗尽")
                    return None

            # ========== 处理参数/认证错误（不可重试） ==========
            if response.status_code == HTTPStatus.BAD_REQUEST:
                print(f"❌ 请求错误: {response.message}")
                return None
            if response.status_code == HTTPStatus.UNAUTHORIZED:
                print(f"❌ API Key无效，请检查配置")
                return None

            # 其他错误
            print(f"❌ {response.code}: {response.message}")
            return None

        # ========== 网络层异常处理 ==========
        except requests.exceptions.Timeout:
            print(f"⏰ 超时")
            if attempt < MAX_RETRIES:
                delay = min(BASE_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                print(f"     等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
            else:
                print(f"  ❌ 重试次数耗尽")
                return None
        except requests.exceptions.ConnectionError as e:
            print(f"🔌 连接错误")
            if attempt < MAX_RETRIES:
                delay = min(BASE_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                print(f"     等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
            else:
                print(f"  ❌ 重试次数耗尽: {e}")
                return None
        except Exception as e:
            print(f"💥 未知异常: {type(e).__name__}")
            return None
    return None

def save_conversation_log(messages: List[Dict], filename: str = "conversation_log.json"):
    """保存对话日志"""
    log_entry = {
        "messages": messages,
        "timestamp": datetime.now().isoformat(),
    }

    # 读取已有日志
    existing_logs = []
    if os.path.exists(filename):
        with open(filename, "r", encoding='utf-8') as f:
            existing_logs = json.load(f)

    # 追加新记录
    existing_logs.append(log_entry)

    # 写回文件
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(existing_logs, f, ensure_ascii=False, indent=2)

    print(f"📁 对话已保存到 {filename}")

def run_robust_chat():
    """运行健壮的多轮对话"""

    print("=" * 50)
    print("🤖 健壮多轮对话系统 v2.0")
    print("=" * 50)
    print("命令: quit(退出) | save(保存对话) | clear(清空历史)")
    print("-" * 50)

    # 初始化对话历史
    messages = [
        {"role": "system", "content": "你是一个友好、乐于助人的助手。"}
    ]

    while True:
        try:
            user_input = input("\n👤 你: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print(f"👋 再见！")
                break

            if user_input.lower() == 'save':
                save_conversation_log(messages)
                continue

            if user_input.lower() == 'clear':
                messages = [{"role": "system", "content": "你是一个友好、乐于助人的助手。"}]
                continue

            # 添加用户信息
            messages.append({"role": "user", "content": user_input})

            # 调用API（带重试）
            print("🤖 AI: ", end="", flush=True)
            reply = call_with_retry(messages)

            if reply:
                print(f"{reply}")
                messages.append({"role": "assistant", "content": reply})
            else:
                print("\n[系统] 抱歉，我遇到了一些问题，请稍后再试。")
                # 移除刚才添加的用户信息，避免污染历史
                messages.pop()

        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except EOFError:
            print("\n👋 再见！")
            break

if __name__ == "__main__":
    # 检查API Key
    run_robust_chat()