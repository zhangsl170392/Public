import dashscope
import json
import time
import os
from datetime import datetime
from http import HTTPStatus
from typing import List, Dict, Optional


class QwenChatBot:
    """通义千问对话机器人封装类"""
    def __init__(self, api_key: str, model: str = "qwen-plus") -> None:
        self.api_key = api_key
        self.model = model
        self.messages = []
        self.max_retries = 3
        self.retry_delay = 1

    def set_system_prompt(self, system_content: str):
        """设置系统提示词"""
        self.messages = [{"role": "system", "content": system_content}]

    def _call_with_retry(self, user_message: str) -> Optional[str]:
        """带重试机制的API调用"""
        self.messages.append({"role": "user", "content": user_message})

        for attempt in range(self.max_retries):
            try:
                response = dashscope.Generation.call(
                    model=self.model,
                    messages=self.messages,
                    timeout=30
                )

                if response.status_code == HTTPStatus.OK:
                    reply = response.output.text
                    self.messages.append({"role": "assistiant", "content": reply})
                    return reply
                elif response.status_code == 429:   # 限频错误
                    wait_time = self.retry_delay * (attempt + 1)
                    print(f"触发限流， {wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print(f"调用失败，状态码：{response.status_code}")
                    return None
            except Exception as e:
                print(f"第{attempt + 1}次尝试失败：{e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None
        return None

    def chat(self) -> None:
        """启动交互式对话"""
        print(f"开始对话！当前模型：{self.model}")
        print("输入'exit'退出，输入'save'保存对话，输入'model'切换模型")

        while True:
            user_input = input("\n你：")
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "save":
                self.save_conversation()
                print("对话已保存")
                continue
            elif user_input.lower() == "model":
                print(f"当前模型：{self.model}")
                continue

            reply = self._call_with_retry(user_input)
            if reply:
                print(f"AI: {reply}")

    def save_conversation(self, filename: str = None) :
        """保存对话记录"""
        if filename is None:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "conversation": self.messages
        }
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"对话已保存到{filename}")

# 使用示例
if __name__ == "__main__":
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    bot = QwenChatBot(api_key=api_key[0], model="qwen-plus")
    bot.set_system_prompt("你是一个友好的AI助手，回答简洁准确。")
    bot.chat()

