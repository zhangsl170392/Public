"""
import dashscope
from http import HTTPStatus
import time

def test_model(model_name, prompt):

from http import HTTPStatus

测试制定模型在给定prompt下的表现
    print(f"\n{'='*50}")
    print(f"测试模型：{model_name}")
    print(f"用户问题：{prompt}")
    print(f"{'-'*50}")

    start_time = time.time()

    messages = [
        {"role": "system", "content": "你是一个严谨的AI助手，请提供准确、清晰的回答。"},
        {"role": "user", "content": prompt}
    ]

    try:
        response = dashscope.Generation.call(
            model=model_name,
            messages=messages,
        )

        elapsed_time = time.time() - start_time

        if response.status_code == HTTPStatus.OK:
            reply = response.output.text
            # 计算回复长度（中文字符数）
            reply_length = len(reply)
            print(f"AI回复（前200字符）：\n{reply[:200]}...")
            print(f"\n📊 耗时：{elapsed_time:.2f} 秒，回复长度：{reply_length} 字符")
        else:
            print(f"调用失败：{response.status_code}")

    except Exception as e:
        print(f"出错：{e}")

# 准备一组从简单到复杂的测试问题
test_questions = [
    "1+1等于几？",  # 简单问答
    "请用一句话介绍你自己。",  # 基础自我介绍
    "请解释什么是递归，并给出一个简单的Python例子。"  # 复杂任务
]

# 需要对比的模型
models_to_test = ["qwen-turbo", "qwen-plus", "qwen-max"]

for question in test_questions:
    for model in models_to_test:
        test_model(model, question)
    print(f"\n{'='*50}\n")


import dashscope
from http import HTTPStatus

response = dashscope.Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "鸡兔同笼，共有35个头，94只脚，问鸡和兔各有多少只？"}],
    extra_body={
        "enable_thinking": True
    }
)

if response.status_code == HTTPStatus.OK:
    print(response)

    # 思考过程
    if hasattr(response.output, 'text'):
        print("🤔 思考过程：\n", response.output.text)
    # 最终回答
    print("\n✅ 最终回答：\n", response.output.text)

import dashscope
from http import HTTPStatus

def chat_streaming():

from http import HTTPStatus

流式对话示例
    prompt = "请写一首关于夏天的短诗，每行换行输出"
    print(f"用户：{prompt}")
    print("AI: ", end="", flush=True)   # flush=True 确保立即显示

    # 设置 stream=True 开启流式输出
    response_generator = dashscope.Generation.call(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        result_format="message" # 使用 message 格式
    )

    for response in response_generator:

        if response.status_code == HTTPStatus.OK:
            # 流式模式下，增量内容在output.choices[0].delta.content中
            delta = response.output.choices[0].message.content
            print(delta, end="", flush=True)
        else:
            print(f"\n错误：{response.code} - {response.message}")
            break

    print("\n")

if __name__ == "__main__":
    chat_streaming()

import dashscope
import time
from http import HTTPStatus

prompt = "请简略介绍Python编程语言的特点和优势。"

# 同步模式
print("【同步模式】")
start = time.time()
response = dashscope.Generation.call(
    model="qwen-turbo",
    messages=[{"role": "user", "content": prompt}],
)
end = time.time()
print(f"完整回复：\n{response.output.text[:200]}...\n")

import dashscope
import time
from http import HTTPStatus

prompt = "请简略介绍Python编程语言的特点和优势。"
# 流式模式
print("【流式模式】")
start = time.time()
response_gen = dashscope.Generation.call(
    model="qwen-turbo",
    messages=[{"role": "user", "content": prompt}],
    stream=True,
    result_format="message"
)
print("AI: ", end="", flush=True)
last_content = ""
for response in response_gen:
    if response.status_code == HTTPStatus.OK:
        content = response.output.choices[0].message.content
        new_part = content[len(last_content):]
        print(new_part, end="", flush=True)
        last_content = content

end = time.time()
print(f"\n总耗时：{end - start:.2f}秒，内容逐字呈现")

import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

completion = client.chat.completions.create(
    model="qwen-turbo",
    messages=[
        {"role": "system", "content": "你是一个乐于助人的AI助手。"},
        {"role": "user", "content": "请用三句话介绍一下你自己"}
    ],
    temperature=0.7,
    max_tokens=500
)

# 输出回复
print(completion.choices[0].message.content)
"""

import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 初始化对话历史
messages = [
    {"role": "system", "content": "你是一个测试助手，回答简洁友好。"}
]

def save_conversation(messages_list):
    """保存对话记录到JSON文件"""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation": messages_list,

    }
    with open("conversation_log_compatible.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

print("开始对话！输入'exit'退出")

while True:
    user_input = input("\n你： ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    try:
        completion = client.chat.completions.create(
            model="qwen-turbo",
            messages=messages,
        )

        reply = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        print(f"AI: {reply}")
        save_conversation(messages)

    except Exception as e:
        print(f"调用失败： {e}")
        continue

print(f"\n对话结束。共进行了{len(messages)//2} 轮对话。")