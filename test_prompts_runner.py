import dashscope
import json
import os
from http import HTTPStatus

import requests

# 1. 设置你的API密钥
dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

def call_qwen(prompt):
    """调用通义千问API并返回回答"""
    try:
        response = dashscope.Generation.call(
            model="qwen-max",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.text
        else:
            return f"API调用失败，错误码：{response.status_code}"
    except Exception as e:
        return f"发生异常：{str(e)}"

# 2. 读取测试prompt集
with open("test_prompt.json", "r", encoding='utf-8') as f:
    test_data = json.load(f)

# 3. 执行测试并打印结果
print("=== Prompt测试结果 ===\n")
for test_case in test_data["test_prompts"]:
    print(f"【{test_case['name']}】")
    print(f"测试Prompt: {test_case['prompt']}")
    print("--------- AI回答 ---------")
    response = call_qwen(test_case["prompt"])
    print(f"{response}")
    print("-" * 50 + "\n")
