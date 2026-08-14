import dashscope
import csv
import time
import os
from http import HTTPStatus

dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

def call_llm(prompt, temperature, top_p, model='qwen-turbo'):
    try:
        response = dashscope.Generation.call(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            top_p=top_p,
            result_format='message',
            seed=42
        )
        if response.status_code == HTTPStatus.OK:

            return response.output.choices[0].message.content
        else:
            return f"Error: {response.code} - {response.message}"
    except Exception as e:
        return f"Error: {e}"

prompts = {
    "factual": "请用一句话介绍杭州，必须包含'浙江省会'、'西湖'。",
    "creative": "请为一家新开的猫咪咖啡馆写一句有吸引力的广告语，风格活泼。"
}

params = [
    {"temperature": 0.0, "top_p": 0.8},
    #{"temperature": 0.5, "top_p": 0.9},
    #{"temperature": 1.0, "top_p": 0.95},
    {"temperature": 1.2, "top_p": 0.99},
]

results = []
repeat_times = 3

for prompt_key, prompt_text in prompts.items():
    for param in params:
        for i in range(repeat_times):
            resp = call_llm(prompt_text, param['temperature'], param['top_p'])
            results.append(
                {
                    "prompt_type": prompt_key,
                    "prompt": prompt_text,
                    "temperature": param['temperature'],
                    "top_p": param['top_p'],
                    "run": i+1,
                    "response": resp,
                    "length": len(resp),
                    "timestamp": time.time()
                }
            )
            time.sleep(1)

# 保存为CSV
with open('param_compare_result.csv', 'w', newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("实验完成，结果已保存至param_compare_results.csv")
