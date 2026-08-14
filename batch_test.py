import dashscope
import json
import csv
import time
import os
from http import HTTPStatus
from datetime import datetime
import pandas as pd

# ================= 配置区域 =================
# 强烈建议从环境变量读取 API Key，不要硬编码
# 在终端设置：export DASHSCOPE_API_KEY="sk-xxx"
# 或者在代码中直接赋值（仅用于测试）
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
MODEL_NAME = "qwen-turbo"
REQUEST_DELAY = 1
CSV_FILENAME = "api_test_results.csv"
MAX_RETRIES = 3

# ===========================================

def call_qwen_with_metrics(prompt):
    """
    调用通义千问 API，返回 (回答内容, 耗时秒数, 状态码/异常信息)
    """
    start_time = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            response = dashscope.Generation.call(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - start_time
            if response.status_code == HTTPStatus.OK:
                reply = response.output.text
                status = "SUCCESS"
                return reply, round(elapsed, 2), status
            elif response.status_code == 429:   # 触发限频
                wait_time = attempt + 1
                print(f"触发限流，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                reply = f"API错误，状态码：{response.status_code}"
                status = f"ERROR_{response.status_code}"
                return None
            # return reply, round(elapsed, 2), status
        except Exception as e:
            print(f"第{attempt+1}次尝试失败：{e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                elapsed = time.time() - start_time
                return f"异常：{str(e)}", round(elapsed, 2), "EXCEPTION"
    return None

def main():
    # 1. 读取 test_prompts.json
    try:
        with open("test_prompt2.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)
        prompts_list = test_data["test_prompts"]
    except FileNotFoundError:
        print("❌ 未找到 test_prompts.json，请先完成第8天的任务。")
        return

    # 2. 准备CSV文件头
    results = []
    print(f"🚀 开始批量测试，模型：{MODEL_NAME}，共 {len(prompts_list)} 个用例\n")

    for idx, case in enumerate(prompts_list, start=1):
        name = case["name"]
        prompt = case["prompt"]
        print(f"[{idx}/{len(prompts_list)}] 正在测试: {name} ...")

        # 调用API并获取结果
        reply, elapsed, status = call_qwen_with_metrics(prompt)

        # 记录结果（用于写入CSV）
        results.append({
            "测试名称": name,
            "Prompt": prompt,
            "模型回答": reply,
            "耗时（秒）": elapsed,
            "状态": status,
            "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        # 控制台简单打印回答的前100个字符
        preview = reply.replace("\n", " ")[:100] + ("..." if len(reply) > 100 else "")
        print(f"   ✅ 完成，耗时 {elapsed}s，回答预览：{preview}\n")

        # 避免请求过快
        time.sleep(REQUEST_DELAY)
    """
    3. 写入CSV文件
    with open(CSV_FILENAME, "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["测试名称", "Prompt", "模型回答", "耗时（秒）", "状态", "时间戳"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    """
    # 4. 写入xlsx文件
    fieldnames = ["测试名称", "Prompt", "模型回答", "耗时（秒）", "状态", "时间戳"]
    df = pd.DataFrame(results, columns=fieldnames)
    df.to_excel("api_test_results.xlsx", index=False)

    print(f"✅ 批量测试完成！结果已保存至api_test_results.xlsx")

if __name__ == "__main__":
    main()

