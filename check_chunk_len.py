import json

def print_oversized_chunks_from_json(
        json_path: str,
        chunk_size: int,
        preview_length: int = 100
) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("JSON 根元素应为列表，实际类型: {}".format(type(data)))

    oversized = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            content = item
        elif isinstance(item, dict):
            content = item.get("content") or item.get("page_content") or item.get("text")
            if content is None:
                print(f"⚠️ 块 {idx} 是字典但缺少 content/page_content/text 字段，跳过")
                continue
        else:
            print(f"⚠️ 块 {idx} 类型不支持 ({type(item)})，跳过")
            continue
    if not oversized:
        print(f"✅ 所有 {len(data)} 个块的长度均未超过 {chunk_size} 字符。")
        return

    print(f"⚠️ 发现 {len(oversized)} 个块长度超过 {chunk_size} 字符：\n")
    for idx, length, content in oversized:
        print(f"--- 块 {idx} ---")
        print(f"  实际长度： {length} 字符 (超过设定值 {length - chunk_size} 字符)")
        preview = content[:preview_length].replace("\n", "")