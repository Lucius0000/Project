# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 19:05:06 2025

@author: Lucius
"""


import os
import pandas as pd
import json
import re
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def init_client(api_key):
    return OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )

def generate_equity_incentive_prompt(text):
    return f"""
以下是某家上市公司报告的股权激励计划内容：
'''{text}'''

**任务说明**
请从上述文本中提取以下结构化信息，并以 JSON 格式返回：

返回格式：如未明确提及则用 null 填充（不要省略字段）：

{{
  "激励工具类型": "限制性股票/股票期权/null",
  "授予数量（万股）": 数字或 null,
  "授予比例（%）": 数字或 null,
  "授予价格（元）": 数字或 null,
  "是否首次授予": "是/否/null",
  "是否包含预留": "是/否/null",
  "激励对象": ["董事", "高级管理人员", "技术高线"] 或 [],
  "有效期（月）": 数字或 null,
  "是否已获股东大会通过": "是/否/null"
}}

**特别说明**
- “激励工具类型” 仅选中“限制性股票” 或 “股票期权”；
- “授予数量”单位为万股；“有效期”单位统一为月；
- “激励对象” 提取文本中常见的角色关键词；
- 所有值需是实际数据或 null，禁止出现计算式，不需要给出计算过程；
- 仅返回 JSON 结果，不要加前缀或说明文字。
- 所有字段必须为最基础类型（字符串、数字或 null），禁止出现嵌套结构。
"""

# 调用 API 获取结构化结果
def call_doubao_api(client, prompt, model="doubao-1-5-lite-32k-250115"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是股权激励信息结构化提取专家"},
            {"role": "user", "content": prompt}
        ],
        stream=False,
        temperature=0
    )
    return response.choices[0].message.content

# 抽取有效 JSON 格式

def extract_json_from_text(text):
    try:
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            return json.loads(match.group())
        else:
            raise ValueError("未找到合法 JSON 对象")
    except Exception as e:
        raise ValueError(f"JSON 解析失败：{e}")

# 单条处理逻辑
def process_row(i, row, client, model):
    raw_text = str(row['方案说明_ProjSta'])
    prompt = generate_equity_incentive_prompt(raw_text)
    try:
        reply = call_doubao_api(client, prompt, model=model)
        raw_result = extract_json_from_text(reply)

        # 使用原始值优先，若为空再使用模型值
        def get_final_value(field_model, field_raw, unit_scale=1.0):
            val = row.get(field_raw, None)
            if pd.notna(val):
                try:
                    return float(val) * unit_scale
                except:
                    return None
            return raw_result.get(field_model, None)

        result_dict = {
            "激励工具类型": raw_result.get("激励工具类型", None),
            "授予数量（万股）": get_final_value("授予数量（万股）", "激励计划授予数量(股)_IncPlanAwardVol", unit_scale=1/10000),
            "授予比例（%）": get_final_value("授予比例（%）", "激励计划授予数量占总股本比例(%)_PCTTotShr"),
            "授予价格（元）": get_final_value("授予价格（元）", "激励计划授予价格(元)_AwardPrice"),
            "是否首次授予": raw_result.get("是否首次授予", None),
            "是否包含预留": raw_result.get("是否包含预留", None),
            "激励对象": raw_result.get("激励对象", None),
            "有效期（月）": raw_result.get("有效期（月）", None),
            "是否已获股东大会通过": raw_result.get("是否已获股东大会通过", None),
            "股票代码_StkCd": row['股票代码_StkCd'],
            "首次授予确定公告日_FistDeterDt": row['首次授予确定公告日_FistDeterDt']
        }
        return result_dict, None
    except Exception as e:
        reply_text = reply if 'reply' in locals() else '无返回内容'
        return {
            "index": i,
            "raw_text": raw_text,
            "reply": reply_text,
            "error": str(e)
        }, e


def process_equity_incentive_file(input_filename, output_filename, api_key, model, max_workers=5):
    client = init_client(api_key)
    df = pd.read_excel(input_filename)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    done_keys = set()
    if os.path.exists(output_filename):
        done_df = pd.read_csv(output_filename, encoding='utf-8-sig')
        done_keys = set(zip(done_df["股票代码_StkCd"], done_df["首次授予确定公告日_FistDeterDt"].astype(str)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, row in df.iterrows():
            key = (row['股票代码_StkCd'], str(row['首次授予确定公告日_FistDeterDt']))
            if key in done_keys:
                continue
            future = executor.submit(process_row, i, row, client, model)
            futures[future] = i

        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            result, error = future.result()
            if error is None:
                pd.DataFrame([result]).to_csv(
                    output_filename,
                    mode='a',
                    header=not os.path.exists(output_filename) or os.path.getsize(output_filename) == 0,
                    index=False,
                    encoding='utf-8-sig'
                )
            else:
                print(f"第 {i+1} 条处理失败：{result['error']}\n模型返回：{result['reply']}")
                with open("结构化失败日志_激励计划.txt", "a", encoding="utf-8") as f:
                    f.write(
                        f"第 {i+1} 条失败：\n"
                        f"原文：{result['raw_text']}\n"
                        f"返回：{result['reply']}\n"
                        f"错误：{result['error']}\n\n"
                        f"------------------------------\n"
                    )

    print(f"处理完成：{input_filename} → {output_filename}")

if __name__ == "__main__":
    input_file = "激励计划表.xlsx"
    output_file = os.path.join("结构化激励计划", "结构化激励计划结果.csv")
    process_equity_incentive_file(
        input_file,
        output_file,
        api_key=os.environ.get("ARK_API_KEY"),
        model="doubao-1-5-lite-32k-250115",
        max_workers=20
    )
