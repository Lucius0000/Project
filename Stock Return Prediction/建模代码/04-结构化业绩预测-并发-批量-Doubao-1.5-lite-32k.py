# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 16:41:26 2025

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

def generate_financial_forecast_prompt(text):
    return f"""
以下是某家上市公司披露的业绩预测内容：
'''{text}'''

**任务说明**
请从上面的文本中提取以下结构化财务预测信息：

请按以下格式返回，若某项信息未提及请写为 null（不要省略字段）：
{{
"净利润下限（万元）": ________,
"净利润上限（万元）": ________,
"同比变动下限（%）": ________,
"同比变动上限（%）": ________,
"每股收益下限（元）": ________,
"每股收益上限（元）": ________,
"是否预盈": "是" / "否",
}}

**特别说明**
- 如果文本为“亏损”或“下降”，请将对应数值写为负数。
- 所有金额单位统一为万元；每股收益单位为元。
- 所有字段的值必须是数字或 null，如果涉及到计算，请直接输出计算结果，不得返回算式。
- 请仅返回上述 JSON 格式结果，不要输出诸如“注：”等任何多余解释。
"""

# 调用 API 获取结构化结果
def call_doubao_api(client, prompt, model="doubao-1.5-pro-32k-250115"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是财务结构化提取专家"},
            {"role": "user", "content": prompt}
        ],
        stream=False,
        temperature=0
    )
    return response.choices[0].message.content

# 提取合法 JSON 主体
def extract_json_from_text(text):
    try:
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            return json.loads(match.group())
        else:
            raise ValueError("未找到合法 JSON 对象")
    except Exception as e:
        raise ValueError(f"JSON 解析失败：{e}")

# 工作线程函数
def process_row(i, row, client, model):
    raw_text = str(row['截取_业绩预测内容'])
    prompt = generate_financial_forecast_prompt(raw_text)
    try:
        reply = call_doubao_api(client, prompt, model=model)
        expected_fields = [
            "净利润下限（万元）", "净利润上限（万元）",
            "同比变动下限（%）", "同比变动上限（%）",
            "每股收益下限（元）", "每股收益上限（元）",
            "是否预盈"
        ]
        raw_result = extract_json_from_text(reply)
        result_dict = {key: raw_result.get(key, None) for key in expected_fields}
        result_dict.update({
            'A股股票代码_A_StkCd': row['A股股票代码_A_StkCd'],
            '截止日期_EndDt': row['截止日期_EndDt'],
            '交易日期（年月）': row['交易日期（年月）']
        })
        return result_dict, None
    except Exception as e:
        reply_text = reply if 'reply' in locals() else '无返回内容'
        return {
            "index": i,
            "raw_text": raw_text,
            "reply": reply_text,
            "error": str(e)
        }, e

def process_forecast_file(input_filename, output_filename, api_key, model, max_workers=5):
    client = init_client(api_key)
    df = pd.read_csv(input_filename, encoding='gbk')

    done_keys = set()
    if os.path.exists(output_filename):
        done_df = pd.read_csv(output_filename, encoding='utf-8-sig')
        done_keys = set(zip(done_df["A股股票代码_A_StkCd"], done_df["截止日期_EndDt"].astype(str)))

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, row in df.iterrows():
            key = (row['A股股票代码_A_StkCd'], str(row['截止日期_EndDt']))
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
                with open("结构化失败日志.txt", "a", encoding="utf-8") as f:
                    f.write(
                        f"第 {i+1} 条失败：\n"
                        f"原文：{result['raw_text']}\n"
                        f"返回：{result['reply']}\n"
                        f"错误：{result['error']}\n\n"
                        f"------------------------------\n"
                    )

    print(f"处理完成：{input_filename} → {output_filename}")


if __name__ == "__main__":
    os.makedirs("结构化输出", exist_ok=True)

    for i in range(15, 16):  
        input_file = f"业绩预测表_第{i}份.csv"
        output_file = os.path.join("结构化输出", f"结构化业绩预测结果_第{i}份.csv")

        print(f"\n开始处理：{input_file}")
        process_forecast_file(
            input_filename=input_file,
            output_filename=output_file,
            api_key=os.environ.get("ARK_API_KEY"),
            model="doubao-1-5-lite-32k-250115",
            max_workers=20
        )

