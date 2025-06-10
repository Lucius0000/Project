from shreport import SH
from pathlib import Path
import pandas as pd
import time
from urllib.parse import urlparse
import os


cookies = {"Cookie": 'ba17301551dcbaf9_gdp_session_id=e789411f-e10a-48fb-92fc-4c0c7cbdbc49; gdp_user_id=gioenc-e33d905b%2C48g6%2C5ec7%2Ca291%2C0abadea18978; ba17301551dcbaf9_gdp_session_id_sent=e789411f-e10a-48fb-92fc-4c0c7cbdbc49; ba17301551dcbaf9_gdp_sequence_ids={%22globalKey%22:13%2C%22VISIT%22:2%2C%22PAGE%22:5%2C%22VIEW_CLICK%22:8}'} 
sh = SH(cookies)
save_dir = Path("report_data")
save_dir.mkdir(exist_ok=True)


print("获取上交所上市公司目录...")
df_company = sh.companys()
df_company.to_excel(save_dir / "上交所公司名录.xlsx", index=False)
print(f"共获取 {len(df_company)} 家公司。")

# 加载已完成公司记录
finished_path = Path("finished.txt")
if finished_path.exists():
    with open(finished_path, "r", encoding="utf-8") as f:
        finished_codes = set(line.strip() for line in f)
else:
    finished_codes = set()

# 遍历公司，处理未完成部分
all_disclosures = []

for idx, row in df_company.iterrows():
    code = row["code"]
    name = row["name"]

    if code in finished_codes:
        print(f"已跳过：{name}（{code}）")
        continue

    print(f"\n正在处理：{name}（{code}）...")

    try:
        # 创建公司目录
        company_dir = save_dir / f"{code}_{name}"
        company_dir.mkdir(exist_ok=True)

        # 下载 PDF 报告
        sh.download(code=code, savepath=company_dir)

        # 获取报告元信息
        df_info = sh.disclosure(code=code)
        df_info["year"] = pd.to_numeric(df_info["year"], errors="coerce")
        df_info = df_info[df_info["year"].between(2010, 2024)]

        # 添加本地路径列
        def get_filename_from_url(url):
            return os.path.basename(urlparse(url).path)
        df_info["local_path"] = df_info["pdf"].apply(
            lambda url: str(company_dir / get_filename_from_url(url))
        )

        # 添加公司名
        df_info["company"] = name

        # 保存单公司信息
        df_info.to_excel(company_dir / f"{code}_{name}_报告信息.xlsx", index=False)
        all_disclosures.append(df_info)

        # 添加到已完成列表
        with open("finished.txt", "a", encoding="utf-8") as f:
            f.write(f"{code}\n")

        time.sleep(1.5)

    except Exception as e:
        print(f"错误：{code} - {name}，错误信息：{e}")
        continue


# 合并所有公司数据
if all_disclosures:
    print("\n正在保存所有公司披露信息汇总表...")
    df_all = pd.concat(all_disclosures, ignore_index=True)
    df_all.to_excel(save_dir / "2010_2024_定期报告信息总表.xlsx", index=False)
    print("全部完成！")
else:
    print("本次未处理任何新公司，未生成总表。")
