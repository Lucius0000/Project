# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 19:39:36 2025

@author: Lucius
"""

import os
import fitz  # PyMuPDF
import shutil


SRC_ROOT = r"C:\Users\Lucius\Desktop\report_data"
DST_ROOT = r"C:\Users\Lucius\Desktop\整理后公司报告"
os.makedirs(DST_ROOT, exist_ok=True)


success_log = os.path.join(DST_ROOT, "success_log.txt")
error_log = os.path.join(DST_ROOT, "error_log.txt")
log_success = open(success_log, "a", encoding="utf-8")
log_error = open(error_log, "a", encoding="utf-8")

def convert_pdf_to_txt(pdf_path, txt_path):
    try:
        if os.path.exists(txt_path):
            print(f"[SKIP] 已存在: {txt_path}")
            return True
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] {pdf_path} --> {txt_path}")
        log_success.write(pdf_path + "\n")
        return True
    except Exception as e:
        print(f"[ERROR] {pdf_path}: {e}")
        log_error.write(f"{pdf_path}: {e}\n")
        return False

def process_all_companies():
    for company_folder in os.listdir(SRC_ROOT):
        company_path = os.path.join(SRC_ROOT, company_folder)
        if not os.path.isdir(company_path):
            continue

        dst_company_dir = os.path.join(DST_ROOT, company_folder)
        os.makedirs(dst_company_dir, exist_ok=True)

        # 复制报告信息
        for file in os.listdir(company_path):
            if file.endswith("_报告信息.xlsx"):
                shutil.copy(os.path.join(company_path, file), os.path.join(dst_company_dir, file))
                break 

        # 定位 PDF 路径
        reports_dir = os.path.join(company_path, "disclosure", "reports")
        if not os.path.exists(reports_dir):
            continue

        for stock_code_folder in os.listdir(reports_dir):
            pdf_dir = os.path.join(reports_dir, stock_code_folder)
            if not os.path.isdir(pdf_dir):
                continue

            for file in os.listdir(pdf_dir):
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(pdf_dir, file)
                    txt_name = file.replace(".pdf", ".txt")
                    txt_path = os.path.join(dst_company_dir, txt_name)
                    convert_pdf_to_txt(pdf_path, txt_path)

process_all_companies()

log_success.close()
log_error.close()
print("批量转换与整理完成！")
