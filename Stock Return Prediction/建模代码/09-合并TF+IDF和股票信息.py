import pandas as pd
import os

stock_excel_path = r"C:\Users\Lucius\Desktop\股票信息(清洗).xlsx"
fin_excel_path = r"C:\Users\Lucius\Desktop\财务报表.xlsx"

stock_pickle_path = r"C:\Users\Lucius\Desktop\stock_df.pkl"
fin_pickle_path = r"C:\Users\Lucius\Desktop\fin_df.pkl"

output_csv_path = r"C:\Users\Lucius\Desktop\结构化数据.csv"

def load_data():
    if os.path.exists(stock_pickle_path):
        stock_df = pd.read_pickle(stock_pickle_path)
        print("股票数据已从缓存载入")
    else:
        stock_df = pd.read_excel(stock_excel_path)
        stock_df.to_pickle(stock_pickle_path)
        print("股票数据首次读取并缓存完毕")

    if os.path.exists(fin_pickle_path):
        fin_df = pd.read_pickle(fin_pickle_path)
        print("财报数据已从缓存载入")
    else:
        fin_df = pd.read_excel(fin_excel_path)
        fin_df.to_pickle(fin_pickle_path)
        print("财报数据首次读取并缓存完毕")

    return stock_df, fin_df

def preprocess(stock_df, fin_df):
    stock_df['StkCd_Match'] = stock_df['股票代码_Stkcd'].astype(str).str.strip()
    fin_df['StkCd_Match'] = fin_df['A股股票代码_A_StkCd'].astype(str).str.strip()

    stock_df['日期_Date'] = pd.to_datetime(stock_df['日期_Date'], errors='coerce')
    fin_df['信息发布日期_InfoPubDt'] = pd.to_datetime(fin_df['信息发布日期_InfoPubDt'], errors='coerce')

    stock_df = stock_df.dropna(subset=['StkCd_Match', '日期_Date'])
    fin_df = fin_df.dropna(subset=['StkCd_Match', '信息发布日期_InfoPubDt'])

    a_share_codes = fin_df['StkCd_Match'].unique()
    stock_df = stock_df[stock_df['StkCd_Match'].isin(a_share_codes)]

    return stock_df, fin_df


def merge_and_write(stock_df, fin_df, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)

    total_rows = 0
    matched_rows = 0
    unmatched_rows = 0
    used_fin_keys = set()

    print("始逐股票合并并写入 CSV ...")
    a_share_codes = fin_df['StkCd_Match'].unique()

    for i, (code, group_stock) in enumerate(stock_df.groupby('StkCd_Match')):
        group_fin = fin_df[fin_df['StkCd_Match'] == code]

        group_stock = group_stock.sort_values('日期_Date')
        group_fin = group_fin.sort_values('信息发布日期_InfoPubDt')

        merged_group = pd.merge_asof(
            group_stock,
            group_fin,
            by='StkCd_Match',
            left_on='日期_Date',
            right_on='信息发布日期_InfoPubDt',
            direction='backward'
        )
        
        merged_group['滞后天数'] = (merged_group['日期_Date'] - merged_group['信息发布日期_InfoPubDt']).dt.days


        total_rows += len(merged_group)
        matched = merged_group['信息发布日期_InfoPubDt'].notna().sum()
        matched_rows += matched
        unmatched_rows += len(merged_group) - matched

        used_fin_keys.update(
            merged_group.loc[merged_group['信息发布日期_InfoPubDt'].notna(), ['StkCd_Match', '信息发布日期_InfoPubDt']].apply(
                lambda row: (row['StkCd_Match'], row['信息发布日期_InfoPubDt']), axis=1
            )
        )

        merged_group.to_csv(
            output_path,
            mode='a',
            index=False,
            header=(i == 0),
            encoding='utf-8-sig'
        )

        if (i + 1) % 100 == 0 or (i + 1) == len(a_share_codes):
            print(f"已完成 {i + 1} / {len(a_share_codes)} 支 A 股")

    fin_df['key'] = list(zip(fin_df['StkCd_Match'], fin_df['信息发布日期_InfoPubDt']))
    fin_df['是否被匹配'] = fin_df['key'].isin(used_fin_keys)

    print("\n匹配统计结果：")
    print(f"行情数据总记录数：{total_rows}")
    print(f"成功匹配财报记录数：{matched_rows}")
    print(f"未匹配财报记录数：{unmatched_rows}")
    print(f"财报记录总数（A 股）：{len(fin_df)}")
    print(f"被使用财报数：{fin_df['是否被匹配'].sum()}")
    print(f"未被使用财报数：{len(fin_df) - fin_df['是否被匹配'].sum()}")



if __name__ == "__main__":
    stock_df, fin_df = load_data()
    stock_df, fin_df = preprocess(stock_df, fin_df)
    merge_and_write(stock_df, fin_df, output_csv_path)
