"""
Excel列名格式转换工具
自动将你的Excel文件转换为程序需要的格式
"""

import pandas as pd
import os
from datetime import datetime

def convert_excel_format(input_file, output_file=None):
    """
    转换Excel文件格式

    Args:
        input_file: 输入的Excel文件路径
        output_file: 输出的Excel文件路径（如果不指定，会在原文件名后加"_已转换"）
    """
    print("=" * 60)
    print("Excel格式转换工具")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 文件不存在: {input_file}")
        return False

    # 读取Excel
    print(f"\n正在读取文件: {input_file}")
    try:
        df = pd.read_excel(input_file)
        print(f"✓ 成功读取 {len(df)} 行数据")
    except Exception as e:
        print(f"❌ 读取文件失败: {str(e)}")
        return False

    # 显示原始列名
    print(f"\n原始列名: {', '.join(df.columns)}")

    # 列名映射规则（支持多种可能的列名）
    column_mapping = {
        # EventID 的各种可能写法
        'event_id': 'EventID',
        'eventid': 'EventID',
        'Event_ID': 'EventID',
        'EVENTID': 'EventID',
        '事件ID': 'EventID',
        '事件编号': 'EventID',

        # Symbol 的各种可能写法
        'symbol': 'Symbol',
        'SYMBOL': 'Symbol',
        'stock_code': 'Symbol',
        'StockCode': 'Symbol',
        '股票代码': 'Symbol',
        '证券代码': 'Symbol',
        '代码': 'Symbol',

        # FirstDeclareDate 的各种可能写法
        'first_declare_date': 'FirstDeclareDate',
        'firstdeclaredate': 'FirstDeclareDate',
        'FIRSTDECLAREDATE': 'FirstDeclareDate',
        'First_Declare_Date': 'FirstDeclareDate',
        '首次披露日期': 'FirstDeclareDate',
        '首次公告日期': 'FirstDeclareDate',
        '首次公告日': 'FirstDeclareDate',

        # LatestDeclareDate 的各种可能写法
        'latest_declare_date': 'LatestDeclareDate',
        'latestdeclaredate': 'LatestDeclareDate',
        'LATESTDECLAREDATE': 'LatestDeclareDate',
        'Latest_Declare_Date': 'LatestDeclareDate',
        '最新披露日期': 'LatestDeclareDate',
        '最新公告日期': 'LatestDeclareDate',

        # FinishDeclareDate 的各种可能写法
        'finish_declare_date': 'FinishDeclareDate',
        'finishdeclaredate': 'FinishDeclareDate',
        'FINISHDECLAREDATE': 'FinishDeclareDate',
        'Finish_Declare_Date': 'FinishDeclareDate',
        '完成披露日期': 'FinishDeclareDate',
        '完成公告日期': 'FinishDeclareDate',
    }

    # 重命名列
    renamed_columns = {}
    for old_col in df.columns:
        if old_col in column_mapping:
            new_col = column_mapping[old_col]
            renamed_columns[old_col] = new_col
        # 如果已经是正确的格式，不改变
        elif old_col in ['EventID', 'Symbol', 'FirstDeclareDate', 'LatestDeclareDate', 'FinishDeclareDate']:
            continue
        # 其他列保持不变

    if renamed_columns:
        df.rename(columns=renamed_columns, inplace=True)
        print(f"\n✓ 已转换列名:")
        for old, new in renamed_columns.items():
            print(f"  {old} → {new}")
    else:
        print("\n✓ 列名已经是标准格式，无需转换")

    # 验证必需字段
    required_fields = ['Symbol', 'FirstDeclareDate']
    missing_fields = [field for field in required_fields if field not in df.columns]

    if missing_fields:
        print(f"\n❌ 错误: 缺少必需字段: {', '.join(missing_fields)}")
        print("\n请确保Excel文件包含以下列:")
        print("  - Symbol (股票代码)")
        print("  - FirstDeclareDate (首次公告日期)")
        return False

    # 如果没有EventID列，自动生成
    if 'EventID' not in df.columns:
        print("\n⚠️  警告: 缺少EventID列，自动生成事件ID...")
        df['EventID'] = range(1, len(df) + 1)
        df['EventID'] = df['EventID'].astype(str)

    # 标准化日期格式
    print("\n正在标准化日期格式...")
    date_columns = ['FirstDeclareDate', 'LatestDeclareDate', 'FinishDeclareDate']

    for col in date_columns:
        if col in df.columns:
            # 转换为pandas datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # 格式化为 YYYY-MM-DD
            df[col] = df[col].dt.strftime('%Y-%m-%d')
            print(f"  ✓ {col}")

    # 标准化股票代码（补齐到6位）
    print("\n正在标准化股票代码...")
    df['Symbol'] = df['Symbol'].astype(str).str.zfill(6)

    # 显示转换后的列名
    print(f"\n转换后的列名: {', '.join(df.columns)}")

    # 保存文件
    if output_file is None:
        # 在原文件名后添加 "_已转换"
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_已转换.xlsx"

    print(f"\n正在保存文件: {output_file}")
    try:
        df.to_excel(output_file, index=False)
        print(f"✓ 成功保存!")
    except Exception as e:
        print(f"❌ 保存文件失败: {str(e)}")
        return False

    # 显示数据预览
    print("\n数据预览（前5行）:")
    print(df[['EventID', 'Symbol', 'FirstDeclareDate']].head())

    print("\n" + "=" * 60)
    print("✓ 格式转换完成!")
    print(f"转换后的文件: {output_file}")
    print("\n你现在可以使用这个文件运行爬虫:")
    print(f'python main.py --input "{output_file}" --output ./结果')
    print("=" * 60)

    return True


if __name__ == '__main__':
    print("\n请输入你的Excel文件路径")
    print("（可以直接拖拽文件到这个窗口，或复制粘贴路径）")
    print("\n示例: C:\\Users\\10657\\Desktop\\并购公告\\并购公告下载.xlsx")
    print("-" * 60)

    input_file = input("\n文件路径: ").strip().strip('"')

    if not input_file:
        print("\n❌ 错误: 未输入文件路径")
        input("\n按回车键关闭...")
        exit(1)

    # 转换文件
    success = convert_excel_format(input_file)

    if not success:
        input("\n按回车键关闭...")
        exit(1)

    input("\n按回车键关闭...")
