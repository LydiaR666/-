"""
生成示例并购简表数据
用于测试和演示
"""

import pandas as pd
from datetime import datetime, timedelta

# 创建示例数据
example_data = [
    {
        'EventID': '20210001',
        'Symbol': '000001',
        'FirstDeclareDate': '2021-03-15',
        'LatestDeclareDate': '2021-05-20',
        'FinishDeclareDate': '2021-06-30',
        'Buyer': '平安银行',
        'Seller': '某资产管理公司',
        'Underlying': '某金融资产包',
        'BusinessID': '2021000001A001',
        'RestructuringTypeID': 'S3001',
        'MergerTypeID': '1',
        'Outline': '收购金融资产包以优化资产结构'
    },
    {
        'EventID': '26210002',
        'Symbol': '000002',
        'FirstDeclareDate': '2021-06-10',
        'LatestDeclareDate': '2021-08-15',
        'FinishDeclareDate': '2021-09-30',
        'Buyer': '万科A',
        'Seller': '某房地产公司股东',
        'Underlying': '目标公司51%股权',
        'BusinessID': '2021000002B001',
        'RestructuringTypeID': 'S3008',
        'MergerTypeID': '1',
        'Outline': '通过股权转让扩大市场份额'
    },
    {
        'EventID': '20210003',
        'Symbol': '600000',
        'FirstDeclareDate': '2021-09-01',
        'LatestDeclareDate': '2021-10-15',
        'FinishDeclareDate': None,
        'Buyer': '浦发银行',
        'Seller': '某金融科技公司',
        'Underlying': '金融科技相关资产',
        'BusinessID': '2021600000A001',
        'RestructuringTypeID': 'S3001',
        'MergerTypeID': '3',
        'Outline': '收购金融科技资产提升数字化能力'
    },
    {
        'EventID': '23210004',
        'Symbol': '600016',
        'FirstDeclareDate': '2021-12-01',
        'LatestDeclareDate': '2022-02-28',
        'FinishDeclareDate': '2022-03-31',
        'Buyer': '民生银行',
        'Seller': None,
        'Underlying': '某小型银行',
        'BusinessID': '2021600016C001',
        'RestructuringTypeID': 'S3004',
        'MergerTypeID': '1',
        'Outline': '吸收合并区域性银行扩大业务规模'
    },
    {
        'EventID': '20220005',
        'Symbol': '000333',
        'FirstDeclareDate': '2022-03-20',
        'LatestDeclareDate': '2022-05-10',
        'FinishDeclareDate': '2022-06-15',
        'Buyer': '美的集团',
        'Seller': '某科技公司',
        'Underlying': '智能制造相关资产',
        'BusinessID': '2022000333A001',
        'RestructuringTypeID': 'S3001',
        'MergerTypeID': '2',
        'Outline': '收购智能制造资产完善产业链'
    },
]

def generate_example_file(output_path: str = 'example_ma_data.xlsx'):
    """
    生成示例Excel文件

    Args:
        output_path: 输出文件路径
    """
    df = pd.DataFrame(example_data)

    # 保存为Excel
    df.to_excel(output_path, index=False)
    print(f"示例文件已生成: {output_path}")
    print(f"包含 {len(df)} 条示例数据")
    print("\n字段说明:")
    print("- EventID: 事件ID（必需）")
    print("- Symbol: 证券代码（必需）")
    print("- FirstDeclareDate: 首次公告日期（必需）")
    print("- LatestDeclareDate: 最新公告日期（可选）")
    print("- FinishDeclareDate: 完成公告日期（可选）")
    print("- Buyer: 买方")
    print("- Seller: 卖方")
    print("- Underlying: 标的方")
    print("- BusinessID: 业务编码")
    print("- RestructuringTypeID: 重组类型编码")
    print("- MergerTypeID: 并购类型编码")
    print("- Outline: 交易概述")
    print("\n使用方法:")
    print(f"python main.py --input {output_path} --output ./output")


if __name__ == '__main__':
    generate_example_file()
