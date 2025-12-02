"""
巨潮资讯并购公告爬虫 - 主程序
"""

import argparse
import pandas as pd
import os
import sys
from datetime import datetime

from download_manager import DownloadManager
import config


def load_ma_data(file_path: str) -> pd.DataFrame:
    """
    加载并购简表数据

    Args:
        file_path: 文件路径（支持Excel和CSV）

    Returns:
        DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 根据文件扩展名选择读取方式
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("不支持的文件格式，请使用Excel或CSV文件")

    # 验证必需字段
    required_fields = ['Symbol', 'FirstDeclareDate']
    missing_fields = [field for field in required_fields if field not in df.columns]

    if missing_fields:
        raise ValueError(f"缺少必需字段: {', '.join(missing_fields)}")

    print(f"成功加载并购数据: {len(df)} 条记录")
    print(f"字段: {', '.join(df.columns.tolist())}")

    return df


def validate_date_format(date_str: str) -> bool:
    """验证日期格式"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='从巨潮资讯网站批量爬取并购公告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python main.py --input ma_data.xlsx --output ./announcements

  # 指定日期范围
  python main.py --input ma_data.xlsx --output ./announcements --start-date 2020-01-01 --end-date 2023-12-31

  # 只验证已下载的文件
  python main.py --input ma_data.xlsx --output ./announcements --validate-only

  # 设置并发数
  python main.py --input ma_data.xlsx --output ./announcements --workers 5
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='并购简表文件路径（Excel或CSV格式）'
    )

    parser.add_argument(
        '--output', '-o',
        default='./output',
        help='输出目录（默认: ./output）'
    )

    parser.add_argument(
        '--start-date',
        help='起始日期（格式: YYYY-MM-DD）'
    )

    parser.add_argument(
        '--end-date',
        help='结束日期（格式: YYYY-MM-DD）'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证已下载的文件，不重新下载'
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=config.REQUEST_CONFIG['max_workers'],
        help=f'并发下载数（默认: {config.REQUEST_CONFIG["max_workers"]}）'
    )

    parser.add_argument(
        '--sample', '-s',
        type=int,
        help='仅处理前N条数据（用于测试）'
    )

    args = parser.parse_args()

    # 验证日期格式
    if args.start_date and not validate_date_format(args.start_date):
        print("错误: 起始日期格式不正确，应为 YYYY-MM-DD")
        sys.exit(1)

    if args.end_date and not validate_date_format(args.end_date):
        print("错误: 结束日期格式不正确，应为 YYYY-MM-DD")
        sys.exit(1)

    # 打印配置信息
    print("=" * 60)
    print("巨潮资讯并购公告爬虫")
    print("=" * 60)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    if args.start_date:
        print(f"起始日期: {args.start_date}")
    if args.end_date:
        print(f"结束日期: {args.end_date}")
    print(f"并发数: {args.workers}")
    print(f"验证模式: {'仅验证' if args.validate_only else '下载并验证'}")
    print("=" * 60)

    try:
        # 加载并购数据
        ma_data = load_ma_data(args.input)

        # 如果指定了sample，只处理前N条
        if args.sample:
            ma_data = ma_data.head(args.sample)
            print(f"\n测试模式: 仅处理前 {args.sample} 条数据")

        # 创建下载管理器
        manager = DownloadManager(output_dir=args.output)

        # 处理数据
        manager.process_ma_data(
            ma_data=ma_data,
            start_date=args.start_date,
            end_date=args.end_date,
            validate_only=args.validate_only,
            max_workers=args.workers
        )

        print("\n✓ 处理完成!")

    except KeyboardInterrupt:
        print("\n\n用户中断，正在保存已下载的数据...")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
