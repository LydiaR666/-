"""
下载管理器模块
协调并购公告的搜索、下载、验证和记录
"""

import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from cninfo_api import CninfoAPI
from content_validator import ContentValidator
import config


class DownloadManager:
    """下载管理器"""

    def __init__(self, output_dir: str = './output'):
        self.output_dir = output_dir
        self.pdf_dir = os.path.join(output_dir, config.SAVE_CONFIG['pdf_folder'])
        self.log_file = os.path.join(output_dir, config.SAVE_CONFIG['log_file'])
        self.valid_file = os.path.join(output_dir, config.SAVE_CONFIG['valid_file'])
        self.invalid_file = os.path.join(output_dir, config.SAVE_CONFIG['invalid_file'])

        # 创建目录
        os.makedirs(self.pdf_dir, exist_ok=True)

        # 初始化API和验证器
        self.api = CninfoAPI()
        self.validator = ContentValidator()

        # 下载记录
        self.download_log = []
        self.valid_announcements = []
        self.invalid_announcements = []

        # 加载已有记录（如果存在）
        self._load_existing_logs()

    def _load_existing_logs(self):
        """加载已有的下载记录"""
        if os.path.exists(self.log_file):
            try:
                df = pd.read_excel(self.log_file)
                self.download_log = df.to_dict('records')
                print(f"已加载 {len(self.download_log)} 条下载记录")
            except Exception as e:
                print(f"加载下载记录失败: {str(e)}")

    def process_ma_data(
        self,
        ma_data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate_only: bool = False,
        max_workers: int = 3
    ):
        """
        处理并购简表数据，批量下载公告

        Args:
            ma_data: 并购简表DataFrame
            start_date: 筛选起始日期（可选）
            end_date: 筛选结束日期（可选）
            validate_only: 仅验证已下载的文件，不重新下载
            max_workers: 并发数
        """
        print(f"\n开始处理并购数据，共 {len(ma_data)} 条记录")

        # 筛选日期范围
        if start_date or end_date:
            ma_data = self._filter_by_date(ma_data, start_date, end_date)
            print(f"日期筛选后剩余 {len(ma_data)} 条记录")

        # 处理每条并购事件
        for idx, row in tqdm(ma_data.iterrows(), total=len(ma_data), desc="处理并购事件"):
            try:
                self._process_single_event(row, validate_only)
            except Exception as e:
                print(f"\n处理事件失败 {row.get('EventID', 'Unknown')}: {str(e)}")
                continue

        # 保存结果
        self._save_results()

        # 打印统计信息
        self._print_summary()

    def _process_single_event(self, event: pd.Series, validate_only: bool = False):
        """
        处理单个并购事件

        Args:
            event: 并购事件数据
            validate_only: 仅验证，不下载
        """
        event_id = event.get('EventID', 'Unknown')
        stock_code = config.format_stock_code(event['Symbol'])
        first_date = self._format_date(event.get('FirstDeclareDate'))
        latest_date = self._format_date(event.get('LatestDeclareDate'))
        finish_date = self._format_date(event.get('FinishDeclareDate'))

        if not first_date:
            print(f"\n事件 {event_id} 缺少首次公告日期，跳过")
            return

        # 搜索相关公告
        if not validate_only:
            announcements = self.api.get_ma_related_announcements(
                stock_code=stock_code,
                first_date=first_date,
                latest_date=latest_date,
                finish_date=finish_date
            )

            print(f"\n事件 {event_id} ({stock_code}): 找到 {len(announcements)} 个相关公告")

            # 下载和验证公告
            for ann in announcements:
                self._download_and_validate_announcement(ann, event)

    def _download_and_validate_announcement(
        self,
        announcement: Dict,
        event: pd.Series
    ):
        """
        下载并验证单个公告

        Args:
            announcement: 公告信息
            event: 并购事件数据
        """
        ann_info = self.api.format_announcement_info(announcement)
        ann_id = ann_info['announcement_id']
        adj_ann_id = ann_info['adj_announcement_id']

        # 检查是否已下载
        if self._is_downloaded(ann_id):
            return

        # 生成文件名
        filename = self._generate_filename(ann_info, event)
        save_path = os.path.join(self.pdf_dir, filename)

        # 下载公告
        print(f"  下载: {ann_info['title'][:50]}...")
        download_success = self.api.download_announcement(
            ann_id, adj_ann_id, save_path
        )

        if not download_success:
            self._log_download(ann_info, event, 'download_failed', None)
            return

        # 验证内容
        is_valid, validation_result = self.validator.validate_pdf(save_path)

        # 记录结果
        status = 'valid' if is_valid else 'invalid'
        self._log_download(ann_info, event, status, validation_result)

        # 打印验证摘要
        if is_valid:
            print(f"    ✓ 有效公告")
        else:
            print(f"    ✗ 无效公告: {validation_result.get('matched_categories', 0)}/4 类别匹配")

    def _is_downloaded(self, announcement_id: str) -> bool:
        """检查公告是否已下载"""
        return any(log['announcement_id'] == announcement_id for log in self.download_log)

    def _log_download(
        self,
        ann_info: Dict,
        event: pd.Series,
        status: str,
        validation_result: Optional[Dict]
    ):
        """
        记录下载结果

        Args:
            ann_info: 公告信息
            event: 并购事件
            status: 状态（valid/invalid/download_failed）
            validation_result: 验证结果
        """
        log_entry = {
            'event_id': event.get('EventID'),
            'stock_code': ann_info['stock_code'],
            'stock_name': ann_info['stock_name'],
            'announcement_id': ann_info['announcement_id'],
            'title': ann_info['title'],
            'announcement_time': ann_info['time'],
            'download_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': status,
            'file_name': self._generate_filename(ann_info, event),
        }

        # 添加验证详情
        if validation_result:
            if 'error' in validation_result:
                log_entry['validation_error'] = validation_result['error']
            else:
                log_entry['matched_categories'] = validation_result['matched_categories']
                log_entry['total_matches'] = validation_result['total_matches']
                log_entry['has_motive'] = validation_result['categories'].get('motive', {}).get('matched', False)
                log_entry['has_target'] = validation_result['categories'].get('target_business', {}).get('matched', False)
                log_entry['has_purpose'] = validation_result['categories'].get('purpose', {}).get('matched', False)
                log_entry['has_impact'] = validation_result['categories'].get('impact', {}).get('matched', False)

        self.download_log.append(log_entry)

        # 分类记录
        if status == 'valid':
            self.valid_announcements.append(log_entry)
        elif status == 'invalid':
            self.invalid_announcements.append(log_entry)

    @staticmethod
    def _generate_filename(ann_info: Dict, event: pd.Series) -> str:
        """
        生成文件名

        Args:
            ann_info: 公告信息
            event: 并购事件

        Returns:
            文件名
        """
        event_id = event.get('EventID', 'Unknown')
        stock_code = ann_info['stock_code']
        date = ann_info['time'].replace('-', '')[:8]
        ann_id = ann_info['announcement_id']

        # 清理标题中的特殊字符
        title = ann_info['title'][:30]
        title = title.replace('/', '_').replace('\\', '_').replace(':', '_')

        filename = f"{event_id}_{stock_code}_{date}_{ann_id}_{title}.pdf"
        return filename

    @staticmethod
    def _format_date(date_value) -> Optional[str]:
        """
        格式化日期为 YYYY-MM-DD

        Args:
            date_value: 日期值（字符串、datetime或其他）

        Returns:
            格式化后的日期字符串，或None
        """
        if pd.isna(date_value):
            return None

        if isinstance(date_value, str):
            # 尝试解析字符串
            for fmt in config.DATE_FORMATS:
                try:
                    date_obj = datetime.strptime(date_value, fmt)
                    return date_obj.strftime('%Y-%m-%d')
                except:
                    continue
            return None

        if isinstance(date_value, datetime):
            return date_value.strftime('%Y-%m-%d')

        # 尝试转换为字符串再解析
        try:
            return pd.to_datetime(date_value).strftime('%Y-%m-%d')
        except:
            return None

    @staticmethod
    def _filter_by_date(
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """按日期范围筛选数据"""
        if start_date:
            df = df[df['FirstDeclareDate'] >= start_date]
        if end_date:
            df = df[df['FirstDeclareDate'] <= end_date]
        return df

    def _save_results(self):
        """保存结果到文件"""
        print("\n保存结果...")

        # 保存下载日志
        if self.download_log:
            df_log = pd.DataFrame(self.download_log)
            df_log.to_excel(self.log_file, index=False)
            print(f"下载日志已保存: {self.log_file}")

        # 保存有效公告列表
        if self.valid_announcements:
            df_valid = pd.DataFrame(self.valid_announcements)
            df_valid.to_excel(self.valid_file, index=False)
            print(f"有效公告列表已保存: {self.valid_file}")

        # 保存无效公告列表
        if self.invalid_announcements:
            df_invalid = pd.DataFrame(self.invalid_announcements)
            df_invalid.to_excel(self.invalid_file, index=False)
            print(f"无效公告列表已保存: {self.invalid_file}")

    def _print_summary(self):
        """打印统计摘要"""
        total = len(self.download_log)
        valid = len(self.valid_announcements)
        invalid = len(self.invalid_announcements)
        failed = total - valid - invalid

        print("\n" + "=" * 60)
        print("下载统计")
        print("=" * 60)
        print(f"总下载数: {total}")
        print(f"有效公告: {valid} ({valid/total*100:.1f}%)" if total > 0 else "有效公告: 0")
        print(f"无效公告: {invalid} ({invalid/total*100:.1f}%)" if total > 0 else "无效公告: 0")
        print(f"下载失败: {failed}")
        print("=" * 60)
