"""
巨潮资讯API接口模块
提供搜索公告、获取公告详情、下载公告等功能
"""

import requests
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from retry import retry
import config


class CninfoAPI:
    """巨潮资讯API封装类"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(config.HEADERS)

    @retry(tries=3, delay=2)
    def search_announcements(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        keywords: Optional[List[str]] = None,
        page_num: int = 1,
        page_size: int = 50
    ) -> Dict:
        """
        搜索公告

        Args:
            stock_code: 股票代码（6位）
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            keywords: 关键词列表
            page_num: 页码
            page_size: 每页数量

        Returns:
            包含公告列表的字典
        """
        # 格式化股票代码
        stock_code = config.format_stock_code(stock_code)

        # 构建搜索关键词
        search_key = f"{stock_code}"
        if keywords:
            search_key += " " + " ".join(keywords)

        # API参数
        params = {
            'searchkey': search_key,
            'sdate': start_date,
            'edate': end_date,
            'isfulltext': 'false',
            'sortName': 'time',
            'sortType': 'desc',
            'pageNum': page_num,
            'pageSize': page_size,
        }

        try:
            response = self.session.post(
                config.CNINFO_API['search_url'],
                data=params,
                timeout=config.REQUEST_CONFIG['timeout']
            )
            response.raise_for_status()

            result = response.json()
            time.sleep(config.REQUEST_CONFIG['request_delay'])
            return result

        except Exception as e:
            print(f"搜索公告失败: {stock_code}, {start_date}~{end_date}, 错误: {str(e)}")
            raise

    def search_all_announcements(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        搜索所有公告（自动翻页）

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            keywords: 关键词列表

        Returns:
            公告列表
        """
        all_announcements = []
        page_num = 1
        page_size = 50

        while True:
            try:
                result = self.search_announcements(
                    stock_code, start_date, end_date, keywords, page_num, page_size
                )

                if 'announcements' not in result or not result['announcements']:
                    break

                announcements = result['announcements']
                all_announcements.extend(announcements)

                # 检查是否还有更多页
                total_announcements = result.get('totalpages', 0) * page_size
                if len(all_announcements) >= total_announcements:
                    break

                page_num += 1

            except Exception as e:
                print(f"翻页搜索失败: {stock_code}, 页码: {page_num}, 错误: {str(e)}")
                break

        return all_announcements

    def get_ma_related_announcements(
        self,
        stock_code: str,
        first_date: str,
        latest_date: Optional[str] = None,
        finish_date: Optional[str] = None
    ) -> List[Dict]:
        """
        获取并购相关的所有公告（包括首次公告和后续公告）

        Args:
            stock_code: 股票代码
            first_date: 首次公告日期
            latest_date: 最新公告日期（可选）
            finish_date: 完成公告日期（可选）

        Returns:
            公告列表，按时间排序
        """
        # 确定搜索的日期范围
        # 为了不遗漏公告，在首次公告日期前后扩展搜索范围
        start_date = self._extend_date(first_date, -config.DATE_RANGE_EXTEND_DAYS)

        # 结束日期取最新日期或完成日期中较晚的
        end_date = first_date
        if latest_date:
            end_date = max(end_date, latest_date) if end_date else latest_date
        if finish_date:
            end_date = max(end_date, finish_date) if end_date else finish_date

        # 向后扩展一些天数，确保不遗漏
        end_date = self._extend_date(end_date, config.DATE_RANGE_EXTEND_DAYS)

        # 搜索并购相关公告
        announcements = self.search_all_announcements(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            keywords=config.MA_KEYWORDS
        )

        # 过滤和排序
        filtered = self._filter_ma_announcements(announcements, first_date)
        filtered.sort(key=lambda x: x.get('announcementTime', ''), reverse=False)

        return filtered

    def _filter_ma_announcements(
        self,
        announcements: List[Dict],
        reference_date: str
    ) -> List[Dict]:
        """
        过滤并购相关公告

        Args:
            announcements: 原始公告列表
            reference_date: 参考日期（首次公告日期）

        Returns:
            过滤后的公告列表
        """
        filtered = []

        for ann in announcements:
            title = ann.get('announcementTitle', '')
            ann_time = ann.get('announcementTime', '')

            # 基本过滤：标题必须包含并购关键词
            if not any(keyword in title for keyword in config.MA_KEYWORDS):
                continue

            # 过滤明显无关的公告类型
            if self._is_invalid_announcement(title):
                continue

            # 如果是更正/补充公告，需要检查是否包含实质内容
            if any(word in title for word in ['更正', '补充', '修订']):
                # 保留包含关键并购词汇的更正/补充公告
                if not any(word in title for word in ['重组', '收购', '资产', '股权', '合并']):
                    continue

            filtered.append(ann)

        return filtered

    def _is_invalid_announcement(self, title: str) -> bool:
        """
        判断是否为无效公告

        Args:
            title: 公告标题

        Returns:
            True表示无效
        """
        # 纯程序性公告，不包含实质内容
        invalid_only_patterns = [
            '仅', '停牌公告', '复牌公告',
            '风险提示公告',
            '致歉公告',
        ]

        for pattern in invalid_only_patterns:
            if pattern in title and title.count('公告') == 1:
                return True

        return False

    @retry(tries=3, delay=2)
    def download_announcement(
        self,
        announcement_id: str,
        adj_announcement_id: str,
        save_path: str
    ) -> bool:
        """
        下载公告PDF文件

        Args:
            announcement_id: 公告ID
            adj_announcement_id: 调整后的公告ID
            save_path: 保存路径

        Returns:
            下载是否成功
        """
        try:
            # 构建下载URL
            # 巨潮资讯的URL格式: http://static.cninfo.com.cn/finalpage/日期/公告ID.PDF
            url = f"{config.CNINFO_API['download_url']}{adj_announcement_id}"

            response = self.session.get(
                url,
                timeout=config.REQUEST_CONFIG['timeout'],
                stream=True
            )
            response.raise_for_status()

            # 保存文件
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            time.sleep(config.REQUEST_CONFIG['request_delay'])
            return True

        except Exception as e:
            print(f"下载公告失败: {announcement_id}, 错误: {str(e)}")
            return False

    @staticmethod
    def _extend_date(date_str: str, days: int) -> str:
        """
        日期扩展

        Args:
            date_str: 日期字符串 YYYY-MM-DD
            days: 扩展天数（正数向后，负数向前）

        Returns:
            扩展后的日期字符串
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            new_date = date_obj + timedelta(days=days)
            return new_date.strftime('%Y-%m-%d')
        except:
            return date_str

    def get_announcement_detail(self, announcement_id: str) -> Optional[Dict]:
        """
        获取公告详情

        Args:
            announcement_id: 公告ID

        Returns:
            公告详情字典
        """
        try:
            url = f"{config.CNINFO_API['announcement_url']}?announceId={announcement_id}"
            response = self.session.get(url, timeout=config.REQUEST_CONFIG['timeout'])
            response.raise_for_status()
            time.sleep(config.REQUEST_CONFIG['request_delay'])
            return response.json()
        except Exception as e:
            print(f"获取公告详情失败: {announcement_id}, 错误: {str(e)}")
            return None

    @staticmethod
    def format_announcement_info(ann: Dict) -> Dict:
        """
        格式化公告信息

        Args:
            ann: 原始公告信息

        Returns:
            格式化后的公告信息
        """
        return {
            'announcement_id': ann.get('announcementId', ''),
            'adj_announcement_id': ann.get('adjunctUrl', ''),
            'title': ann.get('announcementTitle', ''),
            'time': ann.get('announcementTime', ''),
            'stock_code': ann.get('secCode', ''),
            'stock_name': ann.get('secName', ''),
            'type': ann.get('announcementType', ''),
        }
