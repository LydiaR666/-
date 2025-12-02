"""
公告内容验证模块
验证公告是否包含企业动机、标的资产业务范围、并购目的、对收购方影响等关键信息
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from PyPDF2 import PdfReader
import config


class ContentValidator:
    """公告内容验证器"""

    def __init__(self):
        self.validation_keywords = config.CONTENT_VALIDATION
        self.threshold = config.VALIDATION_THRESHOLD

    def validate_pdf(self, pdf_path: str) -> Tuple[bool, Dict[str, any]]:
        """
        验证PDF公告内容

        Args:
            pdf_path: PDF文件路径

        Returns:
            (是否有效, 验证详情)
        """
        if not os.path.exists(pdf_path):
            return False, {'error': '文件不存在'}

        try:
            # 提取PDF文本
            text = self._extract_pdf_text(pdf_path)

            if not text or len(text) < 100:
                return False, {'error': 'PDF内容为空或过短', 'text_length': len(text)}

            # 验证内容
            validation_result = self._validate_content(text)

            # 判断是否通过验证
            is_valid = validation_result['matched_categories'] >= self.threshold

            return is_valid, validation_result

        except Exception as e:
            return False, {'error': f'验证失败: {str(e)}'}

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """
        提取PDF文本内容

        Args:
            pdf_path: PDF文件路径

        Returns:
            文本内容
        """
        try:
            reader = PdfReader(pdf_path)
            text = ''

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'

            # 清理文本
            text = self._clean_text(text)
            return text

        except Exception as e:
            print(f"提取PDF文本失败: {pdf_path}, 错误: {str(e)}")
            return ''

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        清理文本内容

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        return text.strip()

    def _validate_content(self, text: str) -> Dict[str, any]:
        """
        验证文本内容是否包含所需信息

        Args:
            text: 文本内容

        Returns:
            验证结果字典
        """
        result = {
            'matched_categories': 0,
            'categories': {},
            'total_matches': 0,
        }

        # 检查每个类别
        for category, keywords in self.validation_keywords.items():
            matches = self._find_keywords(text, keywords)
            category_matched = len(matches) > 0

            result['categories'][category] = {
                'matched': category_matched,
                'count': len(matches),
                'keywords': matches,
            }

            if category_matched:
                result['matched_categories'] += 1
                result['total_matches'] += len(matches)

        return result

    @staticmethod
    def _find_keywords(text: str, keywords: List[str]) -> List[str]:
        """
        在文本中查找关键词

        Args:
            text: 文本内容
            keywords: 关键词列表

        Returns:
            找到的关键词列表
        """
        found = []
        for keyword in keywords:
            if keyword in text:
                found.append(keyword)
        return found

    def get_validation_summary(self, validation_result: Dict) -> str:
        """
        获取验证结果摘要

        Args:
            validation_result: 验证结果

        Returns:
            摘要文本
        """
        if 'error' in validation_result:
            return f"验证错误: {validation_result['error']}"

        matched = validation_result['matched_categories']
        total = len(self.validation_keywords)

        summary = f"匹配 {matched}/{total} 个类别\n"

        for category, details in validation_result['categories'].items():
            category_name = self._get_category_name(category)
            status = "✓" if details['matched'] else "✗"
            count = details['count']
            keywords = ', '.join(details['keywords'][:3])  # 只显示前3个

            summary += f"  {status} {category_name}: {count}个关键词"
            if keywords:
                summary += f" ({keywords}...)"
            summary += "\n"

        return summary

    @staticmethod
    def _get_category_name(category: str) -> str:
        """获取类别中文名称"""
        names = {
            'motive': '企业动机',
            'target_business': '标的资产和业务',
            'purpose': '并购目的',
            'impact': '对收购方影响',
        }
        return names.get(category, category)

    def batch_validate(self, pdf_paths: List[str]) -> Dict[str, Tuple[bool, Dict]]:
        """
        批量验证PDF文件

        Args:
            pdf_paths: PDF文件路径列表

        Returns:
            {文件路径: (是否有效, 验证详情)}
        """
        results = {}
        for pdf_path in pdf_paths:
            is_valid, details = self.validate_pdf(pdf_path)
            results[pdf_path] = (is_valid, details)
        return results

    def extract_key_sections(self, pdf_path: str) -> Dict[str, str]:
        """
        提取公告中的关键章节

        Args:
            pdf_path: PDF文件路径

        Returns:
            关键章节内容字典
        """
        text = self._extract_pdf_text(pdf_path)
        if not text:
            return {}

        sections = {}

        # 定义章节标题模式
        section_patterns = {
            'motive': [
                r'(一|二|三|四|五|六|七|八|九|十|1|2|3|4|5|6|7|8|9).*?(交易背景|交易原因|交易动机|并购动机).*?(?=\n[一二三四五六七八九十\d]|\Z)',
            ],
            'target': [
                r'(一|二|三|四|五|六|七|八|九|十|1|2|3|4|5|6|7|8|9).*?(标的资产|标的公司|交易标的).*?(?=\n[一二三四五六七八九十\d]|\Z)',
            ],
            'purpose': [
                r'(一|二|三|四|五|六|七|八|九|十|1|2|3|4|5|6|7|8|9).*?(交易目的|本次交易目的|收购目的).*?(?=\n[一二三四五六七八九十\d]|\Z)',
            ],
            'impact': [
                r'(一|二|三|四|五|六|七|八|九|十|1|2|3|4|5|6|7|8|9).*?(对公司.*?影响|交易.*?影响|影响分析).*?(?=\n[一二三四五六七八九十\d]|\Z)',
            ],
        }

        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    section_text = match.group(0)
                    # 限制长度，避免提取过长
                    if len(section_text) > 2000:
                        section_text = section_text[:2000] + '...'
                    sections[section_name] = section_text
                    break

        return sections
