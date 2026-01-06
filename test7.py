from typing import List, Optional, Dict, Any, Tuple
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass
from enum import Enum
import logging
from difflib import SequenceMatcher

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParseMode(Enum):
    """语义解析模式"""
    CLEAN_HTML = "clean_html"      # 清理HTML标签
    MARKDOWN = "markdown"          # 转换为Markdown
    PLAIN_TEXT = "plain_text"      # 纯文本提取

@dataclass
class HTMLSegment:
    """HTML片段数据类"""
    html_content: str           # HTML片段内容
    start_index: int            # 在原始HTML中的起始位置
    end_index: int              # 在原始HTML中的结束位置
    tag_name: str = ""          # 标签名（如果有）
    parsed_content: str = ""    # 解析后的内容
    content_length: int = 0     # 解析后内容的长度
    is_merged: bool = False     # 是否由多个片段合并而成
    overlap_before: int = 0     # 与前一个片段的重叠字符数
    overlap_after: int = 0      # 与后一个片段的重叠字符数
    source_html: str = ""       # 存储原始HTML，用于验证

class PositionValidator:
    """位置验证器 - 修复版"""
    
    @staticmethod
    def normalize_html(html: str) -> str:
        """标准化HTML，去除不必要的差异"""
        # 合并连续的空白字符
        html = re.sub(r'\s+', ' ', html)
        # 移除标签内的多余空格
        html = re.sub(r'\s*([<>])\s*', r'\1', html)
        # 统一引号
        html = html.replace('"', "'")
        # 移除注释
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        return html.strip()
    
    @staticmethod
    def find_best_match(substring: str, full_text: str, start_pos: int = 0) -> Tuple[int, int, float]:
        """
        在完整文本中查找子串的最佳匹配位置
        
        Args:
            substring: 要查找的子串
            full_text: 完整文本
            start_pos: 开始搜索的位置
            
        Returns:
            (起始位置, 结束位置, 相似度)
        """
        if not substring or not full_text:
            return -1, -1, 0.0
        
        substring_norm = PositionValidator.normalize_html(substring)
        full_text_norm = PositionValidator.normalize_html(full_text)
        
        best_start = -1
        best_end = -1
        best_similarity = 0.0
        substring_len = len(substring_norm)
        
        # 如果子串太短，直接查找
        if substring_len < 50:
            pos = full_text_norm.find(substring_norm, start_pos)
            if pos != -1:
                return pos, pos + substring_len, 1.0
        
        # 使用滑动窗口查找最佳匹配
        search_window = 5000  # 搜索窗口大小
        max_search_start = min(start_pos + search_window, len(full_text_norm) - substring_len)
        
        for i in range(start_pos, max_search_start - substring_len + 1, 10):  # 步长10，加速搜索
            window = full_text_norm[i:i + substring_len]
            
            # 计算相似度
            matcher = SequenceMatcher(None, substring_norm, window)
            similarity = matcher.ratio()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_start = i
                best_end = i + substring_len
        
        return best_start, best_end, best_similarity
    
    @staticmethod
    def validate_segment_position(segment: HTMLSegment, original_html: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        验证片段位置是否正确
        
        Args:
            segment: HTML片段
            original_html: 原始HTML文本
            
        Returns:
            (是否验证通过, 错误信息, 详细数据)
        """
        details = {
            "claimed_start": segment.start_index,
            "claimed_end": segment.end_index,
            "claimed_length": len(segment.html_content),
            "similarity": 0.0,
            "actual_start": -1,
            "actual_end": -1
        }
        
        try:
            # 1. 检查索引范围是否合法
            if segment.start_index < 0 or segment.end_index > len(original_html):
                return False, f"索引越界: [{segment.start_index}, {segment.end_index}]，总长度: {len(original_html)}", details
            
            # 2. 提取原始位置的HTML
            extracted_html = original_html[segment.start_index:segment.end_index]
            
            if not extracted_html.strip():
                return False, "提取的HTML为空", details
            
            # 3. 标准化比较
            segment_norm = PositionValidator.normalize_html(segment.html_content)
            extracted_norm = PositionValidator.normalize_html(extracted_html)
            
            # 计算相似度
            matcher = SequenceMatcher(None, segment_norm, extracted_norm)
            similarity = matcher.ratio()
            details["similarity"] = similarity
            
            # 4. 查找最佳匹配位置
            best_start, best_end, best_similarity = PositionValidator.find_best_match(
                segment.html_content, original_html, max(0, segment.start_index - 100)
            )
            
            details["actual_start"] = best_start
            details["actual_end"] = best_end
            details["best_similarity"] = best_similarity
            
            # 5. 验证标准
            if similarity < 0.95:  # 相似度阈值提高到95%
                # 尝试查找更好的匹配
                if best_similarity > similarity and best_start != -1:
                    return False, f"位置可能不准确。声明位置相似度: {similarity:.2%}，最佳匹配位置相似度: {best_similarity:.2%}，建议位置: [{best_start}, {best_end}]", details
                else:
                    return False, f"HTML内容不匹配，相似度: {similarity:.2%}", details
            
            # 6. 验证解析内容是否可以从HTML中提取
            if segment.parsed_content:
                # 从声明位置的HTML提取文本
                soup = BeautifulSoup(extracted_html, 'html.parser')
                extracted_text = soup.get_text(separator=' ', strip=True)
                extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
                
                # 标准化解析内容
                parsed_norm = re.sub(r'\s+', ' ', segment.parsed_content.strip())
                
                if extracted_text and parsed_norm:
                    text_matcher = SequenceMatcher(None, extracted_text, parsed_norm)
                    text_similarity = text_matcher.ratio()
                    
                    if text_similarity < 0.9:  # 文本相似度阈值
                        return False, f"解析内容不匹配，文本相似度: {text_similarity:.2%}", details
            
            return True, f"验证通过，相似度: {similarity:.2%}", details
            
        except Exception as e:
            return False, f"验证过程出错: {str(e)}", details
    
    @staticmethod
    def validate_all_positions(segments: List[HTMLSegment], original_html: str) -> Dict[int, Dict[str, Any]]:
        """
        验证所有片段的位置
        
        Returns:
            验证结果字典：{片段索引: {valid: bool, message: str, details: dict}}
        """
        results = {}
        
        for i, segment in enumerate(segments):
            is_valid, message, details = PositionValidator.validate_segment_position(segment, original_html)
            results[i] = {
                'valid': is_valid,
                'message': message,
                'details': details
            }
            
            if not is_valid:
                logger.warning(f"片段 {i} 验证失败: {message}")
        
        # 检查片段间是否有重叠或间隙
        for i in range(len(segments) - 1):
            current_end = segments[i].end_index
            next_start = segments[i + 1].start_index
            
            if current_end > next_start:
                results[i]['valid'] = False
                results[i]['message'] = f"与片段 {i+1} 重叠: [{current_end} > {next_start}]"
                results[i]['details']['has_overlap'] = True
                results[i]['details']['overlap_with'] = i + 1
            elif current_end < next_start:
                # 检查间隙内容
                gap_content = original_html[current_end:next_start]
                if gap_content.strip():
                    results[i]['valid'] = False
                    results[i]['message'] = f"与片段 {i+1} 有间隙，包含内容: {gap_content[:50]}..."
                    results[i]['details']['has_gap'] = True
                    results[i]['details']['gap_content_preview'] = gap_content[:100]
        
        return results

class SemanticHTMLSplitter:
    """
    语义感知的HTML分割器 - 修复位置记录问题
    
    核心改进：
    1. 直接记录原始HTML位置，不依赖BeautifulSoup的str()方法
    2. 更准确的位置查找算法
    3. 增强的验证逻辑
    """
    
    def __init__(
        self,
        min_content_length: int = 100,
        max_content_length: int = 2000,
        overlap: int = 200,
        parse_mode: ParseMode = ParseMode.CLEAN_HTML,
        split_tags: List[str] = None,
        keep_html_structure: bool = True,
        strip_whitespace: bool = True,
        validate_positions: bool = True,
        position_tolerance: int = 100  # 位置容忍度，允许位置有一定误差
    ):
        """
        初始化分割器
        
        Args:
            position_tolerance: 位置查找时的容忍度，允许在原始HTML中查找时的误差范围
        """
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.overlap = overlap
        self.parse_mode = parse_mode
        self.keep_html_structure = keep_html_structure
        self.strip_whitespace = strip_whitespace
        self.validate_positions = validate_positions
        self.position_tolerance = position_tolerance
        
        # 默认分割标签：块级元素
        if split_tags is None:
            self.split_tags = [
                'p', 'div', 'section', 'article', 
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'li', 'td', 'th', 'blockquote',
                'header', 'footer', 'nav', 'aside'
            ]
        else:
            self.split_tags = split_tags
        
        # 初始化HTML语义分割器
        self.semantic_splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=self.split_tags
        )
        
        # 统计信息
        self.stats = {
            'initial_segments': 0,
            'final_chunks': 0,
            'merged_segments': 0,
            'skipped_empty': 0,
            'overlap_added': 0,
            'validation_errors': 0,
            'position_adjustments': 0
        }
        
        # 验证结果
        self.validation_results = {}
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """从HTML中提取文本内容"""
        if not html_content or html_content.isspace():
            return ""
        
        if self.parse_mode == ParseMode.CLEAN_HTML:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=' ', strip=self.strip_whitespace)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        elif self.parse_mode == ParseMode.PLAIN_TEXT:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for br in soup.find_all('br'):
                br.replace_with('\n')
            
            for tag in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                for elem in soup.find_all(tag):
                    if elem.text.strip():
                        elem.insert_after('\n\n')
            
            text = soup.get_text(separator='\n', strip=self.strip_whitespace)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return '\n'.join(lines)
            
        elif self.parse_mode == ParseMode.MARKDOWN:
            return self._html_to_markdown(html_content)
        else:
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
    
    def _html_to_markdown(self, html_content: str) -> str:
        """将HTML转换为Markdown格式"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for i in range(1, 7):
            for h in soup.find_all(f'h{i}'):
                h.string = f'{"#" * i} {h.get_text()}'
        
        for bold in soup.find_all(['b', 'strong']):
            bold.string = f'**{bold.get_text()}**'
        
        for italic in soup.find_all(['i', 'em']):
            italic.string = f'*{italic.get_text()}*'
        
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href:
                link.string = f'[{link.get_text()}]({href})'
        
        for li in soup.find_all('li'):
            li.string = f'* {li.get_text()}'
        
        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _find_element_positions(self, element, original_html: str, search_start: int = 0) -> Tuple[int, int, str]:
        """
        在原始HTML中查找元素的确切位置
        
        Args:
            element: BeautifulSoup元素
            original_html: 原始HTML文本
            search_start: 开始搜索的位置
            
        Returns:
            (起始位置, 结束位置, 实际HTML内容)
        """
        # 获取元素的HTML表示
        element_html = str(element)
        
        # 标准化element_html和搜索区域
        element_norm = PositionValidator.normalize_html(element_html)
        
        # 在原始HTML中查找
        search_area = original_html[search_start:]
        search_norm = PositionValidator.normalize_html(search_area)
        
        # 查找最佳匹配
        best_start, best_end, best_similarity = PositionValidator.find_best_match(
            element_norm, search_norm
        )
        
        if best_start != -1 and best_similarity > 0.9:  # 相似度阈值
            # 将位置映射回原始HTML
            actual_start = search_start + best_start
            actual_end = search_start + best_end
            
            # 获取实际的HTML内容（非标准化版本）
            actual_html = original_html[actual_start:actual_end]
            
            return actual_start, actual_end, actual_html
        
        # 如果找不到高相似度匹配，使用原始方法
        pattern = re.escape(element_html)
        match = re.search(pattern, original_html[search_start:], re.DOTALL)
        
        if match:
            start = search_start + match.start()
            end = search_start + match.end()
            actual_html = original_html[start:end]
            return start, end, actual_html
        
        # 如果还是找不到，返回估计位置
        logger.warning(f"无法准确定位元素: {element.name}")
        return -1, -1, element_html
    
    def _split_html_by_structure(self, html_text: str) -> List[HTMLSegment]:
        """
        按HTML结构分割，直接记录原始位置
        
        Args:
            html_text: 原始HTML文本
            
        Returns:
            HTML片段列表，包含准确的位置信息
        """
        segments = []
        
        # 使用BeautifulSoup解析
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # 查找所有指定的标签
        for tag_name in self.split_tags:
            for element in soup.find_all(tag_name):
                # 跳过空元素
                if not element.get_text(strip=True):
                    continue
                
                # 查找元素在原始HTML中的位置
                start_idx = 0
                if segments:
                    start_idx = segments[-1].end_index
                
                start, end, actual_html = self._find_element_positions(element, html_text, start_idx)
                
                if start != -1 and end != -1:
                    segment = HTMLSegment(
                        html_content=actual_html,
                        start_index=start,
                        end_index=end,
                        tag_name=tag_name,
                        source_html=html_text  # 保存原始HTML用于验证
                    )
                    segments.append(segment)
        
        # 如果没有找到任何块级元素，将整个文档作为一个片段
        if not segments:
            segment = HTMLSegment(
                html_content=html_text,
                start_index=0,
                end_index=len(html_text),
                tag_name="document",
                source_html=html_text
            )
            segments.append(segment)
        
        # 按起始位置排序
        segments.sort(key=lambda x: x.start_index)
        
        # 处理重叠和间隙
        cleaned_segments = []
        current_end = 0
        
        for segment in segments:
            # 跳过重叠部分
            if segment.start_index < current_end:
                # 调整起始位置
                segment.start_index = current_end
                # 重新截取HTML内容
                if segment.start_index < segment.end_index:
                    segment.html_content = html_text[segment.start_index:segment.end_index]
                    self.stats['position_adjustments'] += 1
                else:
                    continue
            
            # 检查是否有间隙
            if segment.start_index > current_end:
                # 添加间隙内容作为独立的片段
                gap_content = html_text[current_end:segment.start_index]
                if gap_content.strip():  # 只添加非空内容
                    gap_segment = HTMLSegment(
                        html_content=gap_content,
                        start_index=current_end,
                        end_index=segment.start_index,
                        tag_name="text",
                        source_html=html_text
                    )
                    cleaned_segments.append(gap_segment)
            
            cleaned_segments.append(segment)
            current_end = segment.end_index
        
        # 添加最后的间隙（如果有）
        if current_end < len(html_text):
            gap_content = html_text[current_end:]
            if gap_content.strip():
                gap_segment = HTMLSegment(
                    html_content=gap_content,
                    start_index=current_end,
                    end_index=len(html_text),
                    tag_name="text",
                    source_html=html_text
                )
                cleaned_segments.append(gap_segment)
        
        return cleaned_segments
    
    def _parse_segments_content(self, segments: List[HTMLSegment]) -> None:
        """解析每个HTML片段的语义内容"""
        for segment in segments:
            parsed_content = self._extract_text_from_html(segment.html_content)
            segment.parsed_content = parsed_content
            segment.content_length = len(parsed_content)
    
    def _merge_small_segments(self, segments: List[HTMLSegment]) -> List[HTMLSegment]:
        """合并语义内容过小的片段"""
        if not segments or self.min_content_length <= 0:
            return segments
        
        merged_segments = []
        current_batch = []
        
        for segment in segments:
            # 跳过空内容片段
            if segment.content_length == 0:
                self.stats['skipped_empty'] += 1
                continue
            
            # 如果当前批次为空，开始新批次
            if not current_batch:
                current_batch = [segment]
                continue
            
            # 检查当前批次的总长度
            batch_content_length = sum(s.content_length for s in current_batch)
            
            # 如果当前片段很小，且加入后不会超过最大长度，则合并
            if (segment.content_length < self.min_content_length and 
                batch_content_length + segment.content_length < self.max_content_length):
                current_batch.append(segment)
            else:
                # 结束当前批次
                if len(current_batch) == 1:
                    merged_segments.append(current_batch[0])
                else:
                    # 合并多个片段
                    merged_html = "".join(s.html_content for s in current_batch)
                    
                    merged_segment = HTMLSegment(
                        html_content=merged_html,
                        start_index=current_batch[0].start_index,
                        end_index=current_batch[-1].end_index,
                        tag_name="merged",
                        parsed_content=self._extract_text_from_html(merged_html),
                        content_length=len(self._extract_text_from_html(merged_html)),
                        is_merged=True,
                        source_html=current_batch[0].source_html if current_batch else ""
                    )
                    merged_segments.append(merged_segment)
                    self.stats['merged_segments'] += len(current_batch) - 1
                
                # 开始新批次
                current_batch = [segment]
        
        # 处理最后一批
        if current_batch:
            if len(current_batch) == 1:
                merged_segments.append(current_batch[0])
            else:
                merged_html = "".join(s.html_content for s in current_batch)
                
                merged_segment = HTMLSegment(
                    html_content=merged_html,
                    start_index=current_batch[0].start_index,
                    end_index=current_batch[-1].end_index,
                    tag_name="merged",
                    parsed_content=self._extract_text_from_html(merged_html),
                    content_length=len(self._extract_text_from_html(merged_html)),
                    is_merged=True,
                    source_html=current_batch[0].source_html if current_batch else ""
                )
                merged_segments.append(merged_segment)
                self.stats['merged_segments'] += len(current_batch) - 1
        
        return merged_segments
    
    def _add_overlap_to_segments(self, segments: List[HTMLSegment]) -> List[HTMLSegment]:
        """为片段添加重叠内容"""
        if self.overlap <= 0 or len(segments) <= 1:
            return segments
        
        enhanced_segments = []
        
        for i, segment in enumerate(segments):
            # 复制当前片段
            enhanced_segment = HTMLSegment(
                html_content=segment.html_content,
                start_index=segment.start_index,
                end_index=segment.end_index,
                tag_name=segment.tag_name,
                parsed_content=segment.parsed_content,
                content_length=segment.content_length,
                is_merged=segment.is_merged,
                overlap_before=segment.overlap_before,
                overlap_after=segment.overlap_after,
                source_html=segment.source_html
            )
            
            # 添加前向重叠（从前一个片段获取）
            if i > 0:
                prev_segment = segments[i - 1]
                overlap_text = self._get_overlap_text(prev_segment.parsed_content, segment.parsed_content, True)
                if overlap_text:
                    enhanced_segment.parsed_content = overlap_text + enhanced_segment.parsed_content
                    enhanced_segment.overlap_before = len(overlap_text)
                    self.stats['overlap_added'] += 1
            
            # 添加后向重叠（从后一个片段获取）
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                overlap_text = self._get_overlap_text(segment.parsed_content, next_segment.parsed_content, False)
                if overlap_text:
                    enhanced_segment.parsed_content = enhanced_segment.parsed_content + overlap_text
                    enhanced_segment.overlap_after = len(overlap_text)
                    self.stats['overlap_added'] += 1
            
            # 更新内容长度
            enhanced_segment.content_length = len(enhanced_segment.parsed_content)
            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments
    
    def _get_overlap_text(self, source_content: str, target_content: str, from_previous: bool) -> str:
        """获取重叠文本"""
        if self.overlap <= 0:
            return ""
        
        if from_previous:
            # 从前一个片段的末尾获取
            if len(source_content) <= self.overlap:
                return source_content
            return source_content[-self.overlap:]
        else:
            # 从后一个片段的开头获取
            if len(target_content) <= self.overlap:
                return target_content
            return target_content[:self.overlap]
    
    def _validate_positions(self, segments: List[HTMLSegment], original_html: str) -> None:
        """验证所有片段的位置准确性"""
        if not self.validate_positions:
            return
        
        logger.info("开始位置验证...")
        self.validation_results = PositionValidator.validate_all_positions(segments, original_html)
        
        # 统计验证结果
        valid_count = sum(1 for result in self.validation_results.values() if result['valid'])
        invalid_count = len(self.validation_results) - valid_count
        
        self.stats['validation_errors'] = invalid_count
        
        if invalid_count > 0:
            logger.warning(f"位置验证完成: {valid_count}个通过，{invalid_count}个失败")
            
            # 如果有失败，尝试调整位置
            self._adjust_positions_based_on_validation(segments, original_html)
        else:
            logger.info(f"所有{valid_count}个片段位置验证通过")
    
    def _adjust_positions_based_on_validation(self, segments: List[HTMLSegment], original_html: str) -> None:
        """根据验证结果调整位置"""
        adjusted_count = 0
        
        for i, result in self.validation_results.items():
            if not result['valid'] and i < len(segments):
                segment = segments[i]
                details = result['details']
                
                # 如果验证器找到了更好的位置，并且相似度足够高
                if ('actual_start' in details and details['actual_start'] != -1 and 
                    'best_similarity' in details and details['best_similarity'] > 0.95):
                    
                    # 计算新的HTML内容
                    new_start = details['actual_start']
                    new_end = details['actual_end']
                    new_html = original_html[new_start:new_end]
                    
                    # 更新片段
                    segment.html_content = new_html
                    segment.start_index = new_start
                    segment.end_index = new_end
                    segment.parsed_content = self._extract_text_from_html(new_html)
                    segment.content_length = len(segment.parsed_content)
                    
                    adjusted_count += 1
                    logger.info(f"调整片段 {i} 的位置到 [{new_start}, {new_end}]")
        
        self.stats['position_adjustments'] += adjusted_count
        
        if adjusted_count > 0:
            logger.info(f"已调整 {adjusted_count} 个片段的位置")
    
    def split_text(self, html_text: str) -> List[Document]:
        """分割HTML文本为语义块"""
        # 重置统计信息
        self.stats = {
            'initial_segments': 0,
            'final_chunks': 0,
            'merged_segments': 0,
            'skipped_empty': 0,
            'overlap_added': 0,
            'validation_errors': 0,
            'position_adjustments': 0
        }
        
        # 步骤1: 按HTML结构分割
        segments = self._split_html_by_structure(html_text)
        self.stats['initial_segments'] = len(segments)
        
        # 步骤2: 解析每个片段的内容
        self._parse_segments_content(segments)
        
        # 步骤3: 合并小片段
        segments = self._merge_small_segments(segments)
        
        # 步骤4: 添加重叠内容
        if self.overlap > 0:
            segments = self._add_overlap_to_segments(segments)
        
        # 步骤5: 验证位置准确性
        self._validate_positions(segments, html_text)
        
        # 步骤6: 转换为Document对象
        documents = []
        for i, segment in enumerate(segments):
            # 跳过空内容
            if not segment.parsed_content or segment.content_length == 0:
                continue
            
            # 构建元数据
            metadata = {
                "start_index": segment.start_index,
                "end_index": segment.end_index,
                "content_length": segment.content_length,
                "html_length": len(segment.html_content),
                "tag_name": segment.tag_name,
                "is_merged": segment.is_merged,
                "parse_mode": self.parse_mode.value,
                "chunk_id": i,
                "total_chunks": len(segments),
                "overlap_before": segment.overlap_before,
                "overlap_after": segment.overlap_after
            }
            
            # 添加验证结果
            if i in self.validation_results:
                metadata["position_valid"] = self.validation_results[i]['valid']
                metadata["validation_message"] = self.validation_results[i]['message']
                metadata["validation_details"] = self.validation_results[i]['details']
            
            # 添加HTML预览
            html_preview = segment.html_content[:100]
            if len(segment.html_content) > 100:
                html_preview += "..."
            metadata["html_preview"] = html_preview
            
            # 提取文本预览（无重叠）
            text_preview = segment.parsed_content
            if segment.overlap_before > 0:
                text_preview = text_preview[segment.overlap_before:]
            if segment.overlap_after > 0:
                text_preview = text_preview[:-segment.overlap_after] if segment.overlap_after > 0 else text_preview
            metadata["text_preview"] = text_preview[:200] + "..." if len(text_preview) > 200 else text_preview
            
            # 创建Document
            doc = Document(
                page_content=segment.parsed_content,
                metadata=metadata
            )
            documents.append(doc)
        
        self.stats['final_chunks'] = len(documents)
        return documents
    
    def create_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
        """批量处理多个HTML文本"""
        all_documents = []
        for i, text in enumerate(texts):
            base_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            for doc in self.split_text(text):
                doc.metadata.update(base_metadata)
                all_documents.append(doc)
        return all_documents
    
    def print_stats(self):
        """打印统计信息"""
        print(f"初始HTML片段数: {self.stats['initial_segments']}")
        print(f"最终生成块数: {self.stats['final_chunks']}")
        print(f"合并片段数: {self.stats['merged_segments']}")
        print(f"跳过的空片段数: {self.stats['skipped_empty']}")
        if self.overlap > 0:
            print(f"添加重叠次数: {self.stats['overlap_added']}")
        if self.validate_positions:
            print(f"位置验证错误数: {self.stats['validation_errors']}")
            print(f"位置调整次数: {self.stats['position_adjustments']}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """获取验证结果摘要"""
        if not self.validation_results:
            return {"total": 0, "valid": 0, "invalid": 0}
        
        total = len(self.validation_results)
        valid = sum(1 for result in self.validation_results.values() if result['valid'])
        
        return {
            "total": total,
            "valid": valid,
            "invalid": total - valid,
            "error_rate": (total - valid) / total if total > 0 else 0
        }

class PositionAwareHTMLTextSplitter:
    """位置感知的HTML文本分割器（兼容层）"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_content_length: int = 100,
        parse_mode: ParseMode = ParseMode.CLEAN_HTML,
        validate_positions: bool = True,
        **kwargs
    ):
        self.splitter = SemanticHTMLSplitter(
            min_content_length=min_content_length,
            max_content_length=chunk_size,
            overlap=chunk_overlap,
            parse_mode=parse_mode,
            validate_positions=validate_positions,
            **kwargs
        )
    
    def split_text(self, html_text: str) -> List[Document]:
        return self.splitter.split_text(html_text)
    
    def create_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
        return self.splitter.create_documents(texts, metadatas)
    
    def print_stats(self):
        self.splitter.print_stats()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        return self.splitter.get_validation_summary()

if __name__ == "__main__":
    import time
    import logging

    # 禁用所有logging的warning
    logging.captureWarnings(False)

    # 或者只禁用特定logger的warning
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    start_time = time.time()
    # 读取HTML文件
    file_path = "tm2317741d1_f10.htm"
    
    import chardet
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        print(f"编码检测: {encoding} (置信度: {confidence:.2%})")
        
        encoding_map = {
            'ascii': 'utf-8',
            'Windows-1252': 'cp1252',
            'ISO-8859-1': 'iso-8859-1',
            'GB2312': 'gbk',
            'GBK': 'gbk',
            'Big5': 'big5',
            'SHIFT_JIS': 'shift_jis',
            'EUC-JP': 'euc-jp'
        }
        
        encoding_type = encoding_map.get(encoding, encoding)
    
    with open(file_path, 'r', encoding=encoding_type) as file:
        long_html = file.read()
    
    print(f"成功读取文件: {file_path}")
    print(f"文件大小: {len(long_html)} 字符")
    start_time = time.time()
    # 初始化分割器
    splitter = SemanticHTMLSplitter(
        min_content_length=3000,
        max_content_length=5000,
        overlap=200,
        parse_mode=ParseMode.CLEAN_HTML,
        split_tags=['h1', 'h2', 'p', 'div', 'section', 'li'],
        validate_positions=True
    )
    
    # 执行分割
    documents = splitter.split_text(long_html)
    end_time = time.time()
    

    
    # 输出结果
    print(f"\n生成 {len(documents)} 个块:")
    splitter.print_stats()
    
    # 打印验证摘要
    print("\n" + "=" * 60)
    print("位置验证摘要:")
    summary = splitter.get_validation_summary()
    print(f"总验证片段: {summary['total']}")
    print(f"通过: {summary['valid']}")
    print(f"失败: {summary['invalid']}")
    print(f"错误率: {summary['error_rate']:.2%}")
    
    # 显示前几个文档的详细信息
    print("\n文档详情:")
    for i, doc in enumerate(documents):
        print(f"\n--- 文档 #{i+1} ---")
        print(f"内容预览: {doc.metadata.get('text_preview', 'N/A')}")
        print(f"内容长度: {len(doc.page_content)} 字符")
        print(f"原始HTML位置: [{doc.metadata['start_index']} - {doc.metadata['end_index']}]")
        print(f"重叠: 前{doc.metadata.get('overlap_before', 0)}字, 后{doc.metadata.get('overlap_after', 0)}字")
        print(f"位置验证: {'通过' if doc.metadata.get('position_valid', True) else '失败'}")
        print('type: ', type(doc))
    
    print(type(documents))
    print(f"耗时: {end_time - start_time:.2f} 秒")