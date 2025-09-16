"""
工具函数
"""

import re
from typing import Dict, List, Any


def extract_code_from_response(response: str) -> str:
    """从LLM响应中提取代码"""
    # 查找Python代码块
    code_pattern = r'```python\n(.*?)\n```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # 查找通用代码块
    code_pattern = r'```\n(.*?)\n```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # 如果没找到代码块，返回整个响应
    return response.strip()


def parse_check_result(response: str) -> Dict[str, Any]:
    """解析代码检查结果"""
    result = {
        "score": 0,
        "issues": [],
        "suggestions": []
    }
    
    # 提取评分
    score_patterns = [
        r'评分[：:]\s*(\d+)\s*分',
        r'分数[：:]\s*(\d+)',
        r'得分[：:]\s*(\d+)'
    ]
    
    for pattern in score_patterns:
        score_match = re.search(pattern, response)
        if score_match:
            result["score"] = int(score_match.group(1))
            break
    
    # 提取问题和建议
    lines = response.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        # 识别章节
        if any(keyword in line for keyword in ['问题', '缺点', 'Issues']):
            current_section = 'issues'
            continue
        elif any(keyword in line for keyword in ['建议', '改进', 'Suggestions']):
            current_section = 'suggestions'
            continue
        
        # 添加内容到相应章节
        if line and current_section:
            # 清理行内容
            cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line)  # 移除序号
            cleaned_line = re.sub(r'^[-\*]\s*', '', cleaned_line)  # 移除列表符号
            
            if cleaned_line:
                if current_section == 'issues':
                    result["issues"].append(cleaned_line)
                elif current_section == 'suggestions':
                    result["suggestions"].append(cleaned_line)
    
    return result


def format_code_preview(code: str, max_lines: int = 5) -> str:
    """格式化代码预览"""
    if not code:
        return "无代码内容"
    
    lines = code.split('\n')
    if len(lines) <= max_lines:
        return code
    
    preview_lines = lines[:max_lines]
    return '\n'.join(preview_lines) + f'\n... (还有{len(lines) - max_lines}行)'


def get_score_level(score: int) -> str:
    """获取分数等级描述"""
    if score >= 8:
        return "优秀"
    elif score >= 6:
        return "良好"
    elif score >= 4:
        return "一般"
    else:
        return "需改进"