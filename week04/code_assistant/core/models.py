"""
数据模型定义
"""

from typing import Dict, List, Any
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """图状态定义"""
    user_request: str          # 用户需求
    generated_code: str        # 生成的代码
    check_result: Dict[str, Any]  # 检查结果
    validation_score: int      # 验证分数
    validation_results: Dict[str, Any]  # 验证结果详情
    analysis: Dict[str, Any]   # 需求分析结果
    reflection: str            # 反思内容
    max_retry: int            # 最大重试次数
    current_retry: int        # 当前重试次数
    reflect: bool             # 是否需要反思
    messages: List[Any]       # 消息历史


class CheckResult(TypedDict):
    """代码检查结果"""
    score: int                # 评分 (1-10)
    issues: List[str]         # 问题列表
    suggestions: List[str]    # 建议列表


class CodeImprovementResult(TypedDict):
    """代码改进结果"""
    final_code: str           # 最终代码
    check_result: CheckResult # 检查结果
    retry_count: int          # 重试次数
    success: bool             # 是否成功