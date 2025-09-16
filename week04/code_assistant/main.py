"""
代码改进助手 - 主程序
基于LangGraph实现的代码生成和改进流水线
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from core.models import GraphState, CodeImprovementResult
from core.nodes import CodeAssistant
from core.utils import format_code_preview, get_score_level
from config import Config

# 可视化支持
try:
    from IPython.display import Image, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodeImprovementPipeline:
    """代码改进流水线"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.assistant = CodeAssistant(self.config)
        
        logging.getLogger().setLevel(getattr(logging, self.config.log_level))
    
    def visualize_graph(self):
        """可视化工作流图"""
        if IPYTHON_AVAILABLE:
            try:
                display(Image(self.assistant.graph.get_graph().draw_mermaid_png()))
                print("工作流图已显示")
            except Exception as e:
                print(f"可视化失败: {e}")
        else:
            print("需要在Jupyter环境中运行可视化功能")
        
    def run(self, user_request: str, verbose: Optional[bool] = None) -> CodeImprovementResult:
        """运行代码改进流程"""
        
        verbose = verbose if verbose is not None else self.config.verbose
        
        if verbose:
            self._print_header(user_request)
        
        initial_state = GraphState(
            user_request=user_request,
            generated_code="",
            check_result={},
            reflection="",
            max_retry=self.config.max_retry,
            current_retry=0,
            reflect=self.config.enable_reflection,
            messages=[]
        )
        
        final_state = None
        step_count = 0
        
        logger.info(f"开始执行代码改进流程: {user_request}")
        
        for step_output in self.assistant.graph.stream(initial_state):
            step_count += 1
            if verbose:
                self._display_step(step_count, step_output)
            
            for state in step_output.values():
                final_state = state
                
        logger.info("代码改进流程执行完成")
        
        if verbose:
            self._print_results(final_state)
            
        return self._format_result(final_state)
    
    def _print_header(self, request: str):
        """打印头部信息"""
        print("代码改进助手")
        print(f"需求: {request}")
        print("=" * 60)
    
    def _display_step(self, step: int, step_output: Dict):
        """显示步骤信息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"\n[{timestamp}] 步骤 {step}")
        
        for node_name, state in step_output.items():
            self._display_node_result(node_name, state)
    
    def _display_node_result(self, node_name: str, state: GraphState):
        """显示节点结果"""
        if node_name == "analyze_request":
            analysis = state.get("analysis", {})
            complexity = analysis.get("complexity", "unknown")
            print(f"需求分析: 复杂度={complexity}")
            
        elif node_name == "generate":
            code = state.get("generated_code", "")
            retry = state.get("current_retry", 0)
            
            print(f"代码生成: {len(code)} 字符 (第{retry + 1}次)")
            
            if self.config.show_code_preview and code:
                preview = format_code_preview(code, self.config.max_preview_lines)
                print("代码预览:")
                for i, line in enumerate(preview.split('\n'), 1):
                    print(f"  {i:2d}: {line}")
        
        elif node_name == "validate":
            validation_score = state.get("validation_score", 0)
            validation_results = state.get("validation_results", {})
            
            # 强制重新计算验证分数以确保正确显示
            if validation_results:
                scores = []
                for result in validation_results.values():
                    if isinstance(result, dict) and 'score' in result:
                        scores.append(result['score'])
                if scores:
                    calculated_score = int(sum(scores) / len(scores))
                    # 使用计算出的分数，如果状态中的分数为0或不一致
                    if validation_score == 0 or validation_score != calculated_score:
                        validation_score = calculated_score
            
            print(f"代码验证: {validation_score}/10 分")
            
            for check_type, result in validation_results.items():
                if isinstance(result, dict) and 'score' in result:
                    print(f"  - {check_type}: {result['score']}/10")
            
        elif node_name == "check":
            result = state.get("check_result", {})
            score = result.get("score", 0)
            level = get_score_level(score)
            
            print(f"质量检查: {score}/10 分 ({level})")
            
            issues = result.get("issues", [])
            if issues:
                print(f"  发现问题: {len(issues)} 个")
                for i, issue in enumerate(issues[:2], 1):
                    print(f"  {i}. {issue[:60]}{'...' if len(issue) > 60 else ''}")
                if len(issues) > 2:
                    print(f"  ... (还有{len(issues) - 2}个问题)")
            
        elif node_name == "reflect":
            retry = state.get("current_retry", 0)
            reflection = state.get("reflection", "")
            
            print(f"代码反思: 第{retry}次重试")
            if reflection:
                preview = reflection[:80] + "..." if len(reflection) > 80 else reflection
                print(f"反思内容: {preview}")
    
    def _print_results(self, state: GraphState):
        """打印最终结果"""
        print("\n" + "=" * 60)
        print("执行完成")
        
        check_result = state.get("check_result", {})
        score = check_result.get("score", 0)
        level = get_score_level(score)
        retry_count = state.get("current_retry", 0)
        validation_score = state.get("validation_score", 0)
        
        print(f"最终评分: {score}/10 分 ({level})")
        print(f"验证评分: {validation_score}/10 分")
        print(f"重试次数: {retry_count}")
        
        final_code = state.get("generated_code", "")
        if final_code:
            print(f"\n最终代码:")
            print("```python")
            print(final_code)
            print("```")
        
    def _format_result(self, state: GraphState) -> CodeImprovementResult:
        """格式化返回结果"""
        check_result = state.get("check_result", {})
        
        return CodeImprovementResult(
            final_code=state.get("generated_code", ""),
            check_result=check_result,
            retry_count=state.get("current_retry", 0),
            success=check_result.get("score", 0) >= self.config.quality_threshold
        )


def demo_workflow():
    """演示工作流"""
    
    print("LangGraph 代码改进工作流演示")
    print("=" * 50)
    
    config = Config(
        max_retry=2,
        enable_reflection=True,
        quality_threshold=7,
        reflection_threshold=5,
        verbose=True,
        show_code_preview=True,
        max_preview_lines=3
    )
    
    test_cases = [
        "创建一个高性能的斐波那契数列计算器，支持缓存和并发",
        "设计一个简单的计算器类，支持基本四则运算"
    ]
    
    pipeline = CodeImprovementPipeline(config)
    
    # 显示工作流可视化
    print("工作流可视化:")
    pipeline.visualize_graph()
    
    for i, request in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {request}")
        print("-" * 50)
        
        result = pipeline.run(request)
        
        print(f"\n结果统计:")
        print(f"  - 评分: {result['check_result'].get('score', 0)}/10")
        print(f"  - 重试: {result['retry_count']} 次")
        print(f"  - 成功: {'是' if result['success'] else '否'}")
        
        if i < len(test_cases):
            print("\n" + "="*50)


if __name__ == "__main__":
    demo_workflow()