"""
LangGraph节点实现
提供代码生成、验证和检查功能
"""

import logging
import ast
import re
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .models import GraphState
from .utils import extract_code_from_response, parse_check_result

logger = logging.getLogger(__name__)


class CodeAssistant:
    """代码助手核心类"""
    
    def __init__(self, config):
        self.config = config
        self.llm = ChatTongyi(
            model_name=config.model_name,
            temperature=config.temperature
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """构建LangGraph工作流"""
        builder = StateGraph(GraphState)
        
        builder.add_node("analyze_request", self._analyze_request)
        builder.add_node("generate", self._generate_code)
        builder.add_node("validate", self._validate_code)
        builder.add_node("check", self._check_code)
        builder.add_node("reflect", self._reflect_code)
        
        builder.add_edge(START, "analyze_request")
        builder.add_edge("analyze_request", "generate")
        builder.add_edge("generate", "validate")
        
        builder.add_conditional_edges(
            "validate",
            self._decide_after_validation,
            {
                "check": "check",
                "regenerate": "generate",
                "end": END
            }
        )
        
        builder.add_conditional_edges(
            "check",
            self._decide_next_step,
            {
                "end": END,
                "reflect": "reflect",
                "regenerate": "generate",
            }
        )
        
        builder.add_edge("reflect", "generate")
        
        return builder.compile()
    
    def _analyze_request(self, state: GraphState) -> GraphState:
        """分析用户需求"""
        user_request = state["user_request"]
        
        logger.info("开始分析用户需求")
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="分析用户需求，评估复杂度和技术要求。返回简洁的分析结果。"),
            HumanMessage(content=f"用户需求：{user_request}")
        ])
        
        response = self.llm.invoke(analysis_prompt.format_messages())
        
        # 评估复杂度
        complexity = "medium"
        if any(word in user_request.lower() for word in ["高性能", "并发", "缓存", "优化"]):
            complexity = "high"
        elif any(word in user_request.lower() for word in ["简单", "基本", "入门"]):
            complexity = "low"
        
        analysis = {
            "complexity": complexity,
            "features": ["basic"],
            "risks": []
        }
        
        logger.info(f"需求分析完成：复杂度={complexity}")
        
        return {
            **state,
            "analysis": analysis,
            "messages": state.get("messages", []) + [response]
        }
    
    def _generate_code(self, state: GraphState) -> GraphState:
        """生成代码"""
        user_request = state["user_request"]
        reflection = state.get("reflection", "")
        analysis = state.get("analysis", {})
        
        logger.info("开始生成代码")
        
        complexity = analysis.get("complexity", "medium")
        
        if reflection:
            user_prompt = f"""
            根据需求和反思意见，重新生成改进的代码：
            
            需求：{user_request}
            反思意见：{reflection}
            
            请生成高质量的Python代码。
            """
        else:
            user_prompt = f"""
            根据需求生成Python代码：
            
            需求：{user_request}
            复杂度：{complexity}
            
            请生成高质量的Python代码。
            """
        
        system_prompts = {
            "low": "生成简洁清晰的Python代码，注重可读性。",
            "medium": "生成高质量可维护的Python代码，包含错误处理。",
            "high": "生成企业级Python代码，考虑性能、安全性和最佳实践。"
        }
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompts.get(complexity, system_prompts["medium"])),
            HumanMessage(content=user_prompt)
        ])
        
        response = self.llm.invoke(prompt.format_messages())
        generated_code = extract_code_from_response(response.content)
        
        logger.info(f"代码生成完成，长度: {len(generated_code)} 字符")
        
        return {
            **state,
            "generated_code": generated_code,
            "messages": state.get("messages", []) + [response]
        }
    
    def _validate_code(self, state: GraphState) -> GraphState:
        """代码验证"""
        generated_code = state["generated_code"]
        
        logger.info("开始验证代码")
        
        # 语法检查
        syntax_score = self._check_syntax(generated_code)
        
        # 复杂度检查
        complexity_score = self._check_complexity(generated_code)
        
        # 风格检查
        style_score = self._check_style(generated_code)
        
        # 计算总体验证分数
        validation_score = int((syntax_score + complexity_score + style_score) / 3)
        
        validation_results = {
            "syntax": {"score": syntax_score},
            "complexity": {"score": complexity_score},
            "style": {"score": style_score}
        }
        
        logger.info(f"代码验证完成，验证分数: {validation_score}")
        
        return {
            **state,
            "validation_results": validation_results,
            "validation_score": validation_score
        }
    
    def _check_code(self, state: GraphState) -> GraphState:
        """代码检查"""
        generated_code = state["generated_code"]
        validation_score = state.get("validation_score", 0)
        
        logger.info("开始深度代码检查")
        
        check_prompt = f"""
        请检查以下Python代码的质量：
        
        代码：
        ```python
        {generated_code}
        ```
        
        请从功能完整性、代码质量、性能等维度评估，给出1-10分的评分。
        
        返回格式：
        评分：X分
        问题：[具体问题列表]
        建议：[改进建议]
        """
        
        response = self.llm.invoke([HumanMessage(content=check_prompt)])
        check_result = parse_check_result(response.content)
        
        # 结合验证分数调整最终评分
        llm_score = check_result.get('score', 0)
        final_score = int(llm_score * 0.7 + validation_score * 0.3)
        check_result['score'] = final_score
        
        logger.info(f"代码检查完成，最终评分: {final_score}/10")
        
        return {
            **state,
            "check_result": check_result,
            "messages": state.get("messages", []) + [response]
        }
    
    def _reflect_code(self, state: GraphState) -> GraphState:
        """代码反思"""
        generated_code = state["generated_code"]
        check_result = state["check_result"]
        
        logger.info("开始代码反思")
        
        reflect_prompt = f"""
        基于检查结果对代码进行反思：
        
        原代码：
        ```python
        {generated_code}
        ```
        
        检查结果：{check_result}
        
        请提供具体的改进建议。
        """
        
        response = self.llm.invoke([HumanMessage(content=reflect_prompt)])
        
        logger.info("代码反思完成")
        
        return {
            **state,
            "reflection": response.content,
            "current_retry": state.get("current_retry", 0) + 1,
            "messages": state.get("messages", []) + [response]
        }
    
    def _check_syntax(self, code: str) -> int:
        """语法检查"""
        try:
            ast.parse(code)
            return 10
        except SyntaxError:
            return 0
    
    def _check_complexity(self, code: str) -> int:
        """复杂度检查"""
        lines = [line for line in code.split('\n') if line.strip()]
        complexity_score = max(0, 10 - len(lines) // 10)
        return complexity_score
    
    def _check_style(self, code: str) -> int:
        """代码风格检查"""
        score = 10
        
        if not re.search(r'def \w+\(.*\):', code):
            score -= 2
        
        if not re.search(r'""".*"""', code, re.DOTALL):
            score -= 1
        
        return max(0, score)
    
    def _decide_after_validation(self, state: GraphState) -> Literal["check", "regenerate", "end"]:
        """验证后的决策"""
        validation_score = state.get("validation_score", 0)
        
        if validation_score < 3:
            return "regenerate"
        elif validation_score >= 8:
            return "end"
        else:
            return "check"
    
    def _decide_next_step(self, state: GraphState) -> Literal["end", "reflect", "regenerate"]:
        """决定下一步操作"""
        check_result = state.get("check_result", {})
        current_retry = state.get("current_retry", 0)
        max_retry = state.get("max_retry", 3)
        
        score = check_result.get("score", 0)
        
        logger.info(f"决策: 当前评分={score}, 重试次数={current_retry}/{max_retry}")
        
        if score >= self.config.quality_threshold:
            logger.info("代码质量达标，结束流程")
            return "end"
        
        if current_retry >= max_retry:
            logger.info("达到最大重试次数，结束流程")
            return "end"
        
        if score < self.config.reflection_threshold:
            logger.info("分数较低，进行反思")
            return "reflect"
        
        logger.info("直接重新生成代码")
        return "regenerate"