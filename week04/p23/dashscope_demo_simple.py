"""
简化版DashScope演示
模拟DashScope API调用，无需真实API密钥
"""

import json
from typing import Dict, List, Any
from langchain_core.runnables import RunnableParallel


class SimpleRuleEngine:
    """简化的规则引擎"""
    
    def __init__(self, keywords_config: Dict[str, List[str]]):
        self.keywords = keywords_config
    
    def predict(self, text: str) -> str:
        """基于关键词规则判断意图"""
        text = text.lower()
        for intent, words in self.keywords.items():
            if any(word in text for word in words):
                return intent
        return "unknown"


class MockDashScopeMLModel:
    """模拟DashScope机器学习模型"""
    
    def __init__(self):
        # 模拟通义千问的分类能力
        self.patterns = {
            "订单": "query_order",
            "查": "query_order",
            "退": "refund_request",
            "申请": "refund_request", 
            "发票": "issue_invoice",
            "报销": "issue_invoice",
            "物流": "logistics_inquiry",
            "快递": "logistics_inquiry",
            "取消": "cancel_order",
            "不要": "cancel_order"
        }
    
    def predict(self, text: str) -> str:
        """模拟DashScope模型预测"""
        for pattern, intent in self.patterns.items():
            if pattern in text:
                return intent
        return "unknown"


class MockTongyiLLMRouter:
    """模拟通义千问LLM路由"""
    
    def __init__(self, intents: List[str]):
        self.intents = intents
        # 模拟通义千问的语义理解能力
        self.semantic_patterns = {
            "状态": "query_order",
            "怎么": "refund_request",
            "如何": "refund_request",
            "需要": "issue_invoice",
            "要": "issue_invoice",
            "在哪": "logistics_inquiry",
            "哪里": "logistics_inquiry",
            "不想要": "cancel_order",
            "算了": "cancel_order"
        }
    
    def predict(self, text: str, intents: List[str]) -> str:
        """模拟通义千问语义理解"""
        for pattern, intent in self.semantic_patterns.items():
            if pattern in text:
                return intent
        return "unknown"


class VotingLogic:
    """投票逻辑"""
    
    def __init__(self, strategy: str = "priority"):
        self.strategy = strategy
    
    def vote(self, results: Dict[str, str]) -> str:
        """融合多个预测结果"""
        if self.strategy == "priority":
            return self._priority_vote(results)
        elif self.strategy == "majority":
            return self._majority_vote(results)
        else:
            return self._priority_vote(results)
    
    def _priority_vote(self, results: Dict[str, str]) -> str:
        """优先级投票：规则 > DashScope ML > 通义千问LLM"""
        if results.get("rule_intent") != "unknown":
            return results["rule_intent"]
        if results.get("ml_intent") != "unknown":
            return results["ml_intent"]
        return results.get("llm_intent", "unknown")
    
    def _majority_vote(self, results: Dict[str, str]) -> str:
        """多数投票"""
        vote_count = {}
        for method, intent in results.items():
            if intent != "unknown":
                vote_count[intent] = vote_count.get(intent, 0) + 1
        
        if not vote_count:
            return "unknown"
        
        return max(vote_count, key=vote_count.get)


class SimpleDashScopePipeline:
    """简化版DashScope意图识别流水线"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intents = config["intents"]
        
        # 初始化组件
        self.rule_engine = SimpleRuleEngine(config["rule_keywords"])
        self.ml_model = MockDashScopeMLModel()
        self.llm_router = MockTongyiLLMRouter(self.intents)
        self.voting_logic = VotingLogic(config.get("voting_strategy", "priority"))
        
        # 创建并行执行链
        self.parallel_router = RunnableParallel(
            rule_intent=lambda x: self.rule_engine.predict(x["input"]),
            ml_intent=lambda x: self.ml_model.predict(x["input"]),
            llm_intent=lambda x: self.llm_router.predict(x["input"], self.intents)
        )
    
    def predict(self, text: str) -> Dict[str, Any]:
        """预测意图"""
        try:
            # 并行执行各种方法
            results = self.parallel_router.invoke({"input": text})
            
            # 投票决定最终结果
            final_intent = self.voting_logic.vote(results)
            
            return {
                "intent": final_intent,
                "confidence": self._calculate_confidence(results, final_intent),
                "details": results
            }
        
        except Exception as e:
            print(f"意图识别失败: {e}")
            return {"intent": "unknown", "confidence": 0.0, "details": {}}
    
    def _calculate_confidence(self, results: Dict[str, str], final_intent: str) -> float:
        """计算置信度"""
        if final_intent == "unknown":
            return 0.0
        
        agreement_count = sum(1 for intent in results.values() if intent == final_intent)
        total_methods = len(results)
        
        return agreement_count / total_methods if total_methods > 0 else 0.0


def main():
    """演示简化版DashScope多策略融合意图识别"""
    
    # 配置
    config = {
        "intents": [
            "query_order",
            "refund_request", 
            "issue_invoice",
            "logistics_inquiry",
            "cancel_order",
            "unknown"
        ],
        "voting_strategy": "priority",
        "rule_keywords": {
            "query_order": ["查订单", "订单号", "订单状态", "我的订单"],
            "refund_request": ["退钱", "退款", "申请退款"],
            "issue_invoice": ["开发票", "要发票", "报销", "发票"],
            "logistics_inquiry": ["物流", "快递", "配送", "运输"],
            "cancel_order": ["取消订单", "不要了", "取消"]
        }
    }
    
    # 创建流水线
    pipeline = SimpleDashScopePipeline(config)
    
    # 测试用例
    test_cases = [
        "我想查一下我的订单状态",
        "怎么申请退款？", 
        "需要开发票",
        "物流信息在哪里看？",
        "取消订单",
        "我要报销，需要发票",
        "这个产品不想要了",
        "查看我的订单",
        "快递到哪里了？",
        "算了，不买了"
    ]
    
    print("=== 简化版DashScope多策略融合意图识别演示 ===\n")
    
    for text in test_cases:
        result = pipeline.predict(text)
        print(f"输入: {text}")
        print(f"预测意图: {result['intent']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"各方法结果:")
        for method, intent in result['details'].items():
            method_name = {
                'rule_intent': '规则引擎',
                'ml_intent': 'DashScope模型', 
                'llm_intent': '通义千问LLM'
            }.get(method, method)
            print(f"  {method_name}: {intent}")
        print("-" * 50)
    
    # 演示不同投票策略
    print("\n=== 不同投票策略对比 ===\n")
    
    config_majority = config.copy()
    config_majority["voting_strategy"] = "majority"
    pipeline_majority = SimpleDashScopePipeline(config_majority)
    
    test_text = "算了，不买了"
    
    result_priority = pipeline.predict(test_text)
    result_majority = pipeline_majority.predict(test_text)
    
    print(f"测试文本: {test_text}")
    print(f"优先级策略结果: {result_priority['intent']} (置信度: {result_priority['confidence']:.2f})")
    print(f"多数投票策略结果: {result_majority['intent']} (置信度: {result_majority['confidence']:.2f})")
    print(f"详细结果: {result_priority['details']}")


if __name__ == "__main__":
    main()