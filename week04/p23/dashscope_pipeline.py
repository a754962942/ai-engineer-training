"""
基于DashScope的多策略融合意图识别流水线
使用通义千问模型替代OpenAI和HuggingFace
"""

import json
import requests
from typing import Dict, List, Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_community.llms import Tongyi


class RuleEngine:
    """规则引擎 - 基于关键词匹配的快速意图识别"""
    
    def __init__(self, keywords_config: Dict[str, List[str]]):
        self.keywords = keywords_config
    
    def predict(self, text: str) -> str:
        """基于关键词规则判断意图"""
        text = text.lower()
        for intent, words in self.keywords.items():
            if any(word in text for word in words):
                return intent
        return "unknown"


class DashScopeMLModel:
    """DashScope机器学习模型 - 使用通义千问进行意图分类"""
    
    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def predict(self, text: str) -> str:
        """调用DashScope模型进行意图预测"""
        try:
            # 构建分类提示
            prompt = f"""请对以下文本进行意图分类，从这些选项中选择一个：
                        query_order（查询订单）
                        refund_request（退款申请）
                        issue_invoice（开具发票）
                        logistics_inquiry（物流查询）
                        cancel_order（取消订单）

                        文本：{text}

                        请只返回意图名称，不要其他解释。"""

            payload = {
                "model": self.model_name,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                },
                "parameters": {
                    "temperature": 0.1,
                    "max_tokens": 50
                }
            }
            
            response = requests.post(
                f"{self.base_url}/services/aigc/text-generation/generation",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if "output" in result and "text" in result["output"]:
                    intent = result["output"]["text"].strip().lower()
                    # 清理返回结果，提取意图名称
                    for valid_intent in ["query_order", "refund_request", "issue_invoice", "logistics_inquiry", "cancel_order"]:
                        if valid_intent in intent:
                            return valid_intent
            
            return "unknown"
        except Exception as e:
            print(f"DashScope API调用失败: {e}")
            return "unknown"


class TongyiLLMRouter:
    """通义千问LLM路由 - 处理复杂和模糊的意图识别"""
    
    def __init__(self, api_key: str, model_name: str = "qwen-plus"):
        self.llm = Tongyi(
            dashscope_api_key=api_key,
            model_name=model_name,
            temperature=0
        )
        self.prompt = ChatPromptTemplate.from_template("""
                    你是一个意图分类器，请从以下选项中选择最匹配的意图：
                    {intents}

                    示例：
                    输入：我想查订单 → query_order
                    输入：怎么退款？ → refund_request
                    输入：开发票 → issue_invoice
                    输入：物流在哪里看？ → logistics_inquiry
                    输入：不要这个订单了 → cancel_order

                    用户输入：{input}

                    请只返回意图名称。
        """)
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def predict(self, text: str, intents: List[str]) -> str:
        """使用通义千问进行意图预测"""
        try:
            result = self.chain.invoke({
                "intents": "\n".join([f"- {intent}" for intent in intents]),
                "input": text
            })
            
            # 清理返回结果
            result = result.strip().lower()
            for intent in intents:
                if intent.lower() in result:
                    return intent
            
            return "unknown"
        except Exception as e:
            print(f"通义千问预测失败: {e}")
            return "unknown"


class VotingLogic:
    """投票逻辑 - 融合多种方法的预测结果"""
    
    def __init__(self, strategy: str = "priority"):
        """
        初始化投票策略
        strategy: 'priority' | 'majority' | 'confidence'
        """
        self.strategy = strategy
    
    def vote(self, results: Dict[str, str]) -> str:
        """根据策略融合多个预测结果"""
        if self.strategy == "priority":
            return self._priority_vote(results)
        elif self.strategy == "majority":
            return self._majority_vote(results)
        else:
            return self._priority_vote(results)
    
    def _priority_vote(self, results: Dict[str, str]) -> str:
        """优先级投票：规则 > ML模型 > LLM"""
        if results.get("rule_intent") != "unknown":
            return results["rule_intent"]
        if results.get("ml_intent") != "unknown":
            return results["ml_intent"]
        return results.get("llm_intent", "unknown")
    
    def _majority_vote(self, results: Dict[str, str]) -> str:
        """多数投票：统计各方法的预测结果，选择得票最多的"""
        vote_count = {}
        for method, intent in results.items():
            if intent != "unknown":
                vote_count[intent] = vote_count.get(intent, 0) + 1
        
        if not vote_count:
            return "unknown"
        
        return max(vote_count, key=vote_count.get)


class DashScopeIntentPipeline:
    """基于DashScope的多策略融合意图识别流水线"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intents = config["intents"]
        
        # 初始化各个组件
        self._init_rule_engine()
        self._init_ml_model()
        self._init_llm_router()
        self._init_voting_logic()
        
        # 创建并行执行链
        self._create_pipeline()
    
    def _init_rule_engine(self):
        """初始化规则引擎"""
        if self.config.get("enable_rule_engine", True):
            self.rule_engine = RuleEngine(self.config["rule_keywords"])
        else:
            self.rule_engine = None
    
    def _init_ml_model(self):
        """初始化DashScope ML模型"""
        if self.config.get("enable_ml_model", True):
            ml_config = self.config.get("ml_model", {})
            self.ml_model = DashScopeMLModel(
                model_name=ml_config.get("model_name", "qwen-turbo"),
                api_key=ml_config.get("api_key", ""),
                base_url=ml_config.get("base_url", "https://dashscope.aliyuncs.com/api/v1")
            )
        else:
            self.ml_model = None
    
    def _init_llm_router(self):
        """初始化通义千问LLM路由"""
        if self.config.get("enable_llm_router", True):
            llm_config = self.config.get("llm", {})
            self.llm_router = TongyiLLMRouter(
                api_key=llm_config.get("api_key", ""),
                model_name=llm_config.get("model", "qwen-plus")
            )
        else:
            self.llm_router = None
    
    def _init_voting_logic(self):
        """初始化投票逻辑"""
        strategy = self.config.get("voting_strategy", "priority")
        self.voting_logic = VotingLogic(strategy)
    
    def _create_pipeline(self):
        """创建并行执行流水线"""
        runnables = {}
        
        if self.rule_engine:
            runnables["rule_intent"] = lambda x: self.rule_engine.predict(x["input"])
        
        if self.ml_model:
            runnables["ml_intent"] = lambda x: self.ml_model.predict(x["input"])
        
        if self.llm_router:
            runnables["llm_intent"] = lambda x: self.llm_router.predict(x["input"], self.intents)
        
        if runnables:
            self.parallel_router = RunnableParallel(**runnables)
        else:
            self.parallel_router = None
    
    def predict(self, text: str) -> Dict[str, Any]:
        """预测用户输入的意图"""
        if not self.parallel_router:
            return {"intent": "unknown", "confidence": 0.0, "details": {}}
        
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
        """计算预测置信度"""
        if final_intent == "unknown":
            return 0.0
        
        agreement_count = sum(1 for intent in results.values() if intent == final_intent)
        total_methods = len(results)
        
        return agreement_count / total_methods if total_methods > 0 else 0.0


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def main():
    """演示基于DashScope的多策略融合意图识别"""
    # 加载配置
    config = load_config("config.json")
    
    # 检查API密钥配置
    if not config.get("ml_model", {}).get("api_key") or config["ml_model"]["api_key"] == "your_dashscope_api_key_here":
        print("警告: 请在config.json中配置DashScope API密钥")
        print("当前将只使用规则引擎进行演示")
        config["enable_ml_model"] = False
        config["enable_llm_router"] = False
    
    # 创建意图识别流水线
    pipeline = DashScopeIntentPipeline(config)
    
    # 测试用例
    test_cases = [
        "我想查一下我的订单状态",
        "怎么申请退款？",
        "需要开发票",
        "物流信息在哪里看？",
        "取消订单",
        "我要报销，需要发票"
    ]
    
    print("=== 基于DashScope的多策略融合意图识别演示 ===\n")
    
    for text in test_cases:
        result = pipeline.predict(text)
        print(f"输入: {text}")
        print(f"预测意图: {result['intent']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"详细结果: {result['details']}")
        print("-" * 50)


if __name__ == "__main__":
    main()