# 代码改进助手

基于LangGraph实现的智能代码生成和改进流水线，支持多维度代码验证和自适应反思机制。

## 核心特性

- **智能需求分析**: 自动评估代码复杂度和技术要求
- **多维度验证**: 语法、复杂度、风格三重检查
- **自适应反思**: 根据代码质量自动触发改进流程
- **灵活分支**: 基于验证结果的智能决策路径

## 项目结构

```
code_assistant/
├── main.py              # 主程序入口
├── config.py            # 配置管理
├── requirements.txt     # 依赖包列表
├── core/
│   ├── nodes.py         # LangGraph节点实现
│   ├── models.py        # 数据模型定义
│   └── utils.py         # 工具函数
└── README.md           # 项目说明
```

## 工作流程

1. **需求分析** - 解析用户需求，评估复杂度
2. **代码生成** - 根据需求生成高质量代码
3. **代码验证** - 多维度质量检查
4. **质量检查** - LLM深度评估
5. **反思改进** - 低分代码自动优化

## 使用方法

### 基本使用

```python
from main import CodeImprovementPipeline
from config import Config

# 创建配置
config = Config(
    quality_threshold=7,
    max_retry=2,
    verbose=True
)

# 创建流水线
pipeline = CodeImprovementPipeline(config)

# 运行改进流程
result = pipeline.run("创建一个计算器类")

print(f"最终评分: {result['check_result']['score']}/10")
print(f"生成代码:\n{result['final_code']}")
```

### 配置选项

- `quality_threshold`: 质量达标分数 (默认: 8)
- `reflection_threshold`: 触发反思的分数阈值 (默认: 6)
- `max_retry`: 最大重试次数 (默认: 3)
- `enable_reflection`: 是否启用反思机制 (默认: True)
- `verbose`: 是否显示详细过程 (默认: True)

## 验证维度

### 语法检查
- Python AST语法解析
- 语法错误检测

### 复杂度检查
- 代码行数评估
- 复杂度等级划分

### 风格检查
- 函数定义检查
- 文档字符串检查
- 代码规范验证

## 输出示例

```
代码改进助手
需求: 创建一个高性能的斐波那契数列计算器，支持缓存和并发
============================================================

[16:11:17] 步骤 1
需求分析: 复杂度=high

[16:11:18] 步骤 2
代码生成: 1636 字符 (第1次)
代码预览:
   1: import threading
   2: from functools import lru_cache
   3: from concurrent.futures import ThreadPoolExecutor

[16:11:19] 步骤 3
代码验证: 8/10 分
  - syntax: 10/10
  - complexity: 6/10
  - style: 10/10

[16:11:20] 步骤 4
质量检查: 8/10 分 (优秀)

============================================================
执行完成
最终评分: 8/10 分 (优秀)
验证评分: 8/10 分
重试次数: 0
```

## 依赖要求

- Python 3.8+
- langchain-community
- langgraph
- dashscope (通义千问API)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行演示

```bash
python main.py
```

## 核心修复

本版本修复了以下关键问题：

1. **验证分数显示错误** - 修复了显示0分的问题，现在正确显示实际验证分数
2. **代码精简** - 去除了所有装饰性符号，提高代码可读性
3. **注释优化** - 适合长期保存和首次阅读的清晰注释
4. **文件结构** - 使用正式文件名，保留核心功能代码