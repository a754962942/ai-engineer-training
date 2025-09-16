"""
配置管理模块
定义代码改进助手的所有配置参数
"""

from dataclasses import dataclass


@dataclass
class Config:
    """
    代码改进助手配置类
    
    包含LLM模型配置、流程控制参数、输出设置等
    """
    
    # LLM模型配置
    model_name: str = "qwen-turbo"          # 使用的语言模型名称
    temperature: float = 0.7                # 模型创造性参数，0-1之间
    
    # 流程控制配置
    max_retry: int = 3                      # 最大重试次数
    enable_reflection: bool = True          # 是否启用反思机制
    quality_threshold: int = 8              # 代码质量达标分数
    reflection_threshold: int = 6           # 触发反思的分数阈值
    
    # 输出显示配置
    verbose: bool = True                    # 是否显示详细执行过程
    show_code_preview: bool = True          # 是否显示代码预览
    max_preview_lines: int = 5              # 代码预览最大行数
    
    # 日志配置
    log_level: str = "INFO"                 # 日志级别
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """从字典创建配置对象"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    def to_dict(self) -> dict:
        """将配置对象转换为字典"""
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_retry': self.max_retry,
            'enable_reflection': self.enable_reflection,
            'quality_threshold': self.quality_threshold,
            'reflection_threshold': self.reflection_threshold,
            'verbose': self.verbose,
            'show_code_preview': self.show_code_preview,
            'max_preview_lines': self.max_preview_lines,
            'log_level': self.log_level
        }