# 导入 Python 内置的 abc 模块，用于创建抽象基类 (Abstract Base Classes)
# abstractmethod 是一个装饰器，用来标记必须被子类实现的方法
from abc import ABC, abstractmethod
# 导入类型提示，Any 表示任意类型，Optional 表示该变量可以是某种类型，也可以是 None
from typing import Any, Optional

class BaseLLM(ABC):
    """
    大型语言模型 (LLM) 的统一接口/基类。
    所有具体的 LLM 类（如智谱、Groq、OpenAI等）都必须继承这个类。

    参数说明:
        model_name (str): 语言模型的名称 (例如 "glm-4", "llama3-8b-8192")。
        model_params (Optional[dict[str, Any]], optional): 向模型发送文本时传递的额外参数（如 temperature 温度、max_tokens 等）。默认为 None。
        **kwargs (Any): 在初始化类时传递给模型的其他关键字参数。
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        # 初始化方法：保存模型名称
        self.model_name = model_name
        # 保存模型参数，如果没传，则初始化为一个空字典 {}
        self.model_params = model_params or {}

    # @abstractmethod 是核心！它告诉 Python：
    # "这个 predict 方法在这里只是个空壳，任何继承 BaseLLM 的子类，都必须自己写代码实现这个方法！否则就报错！"
    @abstractmethod
    def predict(self, input: str) -> str:
        """
        向大语言模型 (LLM) 发送文本输入，并获取模型的回答。

        参数:
            input (str): 发送给 LLM 的提示词 (Prompt) / 文本内容。

        返回:
            str: LLM 生成的回复文本。
        """
        # 这里的 pass 表示占位，基类不需要实现具体逻辑
        pass