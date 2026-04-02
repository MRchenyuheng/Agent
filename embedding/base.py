from abc import ABC, abstractmethod
# List 用于表示列表类型
from typing import List, Any, Optional

class BaseEmb(ABC):
    """
    文本向量化 (Embedding) 模型的统一接口/基类。
    Embedding 的作用是把人类的文字转换成计算机能理解的数字向量 (一串浮点数)。
    """
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        # 同样地，保存向量模型的名称（如 "embedding-2"）
        self.model_name = model_name
        # 保存其他配置参数
        self.model_params = model_params or {}

    # 强制子类必须实现 get_emb 方法
    @abstractmethod
    def get_emb(self, input: str) -> List[float]:
        """
        向 Embedding 模型发送文本，并获取该文本对应的向量。

        参数:
            input (str): 需要被向量化的文本。

        返回:
            List[float]: 模型返回的嵌入向量（一个由浮点数组成的列表，例如 [0.12, -0.45, 0.88, ...]）。
        """
        pass