from openai import OpenAI
from typing import Any, Optional
# 导入你的基类
from .base import BaseLLM

class siliconflowLLM(BaseLLM):
    """使用 SiliconFlow API (兼容 OpenAI 格式) 实现的 BaseLLM 接口。"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        model_params: Optional[dict[str, Any]] = None,
        base_url: str = "https://api.siliconflow.cn/v1", # SiliconFlow 的默认 Base URL
        **kwargs: Any,
    ):
        # 1. 调用父类初始化
        super().__init__(model_name, model_params, **kwargs)
        
        # 2. 实例化 OpenAI 客户端 (指向 SiliconFlow 的地址)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def predict(self, input: str) -> str:
        """向 SiliconFlow 模型发送文本，并获取回复。"""
        # 调用兼容 OpenAI 的 Chat Completions 接口
        response = self.client.chat.completions.create(
            model=self.model_name,       # 例如 "Qwen/Qwen2.5-7B-Instruct"
            messages=[{"role": "user", "content": input}],
            **self.model_params,
        )
        
        # 提取并返回文本内容
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content
        else:
            raise ValueError("No response from SiliconFlow model")