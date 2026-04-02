from zhipuai import ZhipuAI
from typing import Any, Optional
# 导入我们刚刚写好的基类
from .base import BaseLLM

class zhipuLLM(BaseLLM):
    """使用智谱 AI SDK 实现的 BaseLLM 接口。"""

    def __init__(
        self,
        model_name: str,
        api_key: str,  # 相比基类，这里多了一个必须的参数：API 密钥
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        # 1. 调用父类 (BaseLLM) 的初始化方法，把 model_name 等参数存起来
        super().__init__(model_name, model_params, **kwargs)
        # 2. 实例化智谱的客户端，这样我们就可以通过 client 与智谱服务器通信了
        self.client = ZhipuAI(api_key=api_key)

    def predict(self, input: str) -> str:
        """向智谱模型发送文本，并获取回复。"""
        # 调用智谱的聊天补全接口 (Chat Completions API)
        response = self.client.chat.completions.create(
            model=self.model_name,  # 使用初始化时指定的模型名称（如 "glm-4"）
            messages=[{"role": "user", "content": input}],  # 组装消息格式
            **self.model_params,  # 传递额外的模型参数，如 temperature 等
        )
        # 提取并返回生成的纯文本内容
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content
        else:
            raise ValueError("No response from ZhipuAI model")