from zhipuai import ZhipuAI
from typing import List
from .base import BaseEmb

class zhipuEmb(BaseEmb):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        # 同样地，初始化基类
        super().__init__(model_name=model_name, **kwargs)
        # 初始化智谱客户端
        self.client = ZhipuAI(api_key=api_key)

    def get_emb(self, text: str) -> List[float]:
        """向智谱向量模型发送文本，获取向量表示。"""
        # 调用智谱的向量接口 (Embeddings API)
        emb = self.client.embeddings.create(
            model=self.model_name, # 比如 "embedding-2"
            input=text,
        )
        # 提取向量数据（一个浮点数列表）并返回
        return emb.data[0].embedding