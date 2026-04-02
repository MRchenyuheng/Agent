import json
import uuid
import time
import os
from typing import List, Dict, Optional

class JSONMemoryBank:
    def __init__(self, embedding_model, file_path: str = "memory_bank.json"):
        """
        初始化 JSON 记忆库
        :param embedding_model: 向量化模型实例 (你的 zhipuEmb)
        :param file_path: JSON 文件存储路径
        """
        self.emb = embedding_model
        self.file_path = file_path
        self.memories: List[Dict] = []  # 内存中的记忆列表
        
        # 启动时加载已有记忆
        self._load_from_disk()

    def _load_from_disk(self):
        """从 JSON 文件加载记忆到内存"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.memories = json.load(f)
                print(f"[Memory] 成功加载 {len(self.memories)} 条记忆")
            except json.JSONDecodeError:
                print(f"[Memory] 文件损坏，初始化空记忆库")
                self.memories = []
        else:
            print(f"[Memory] 文件不存在，将创建新记忆库: {self.file_path}")

    def _save_to_disk(self):
        """将内存中的记忆保存到 JSON 文件 (原子写入，防止数据损坏)"""
        # 先写入临时文件
        temp_path = f"{self.file_path}.tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
            # 原子替换
            os.replace(temp_path, self.file_path)
        except Exception as e:
            print(f"[Memory] 保存失败: {e}")
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def add(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        添加一条记忆到库中
        :param text: 记忆的文本内容
        :param metadata: 可选的元数据 (如来源、类型等)
        :return: 生成的记忆 ID
        """
        # 1. 生成唯一ID和时间戳
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

        # 2. 生成向量 (调用你的 Embedding 模型)
        # --- 这里已修正为 get_emb ---
        print(f"[Memory] 正在向量化记忆...")
        vector = self.emb.get_emb(text) 

        # 3. 构建记忆对象
        memory_item = {
            "id": memory_id,
            "text": text,
            "vector": vector,  # 存储向量，方便后续做检索
            "metadata": metadata or {},
            "timestamp": timestamp,
            "formatted_time": formatted_time
        }

        # 4. 写入内存并持久化
        self.memories.append(memory_item)
        self._save_to_disk()
        
        print(f"[Memory] 已添加记忆 (ID: {memory_id[-8:]}...)")
        return memory_id

    def get_all(self) -> List[Dict]:
        """获取所有记忆 (用于调试)"""
        return self.memories

    def clear(self):
        """清空记忆库 (危险操作，仅用于测试)"""
        self.memories = []
        self._save_to_disk()
        print("[Memory] 已清空所有记忆")