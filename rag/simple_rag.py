import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
from milvus_client import milvus_service
import hashlib
from typing import List, Dict, Any


class SimpleRAG:
    def __init__(
        self, chunk_size=1000, overlap=200, batch_size=64, collection_name="RAG_learn"
    ):
        """
        初始化SimpleRAG类
        Args:
            chunk_size (int): 每个文本块的最大长度，默认1000
            overlap (int): 相邻块之间的重叠长度，默认200
            batch_size (int): 批处理大小，默认64
            collection_name (str): Milvus集合名称，必须指定
        """
        self.llm_client = GeminiLLM()
        self.file_reader = file_reader_service
        self.embedding_client = embedding_service
        self.milvus_client = milvus_service

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.collection_name = collection_name

        self.system_prompt = (
            "你是一个AI助手，严格根据给定的上下文进行回答。"
            "如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        将文本分割成重叠的块

        Args:
            text: 输入文本

        Returns:
            文本块列表
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():  # 只添加非空块
                chunks.append(chunk)

        return chunks

    def _batch_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        if not texts:
            raise ValueError("文本列表不能为空")

        try:
            # 如果文本数量超过批处理大小，分批处理
            if len(texts) <= self.batch_size:
                return self.embedding_client.embed_texts(texts)
            
            # 分批处理大量文本
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.embedding_client.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            raise Exception(f"嵌入向量生成失败: {e}")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        使用语义相似度搜索相关文档

        Args:
            query: 搜索查询文本
            limit: 返回的最大结果数

        Returns:
            搜索结果列表
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")

        try:
            return self.milvus_client.search_by_text(
                collection_name=self.collection_name,
                text=query,
                limit=limit,
                output_fields=["text", "source", "chunk_index"],
                metric_type="COSINE",
                embedding_client=self.embedding_client,
            )
        except Exception as e:
            raise Exception(f"搜索失败: {e}")

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        处理文档并存储到向量数据库

        Args:
            file_path: 文件路径

        Returns:
            处理结果字典
        """
        try:
            # 读取文档并分块
            text = self.file_reader.read_file(file_path)
            text_chunks = self.chunk_text(text)

            if not text_chunks:
                raise ValueError("文档分块后没有有效内容")

            # 批量生成嵌入向量
            embeddings = self._batch_embed_texts(text_chunks)

            # 准备插入数据
            data_to_insert = []
            for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
                chunk_id = self._generate_chunk_id(file_path, i)
                data_to_insert.append(
                    {
                        "id": chunk_id,
                        "vector": embedding,
                        "text": chunk,
                        "source": file_path,
                        "chunk_index": i,
                    }
                )

            # 批量插入到Milvus
            result = self.milvus_client.insert_data(
                self.collection_name, data_to_insert
            )

            return {
                "success": True,
                "chunks_count": len(text_chunks),
                "file_path": file_path,
                **result,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "file_path": file_path}

    def generate_response(self, query: str, context: str) -> str:
        """
        基于查询和上下文生成响应

        Args:
            query: 用户的问题
            context: 从搜索结果检索的上下文

        Returns:
            生成的响应文本
        """
        if not context or not context.strip():
            return "我没有找到相关的信息来回答这个问题。"

        user_prompt = f"上下文:\n{context}\n\n用户问题：{query}"
        return self.llm_client.generate_text(
            user_prompt, system_instruction=self.system_prompt
        )

    def query(self, question: str, limit: int = 5) -> str:
        """
        完整的查询流水线：搜索、准备上下文和生成响应

        Args:
            question: 用户的问题
            limit: 最大搜索结果数

        Returns:
            生成的回答
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        try:
            # 搜索相关文档
            search_results = self.search(question, limit)

            # 准备上下文
            context_parts = []
            for i, result in enumerate(search_results):
                # 正确访问Milvus搜索结果的结构
                text = (
                    result.get("entity", {}).get("text", "")
                    if "entity" in result
                    else result.get("text", "")
                )
                if text.strip():
                    context_parts.append(f"上下文{i + 1}:\n{text}")

            context = "\n\n".join(context_parts)

            # 生成响应
            return self.generate_response(question, context)

        except Exception as e:
            return f"查询处理失败: {e}"

    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """生成文本块的唯一ID"""
        content = f"{file_path}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()


# 示例用法
if __name__ == "__main__":
    # 创建SimpleRAG实例
    rag = SimpleRAG()

    try:
        # 处理文档（取消注释以使用）
        # result = rag.process_document("Agent基础.md")
        # print(f"处理结果: {result}")

        # 查询问题
        query = "思维链是什么？举个例子"
        response = rag.query(query)

        print(f"查询: {query}")
        print(f"回答: {response}")
    except Exception as e:
        print(f"错误: {e}")
