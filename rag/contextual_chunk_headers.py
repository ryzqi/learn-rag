"""
上下文增强分块处理器

该模块提供了用于文本分块、标题生成、嵌入向量生成和语义搜索的客户端类。
支持基于上下文的文本分块，为每个文本块生成描述性标题，并提供语义搜索和问答功能。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Dict, Any
from LLM import GeminiLLM
from file_reader import file_reader_service, FileReader
from embedding import embedding_service, EmbeddingClient
from milvus_client import milvus_service
import hashlib


class ContextualChunkProcessor:
    """上下文增强分块处理器

    该类实现了基于上下文的文本分块和检索功能，
    为每个文本块生成描述性标题，并提供语义搜索和问答功能。
    使用Milvus进行向量存储以提升性能和可扩展性。
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        batch_size: int = 64,
        llm_client: Optional[GeminiLLM] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        file_reader: Optional[FileReader] = None,
        collection_name: str = "RAG_learn",
    ):
        """初始化上下文增强分块处理器

        Args:
            chunk_size: 文本块大小，默认为1000字符
            overlap: 文本块重叠大小，默认为200字符
            batch_size: 批处理大小，默认为64
            llm_client: LLM客户端实例，如果未提供则创建新实例
            embedding_client: 嵌入客户端实例，如果未提供则使用全局服务
            file_reader: 文件读取器实例，如果未提供则使用全局服务
            collection_name: Milvus集合名称
        """
        # 配置参数
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size

        # 验证参数
        if chunk_size <= 0:
            raise ValueError("文本块大小必须大于0")
        if overlap < 0:
            raise ValueError("重叠大小不能为负数")
        if overlap >= chunk_size:
            raise ValueError("重叠大小不能大于或等于文本块大小")

        # 初始化服务客户端
        self.llm_client = llm_client or GeminiLLM()
        self.embedding_client = embedding_client or embedding_service
        self.file_reader = file_reader or file_reader_service
        self.milvus_client = milvus_service
        self.collection_name = collection_name

        # 设置默认系统提示
        self.header_generation_prompt = (
            "为给定的文本生成一个简洁且信息丰富的标题，直接返回标题，不得返回其余内容。"
        )
        self.qa_system_prompt = (
            "你是一个AI助手，严格根据给定的上下文进行回答。"
            "如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"
        )

        # 存储处理后的数据
        self._text_chunks: List[Dict[str, str]] = []

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """处理文件并生成分块和嵌入向量，存储到Milvus

        Args:
            file_path: 文件路径

        Returns:
            Dict: 包含处理结果的字典
        """
        if not file_path or not file_path.strip():
            raise ValueError("文件路径不能为空")

        try:
            # 读取文件内容
            text = self.file_reader.read_file(file_path)

            # 生成文本块和标题
            self._text_chunks = self.chunk_text_with_headers(text)

            if not self._text_chunks:
                raise ValueError("文档分块后没有有效内容")

            # 批量生成嵌入向量并存储到Milvus
            result = self._generate_and_store_embeddings(file_path, self._text_chunks)

            return {
                "success": True,
                "chunks_count": len(self._text_chunks),
                "file_path": file_path,
                **result,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"文件处理失败: {e}",
                "file_path": file_path,
            }

    def chunk_text_with_headers(self, text: str) -> List[Dict[str, str]]:
        """将文本分块并为每个块生成标题

        Args:
            text: 要分块的文本

        Returns:
            包含标题和文本的字典列表
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        chunks = []

        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i : i + self.chunk_size]
            if chunk_text.strip():  # 只添加非空块
                try:
                    header = self._generate_chunk_header(chunk_text)
                    chunks.append({"header": header, "text": chunk_text})
                except Exception:
                    # 如果标题生成失败，使用默认标题
                    default_header = f"文本块 {len(chunks) + 1}"
                    chunks.append({"header": default_header, "text": chunk_text})

        return chunks

    def _batch_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本嵌入向量，包含错误处理"""
        if not texts:
            raise ValueError("文本列表不能为空")

        try:
            # 如果文本数量超过批处理大小，分批处理
            if len(texts) <= self.batch_size:
                return self.embedding_client.embed_texts(texts)

            # 分批处理大量文本
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_embeddings = self.embedding_client.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)

            return all_embeddings
        except Exception as e:
            raise Exception(f"嵌入向量生成失败: {e}")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """执行语义搜索

        Args:
            query: 查询文本
            limit: 返回的最相关文本块数量

        Returns:
            最相关的文本块列表
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        try:
            # 生成查询嵌入向量
            query_embedding = self.embedding_client.embed_text(query.strip())
            
            # 使用Milvus进行文本向量搜索，获取较多候选结果
            text_results = self.milvus_client.search_by_vector(
                collection_name=self.collection_name,
                vector=query_embedding,
                vector_field="text_vector",
                limit=limit * 3,  # 获取更多候选以便重新排序
                output_fields=["text", "header", "source", "chunk_index", "text_vector", "header_vector"],
                metric_type="COSINE"
            )
            
            if not text_results:
                return []
            
            # 重新计算相似度分数
            similarities = []
            for result in text_results:
                entity = result.get("entity", result)
                
                # 分别计算文本和标题的余弦相似度
                text_vector = entity.get("text_vector", [])
                header_vector = entity.get("header_vector", [])
                
                if text_vector and header_vector:
                    sim_text = self._cosine_similarity(query_embedding, text_vector)
                    sim_header = self._cosine_similarity(query_embedding, header_vector)
                    
                    # 计算平均相似度分数
                    avg_similarity = (sim_text + sim_header) / 2
                    
                    similarities.append((result, avg_similarity))
            
            # 按相似度分数降序排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 返回top-k最相关的块
            return [x[0] for x in similarities[:limit]]

        except Exception as e:
            raise Exception(f"语义搜索失败: {e}")

    def generate_response(self, query: str, limit: int = 3) -> str:
        """基于检索到的上下文生成回答

        Args:
            query: 用户查询
            limit: 检索的文本块数量

        Returns:
            AI生成的回答
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        try:
            # 检索相关文本块
            top_chunks = self.search(query, limit)

            if not top_chunks:
                return "我没有找到相关的信息来回答这个问题。"

            # 构建上下文
            context = "\n".join(
                [
                    f"上下文 {i + 1} - {chunk.get('entity', {}).get('header', '') if 'entity' in chunk else chunk.get('header', '')}:\n{chunk.get('entity', {}).get('text', '') if 'entity' in chunk else chunk.get('text', '')}"
                    for i, chunk in enumerate(top_chunks)
                    if (
                        chunk.get("entity", {}).get("text", "")
                        if "entity" in chunk
                        else chunk.get("text", "")
                    ).strip()
                ]
            )

            # 构建用户提示
            user_prompt = f"{context}\n\n问题: {query}"

            # 生成回答
            response = self.llm_client.generate_text(
                user_prompt, system_instruction=self.qa_system_prompt
            )

            return response.strip()

        except Exception as e:
            return f"回答生成失败: {e}"

    def query(self, question: str, limit: int = 3) -> Dict[str, Any]:
        """完整的查询流程，返回回答和相关上下文

        Args:
            question: 用户问题
            limit: 检索的文本块数量

        Returns:
            包含回答和上下文的字典
        """
        try:
            # 检索相关文本块
            relevant_chunks = self.search(question, limit)

            # 生成回答
            answer = self.generate_response(question, limit)

            return {
                "question": question,
                "answer": answer,
                "relevant_chunks": [
                    {
                        "text": chunk.get("entity", {}).get("text", "")
                        if "entity" in chunk
                        else chunk.get("text", ""),
                        "header": chunk.get("entity", {}).get("header", "")
                        if "entity" in chunk
                        else chunk.get("header", ""),
                    }
                    for chunk in relevant_chunks
                ],
                "chunk_count": len(relevant_chunks),
            }

        except Exception as e:
            return {
                "question": question,
                "answer": f"查询处理失败: {e}",
                "relevant_chunks": [],
                "chunk_count": 0,
            }

    def _generate_chunk_header(self, chunk: str) -> str:
        """为文本块生成标题（私有方法）

        Args:
            chunk: 文本块

        Returns:
            生成的标题
        """
        response = self.llm_client.generate_text(
            chunk, system_instruction=self.header_generation_prompt
        )
        return response.strip()

    def _generate_and_store_embeddings(
        self, file_path: str, chunks: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """生成嵌入向量并存储到Milvus"""
        try:
            # 收集所有文本和标题用于批量处理
            texts = [chunk["text"] for chunk in chunks]
            headers = [chunk["header"] for chunk in chunks]

            # 批量生成嵌入向量
            text_embeddings = self._batch_embed_texts(texts)
            header_embeddings = self._batch_embed_texts(headers)

            # 准备插入数据
            data_to_insert = []
            for i, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(file_path, i)

                # 分别存储文本和标题嵌入向量
                data_to_insert.append(
                    {
                        "id": chunk_id,
                        "text_vector": text_embeddings[i],
                        "header_vector": header_embeddings[i],
                        "text": chunk["text"],
                        "header": chunk["header"],
                        "source": file_path,
                        "chunk_index": i,
                    }
                )

            # 批量插入到Milvus
            result = self.milvus_client.insert_data(
                self.collection_name, data_to_insert
            )
            return result

        except Exception as e:
            raise Exception(f"存储到Milvus失败: {e}")

    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """生成文本块的唯一ID"""
        content = f"{file_path}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            余弦相似度分数
        """
        import numpy as np
        
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        
        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    @property
    def chunk_count(self) -> int:
        """获取文本块数量"""
        return len(self._text_chunks)


# 创建全局服务实例，便于直接导入使用
contextual_chunk_service = ContextualChunkProcessor()


if __name__ == "__main__":
    # 创建处理器实例
    processor = ContextualChunkProcessor()

    try:
        # 处理文件（取消注释以使用）
        # result = processor.process_file("Agent基础.md")
        # print(f"处理结果: {result}")

        # 执行查询
        query = "思维链(CoT)是什么？举个例子"
        result = processor.query(query, limit=3)

        print(f"问题: {result['question']}")
        print(f"回答: {result['answer']}")

    except Exception as e:
        print(f"处理失败: {e}")
