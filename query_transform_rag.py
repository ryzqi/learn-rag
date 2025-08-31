"""
查询转换RAG系统

该模块提供了基于查询转换技术的RAG（检索增强生成）系统。
支持查询重写、回退查询和查询分解等多种查询转换策略，以提升检索效果。
"""

import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from LLM import GeminiLLM
from file_reader import file_reader_service, FileReader
from embedding import embedding_service, EmbeddingClient
from milvus_client import milvus_service


@dataclass
class QueryTransformConfig:
    """查询转换配置类"""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    default_top_k: int = 3
    decompose_num_subqueries: int = 4
    rewrite_temperature: float = 0.0
    step_back_temperature: float = 0.1
    decompose_temperature: float = 0.2


class QueryTransformRAG:
    """查询转换RAG系统

    基于查询转换技术的RAG系统，支持多种查询转换策略：
    - 查询重写：优化查询的具体性和详细程度
    - 回退查询：生成更通用的查询以获取背景信息
    - 查询分解：将复杂查询分解为多个子查询
    """

    def __init__(
        self,
        llm_client: Optional[GeminiLLM] = None,
        file_reader: Optional[FileReader] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        milvus_client=None,
        collection_name: str = "RAG_learn",
        config: Optional[QueryTransformConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化查询转换RAG系统

        Args:
            llm_client: LLM客户端，用于查询转换和响应生成
            file_reader: 文件读取服务
            embedding_client: 嵌入服务，用于向量生成
            milvus_client: 向量数据库客户端
            collection_name: Milvus集合名称
            config: 查询转换配置参数
            logger: 日志记录器
        """
        # 使用默认服务或提供的服务初始化
        self.llm_client = llm_client or GeminiLLM()
        self.file_reader = file_reader or file_reader_service
        self.embedding_client = embedding_client or embedding_service
        self.milvus_client = milvus_client or milvus_service

        # 配置
        self.collection_name = collection_name
        self.config = config or QueryTransformConfig()

        # 日志设置
        self.logger = logger or self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def rewrite_query(self, original_query: str) -> str:
        """查询重写

        通过重写用户查询，使其更加具体、详细，并提升检索相关信息的有效性。

        Args:
            original_query: 原始查询文本

        Returns:
            重写后的查询文本

        Raises:
            ValueError: 当查询为空时
        """
        if not original_query or not original_query.strip():
            raise ValueError("查询文本不能为空")

        original_query = original_query.strip()

        system_prompt = "您是一个专注于优化搜索查询的AI助手。您的任务是通过重写用户查询，使其更加具体、详细，并提升检索相关信息的有效性。"
        user_prompt = f"""
        请优化以下搜索查询，使其满足：
        1. 增强查询的具体性和详细程度
        2. 包含有助于获取准确信息的相关术语和核心概念

        原始查询：{original_query}

        优化后的查询：
        """

        try:
            response = self.llm_client.generate_text(
                user_prompt,
                system_instruction=system_prompt,
                temperature=self.config.rewrite_temperature,
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"查询重写失败: {e}")
            raise Exception(f"查询重写失败: {e}")

    def generate_step_back_query(self, original_query: str) -> str:
        """生成回退查询

        将特定查询转化为更宽泛、更通用的版本，以帮助检索相关背景信息。

        Args:
            original_query: 原始查询文本

        Returns:
            回退查询文本

        Raises:
            ValueError: 当查询为空时
        """
        if not original_query or not original_query.strip():
            raise ValueError("查询文本不能为空")

        original_query = original_query.strip()

        system_prompt = "您是一个专注于搜索策略的AI助手。您的任务是将特定查询转化为更宽泛、更通用的版本，以帮助检索相关背景信息。"
        user_prompt = f"""
        请基于以下具体查询生成更通用的版本，要求：
        1. 扩大查询范围以涵盖背景信息
        2. 包含潜在相关领域的关键概念
        3. 保持语义完整性

        原始查询: {original_query}

        通用化查询：
        """

        try:
            response = self.llm_client.generate_text(
                user_prompt,
                system_instruction=system_prompt,
                temperature=self.config.step_back_temperature,
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"回退查询生成失败: {e}")
            raise Exception(f"回退查询生成失败: {e}")

    def decompose_query(
        self, original_query: str, num_subqueries: Optional[int] = None
    ) -> List[str]:
        """查询分解

        将复杂的查询拆解成更简单的子问题，这些子问题的答案组合起来能够解决原始查询。

        Args:
            original_query: 原始查询文本
            num_subqueries: 子查询数量，如果未指定则使用配置中的默认值

        Returns:
            子查询列表

        Raises:
            ValueError: 当查询为空时
        """
        if not original_query or not original_query.strip():
            raise ValueError("查询文本不能为空")

        original_query = original_query.strip()
        num_subqueries = num_subqueries or self.config.decompose_num_subqueries

        system_prompt = "您是一个专门负责分解复杂问题的AI助手。您的任务是将复杂的查询拆解成更简单的子问题，这些子问题的答案组合起来能够解决原始查询。"
        user_prompt = f"""
        将以下复杂查询分解为{num_subqueries}个更简单的子问题。每个子问题应聚焦原始问题的不同方面。

        原始查询: {original_query}

        请生成{num_subqueries}个子问题，每个问题单独一行，按以下格式：
        1. [第一个子问题]
        2. [第二个子问题]
        依此类推...
        """

        try:
            response = self.llm_client.generate_text(
                user_prompt,
                system_instruction=system_prompt,
                temperature=self.config.decompose_temperature,
            )

            content = response.strip()
            pattern = r"^\d+\.\s*(.*)"

            subqueries = [
                re.match(pattern, line).group(1)
                for line in content.split("\n")
                if line.strip() and re.match(pattern, line)
            ]

            if not subqueries:
                self.logger.warning("未能解析出有效的子查询，返回原始查询")
                return [original_query]

            return subqueries

        except Exception as e:
            self.logger.error(f"查询分解失败: {e}")
            raise Exception(f"查询分解失败: {e}")

    def _chunk_text(self, text: str) -> List[str]:
        """文本分块

        Args:
            text: 要分块的文本

        Returns:
            文本块列表
        """
        chunks = []
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap

        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i : i + chunk_size])
        return chunks

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """处理文档并存储到向量数据库

        Args:
            file_path: 文件路径

        Returns:
            处理结果信息

        Raises:
            ValueError: 当文件路径为空时
            Exception: 当文档处理失败时
        """
        if not file_path or not file_path.strip():
            raise ValueError("文件路径不能为空")

        try:
            self.logger.info("读取文件...")
            text = self.file_reader.read_file(file_path)

            self.logger.info("分块...")
            chunks = self._chunk_text(text)
            self.logger.info(f"生成{len(chunks)}个块...")

            self.logger.info("生成嵌入向量...")
            chunk_embeddings = self.embedding_client.embed_texts(chunks)

            self.logger.info("存储到向量数据库中...")
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                self.milvus_client.insert_data(
                    self.collection_name,
                    {
                        "vector": embedding,
                        "text": chunk,
                        "metadata": {"source": file_path, "chunk_index": i},
                    },
                )

            collection_info = self.milvus_client.get_collection_info(
                self.collection_name
            )
            self.logger.info(f"文档处理完成，共处理{len(chunks)}个文本块")

            return {
                "success": True,
                "file_path": file_path,
                "chunks_count": len(chunks),
                "collection_info": collection_info,
            }

        except Exception as e:
            self.logger.error(f"文档处理失败: {e}")
            raise Exception(f"文档处理失败: {e}")

    def search_with_transformation(
        self, query: str, transformation_type: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """使用查询转换进行搜索

        Args:
            query: 查询文本
            transformation_type: 转换类型 ("rewrite", "step_back", "decompose")
            top_k: 返回结果数量，如果未指定则使用配置中的默认值

        Returns:
            搜索结果列表

        Raises:
            ValueError: 当查询为空或转换类型不支持时
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        if transformation_type not in ["rewrite", "step_back", "decompose"]:
            raise ValueError(f"不支持的转换类型: {transformation_type}")

        top_k = top_k or self.config.default_top_k

        try:
            self.logger.info(f"转换类型: {transformation_type}")
            self.logger.info(f"原始查询: {query}")

            if transformation_type == "rewrite":
                transformed_query = self.rewrite_query(query)
                self.logger.info(f"重写后的查询: {transformed_query}")
                
                results = self.milvus_client.search_by_text(
                    collection_name=self.collection_name,
                    text=transformed_query,
                    limit=top_k,
                    output_fields=["text", "source", "chunk_index"],
                    metric_type="COSINE",
                    embedding_client=self.embedding_client,
                )

            elif transformation_type == "step_back":
                transformed_query = self.generate_step_back_query(query)
                self.logger.info(f"回退查询: {transformed_query}")
                
                results = self.milvus_client.search_by_text(
                    collection_name=self.collection_name,
                    text=transformed_query,
                    limit=top_k,
                    output_fields=["text", "source", "chunk_index"],
                    metric_type="COSINE",
                    embedding_client=self.embedding_client,
                )

            elif transformation_type == "decompose":
                transformed_queries = self.decompose_query(query)
                self.logger.info("分解为子查询:")
                for i, sub_q in enumerate(transformed_queries, 1):
                    self.logger.info(f"{i}. {sub_q}")
                
                all_results = []

                for transformed_query in transformed_queries:
                    sub_results = self.milvus_client.search_by_text(
                        collection_name=self.collection_name,
                        text=transformed_query,
                        limit=2,
                        output_fields=["text", "source", "chunk_index"],
                        metric_type="COSINE",
                        embedding_client=self.embedding_client,
                    )
                    all_results.extend(sub_results)

                # 去重(保留相似度最高的结果)
                # 注意：Milvus使用distance字段，距离越小相似度越高（对于COSINE）
                seen_texts = {}
                for result in all_results:
                    # 兼容不同的结果格式
                    text = None
                    distance = None
                    
                    if isinstance(result, dict):
                        if "entity" in result:
                            # Milvus 2.x 格式
                            text = result.get("entity", {}).get("text")
                            distance = result.get("distance", float('inf'))
                        else:
                            # 简化格式
                            text = result.get("text")
                            distance = result.get("distance", result.get("score", float('inf')))
                    
                    if text and text.strip():
                        # 距离越小相似度越高，保留距离最小的结果
                        if text not in seen_texts or distance < seen_texts[text].get("distance", float('inf')):
                            result_copy = result.copy()
                            result_copy["distance"] = distance
                            seen_texts[text] = result_copy

                # 按距离排序（距离越小相似度越高）
                results = sorted(
                    seen_texts.values(), key=lambda x: x.get("distance", float('inf'))
                )[:top_k]

            return results

        except Exception as e:
            self.logger.error(f"转换搜索失败: {e}")
            raise Exception(f"转换搜索失败: {e}")

    def generate_response(self, query: str, context: str) -> str:
        """基于上下文生成回答

        Args:
            query: 用户查询
            context: 检索到的上下文信息

        Returns:
            AI生成的回答

        Raises:
            ValueError: 当查询或上下文为空时
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        if not context or not context.strip():
            raise ValueError("上下文不能为空")

        system_prompt = "您是一个乐于助人的AI助手。请仅根据提供的上下文来回答用户的问题。如果在上下文中找不到答案，请直接说'没有足够的信息'。"
        user_prompt = f"""
        上下文内容:
        {context}

        问题: {query}

        请基于上述上下文内容提供一个全面详尽的答案。
        """

        try:
            response = self.llm_client.generate_text(
                user_prompt, system_instruction=system_prompt, temperature=0
            )
            return response.strip()
        except Exception as e:
            self.logger.error(f"响应生成失败: {e}")
            raise Exception(f"响应生成失败: {e}")

    def query_with_transformation(
        self, query: str, transformation_type: str, top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """使用查询转换进行完整的RAG查询

        Args:
            query: 用户查询
            transformation_type: 转换类型 ("rewrite", "step_back", "decompose")
            top_k: 返回结果数量

        Returns:
            包含查询、转换类型、上下文和回答的字典
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        try:
            # 执行转换搜索
            results = self.search_with_transformation(query, transformation_type, top_k)

            # 准备上下文
            context_parts = []
            for i, result in enumerate(results):
                # 兼容不同的Milvus结果格式
                text = ""
                if isinstance(result, dict):
                    if "entity" in result:
                        # Milvus 2.x 格式
                        text = result.get("entity", {}).get("text", "")
                    else:
                        # 简化格式
                        text = result.get("text", "")
                
                if text and text.strip():
                    context_parts.append(f"段落 {i + 1}:\n{text}")

            context = "\n\n".join(context_parts)

            if not context.strip():
                return {
                    "original_query": query,
                    "transformation_type": transformation_type,
                    "context": "",
                    "response": "没有找到相关信息",
                }

            # 生成回答
            response = self.generate_response(query, context)

            return {
                "original_query": query,
                "transformation_type": transformation_type,
                "context": context,
                "response": response,
            }

        except Exception as e:
            self.logger.error(f"查询处理失败: {e}")
            raise Exception(f"查询处理失败: {e}")


# 创建全局服务实例，便于直接导入使用
query_transform_rag_service = QueryTransformRAG()


if __name__ == "__main__":
    # 示例用法
    try:
        # 创建RAG实例
        rag = QueryTransformRAG(collection_name="RAG_learn")

        # 处理文档（如果需要）
        # result = rag.process_document("Agent基础.md")
        # print(f"文档处理结果: {result}")

        # 执行查询转换RAG
        query = "思维链(CoT)是什么？举个例子"
        response = rag.query_with_transformation(query, "decompose")

        print(f"原始查询: {response['original_query']}")
        print(f"转换类型: {response['transformation_type']}")
        print(f"回答: {response['response']}")

    except Exception as e:
        print(f"执行失败: {e}")
