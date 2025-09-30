"""
重排序RAG系统

该模块提供了基于重排序技术的RAG（检索增强生成）系统。
支持文档处理、向量检索和重排序优化，以提升检索质量和回答准确性。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from LLM import GeminiLLM
from file_reader import file_reader_service, FileReader
from embedding import embedding_service, EmbeddingClient
from milvus_client import milvus_service
from rerank import rerank_service


@dataclass
class RerankRAGConfig:
    """重排序RAG配置类"""

    chunk_size: int = 1000  # 文本块大小
    chunk_overlap: int = 200  # 块之间的重叠
    collection_name: str = "RAG_learn"  # 向量数据库集合名称
    initial_search_limit: int = 10  # 初始检索数量（用于重排序）
    default_top_k: int = 3  # 默认返回的重排序结果数量
    system_prompt: str = "你是一个乐于助人的AI助手。请仅根据提供的上下文来回答用户的问题。如果在上下文中找不到答案，请直接说'没有足够的信息'。"


class RerankRAG:
    """重排序RAG系统

    提供基于重排序技术的检索增强生成功能，包括：
    - 文档处理和向量化
    - 向量相似度检索
    - 重排序优化
    - 上下文生成和回答
    """

    def __init__(
        self,
        llm_client: Optional[GeminiLLM] = None,
        file_reader: Optional[FileReader] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        milvus_client=None,
        rerank_client=None,
        config: Optional[RerankRAGConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """初始化重排序RAG系统

        Args:
            llm_client: LLM客户端，用于生成回答
            file_reader: 文件读取服务
            embedding_client: 嵌入服务，用于向量生成
            milvus_client: 向量数据库客户端
            rerank_client: 重排序服务客户端
            config: 系统配置参数
            logger: 日志记录器
        """
        # 使用默认服务或提供的服务初始化
        self.llm_client = llm_client or GeminiLLM()
        self.file_reader = file_reader or file_reader_service
        self.embedding_client = embedding_client or embedding_service
        self.milvus_client = milvus_client or milvus_service
        self.rerank_client = rerank_client or rerank_service

        # 配置
        self.config = config or RerankRAGConfig()

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

    def chunk_text(self, text: str) -> List[str]:
        """将文本分割为块

        Args:
            text: 要分割的文本

        Returns:
            文本块列表

        Raises:
            ValueError: 当文本为空时
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():  # 只添加非空块
                chunks.append(chunk)

        return chunks

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """处理文档并存储到向量数据库

        Args:
            file_path: 文档文件路径

        Returns:
            处理结果字典，包含成功状态和统计信息

        Raises:
            ValueError: 当文件路径无效或处理失败时
            Exception: 当向量数据库操作失败时
        """
        try:
            self.logger.info(f"开始处理文档: {file_path}")

            # 读取文档
            text = self.file_reader.read_file(file_path)
            if not text:
                raise ValueError(f"文档内容为空: {file_path}")

            # 分块文本
            text_chunks = self.chunk_text(text)
            self.logger.info(f"文档分割为 {len(text_chunks)} 个文本块")

            # 生成嵌入向量
            embeddings = self.embedding_client.embed_texts(text_chunks)

            # 准备插入数据
            data_to_insert = []
            for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
                data_to_insert.append(
                    {
                        "vector": embedding,
                        "text": chunk,
                        "metadata": {"index": i, "source": file_path},
                    }
                )

            # 批量插入到Milvus
            result = self.milvus_client.insert_data(
                self.config.collection_name, data_to_insert
            )

            self.logger.info(f"文档处理完成，插入 {len(data_to_insert)} 条记录")

            return {
                "success": True,
                "file_path": file_path,
                "chunks_count": len(text_chunks),
                "inserted_count": len(data_to_insert),
                "milvus_result": result,
            }

        except Exception as e:
            self.logger.error(f"文档处理失败: {e}")
            raise Exception(f"文档处理失败: {e}")

    def search_and_rerank(
        self,
        query: str,
        top_k: Optional[int] = None,
        initial_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """执行搜索和重排序

        Args:
            query: 查询文本
            top_k: 返回的重排序结果数量
            initial_limit: 初始检索数量

        Returns:
            重排序后的结果列表

        Raises:
            ValueError: 当查询为空时
            Exception: 当搜索或重排序失败时
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        top_k = top_k or self.config.default_top_k
        initial_limit = initial_limit or self.config.initial_search_limit

        try:
            self.logger.info(f"执行查询: {query}")

            # 生成查询向量
            query_embedding = self.embedding_client.embed_text(query)

            # 初始向量检索
            initial_results = self.milvus_client.search_data(
                collection_name=self.config.collection_name,
                data=query_embedding,
                limit=initial_limit,
                output_fields=["text", "metadata"],
                metric_type="COSINE",
            )

            if not initial_results:
                self.logger.warning("未找到相关文档")
                return []

            # 提取文档文本
            documents = [result["entity"]["text"] for result in initial_results]

            # 重排序
            reranked_results = self.rerank_client.rerank(query, documents, top_k)

            self.logger.info(f"重排序完成，返回 {len(reranked_results)} 个结果")

            return reranked_results

        except Exception as e:
            self.logger.error(f"搜索和重排序失败: {e}")
            raise Exception(f"搜索和重排序失败: {e}")

    def generate_response(
        self, query: str, reranked_results: List[Dict[str, Any]]
    ) -> str:
        """基于重排序结果生成回答

        Args:
            query: 原始查询
            reranked_results: 重排序后的结果

        Returns:
            生成的回答文本

        Raises:
            Exception: 当回答生成失败时
        """
        try:
            if not reranked_results:
                return "没有找到相关信息来回答您的问题。"

            # 构建上下文
            context = "\n\n===\n\n".join(
                [result["document"]["text"] for result in reranked_results]
            )

            # 生成回答
            prompt = f"上下文:\n{context}\n\n问题: {query}"
            response = self.llm_client.generate_text(
                prompt, system_instruction=self.config.system_prompt
            )

            return response

        except Exception as e:
            self.logger.error(f"回答生成失败: {e}")
            raise Exception(f"回答生成失败: {e}")

    def query(
        self, query: str, top_k: Optional[int] = None, return_context: bool = False
    ) -> Dict[str, Any]:
        """完整的查询流程

        Args:
            query: 查询文本
            top_k: 返回的结果数量
            return_context: 是否返回上下文信息

        Returns:
            包含回答和可选上下文的字典

        Raises:
            ValueError: 当查询为空时
            Exception: 当查询处理失败时
        """
        try:
            # 搜索和重排序
            reranked_results = self.search_and_rerank(query, top_k)

            # 生成回答
            response = self.generate_response(query, reranked_results)

            result = {
                "query": query,
                "response": response,
                "results_count": len(reranked_results),
            }

            if return_context:
                result["reranked_results"] = reranked_results
                result["context"] = (
                    "\n\n===\n\n".join(
                        [r["document"]["text"] for r in reranked_results]
                    )
                    if reranked_results
                    else ""
                )

            return result

        except Exception as e:
            self.logger.error(f"查询处理失败: {e}")
            raise Exception(f"查询处理失败: {e}")


# 工厂函数
def create_rerank_rag(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection_name: str = "RAG_learn",
    initial_search_limit: int = 10,
    default_top_k: int = 3,
    log_level: str = "INFO",
) -> RerankRAG:
    """创建具有自定义配置的RerankRAG实例的工厂函数

    Args:
        chunk_size: 文本块大小
        chunk_overlap: 块之间的重叠
        collection_name: 向量数据库集合名称
        initial_search_limit: 初始检索数量
        default_top_k: 默认返回结果数量
        log_level: 日志级别

    Returns:
        配置好的RerankRAG实例
    """
    config = RerankRAGConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=collection_name,
        initial_search_limit=initial_search_limit,
        default_top_k=default_top_k,
    )

    # 设置日志记录器
    logger = logging.getLogger("RerankRAG")
    logger.setLevel(getattr(logging, log_level.upper()))

    return RerankRAG(config=config, logger=logger)


# 创建全局服务实例，便于直接导入使用
rerank_rag_service = RerankRAG()


if __name__ == "__main__":
    """
    RerankRAG类的使用示例

    演示如何：
    1. 创建RAG实例
    2. 处理文档
    3. 查询系统
    """

    # 使用自定义配置创建RAG实例
    rag = create_rerank_rag(
        chunk_size=1000,
        chunk_overlap=200,
        collection_name="RAG_learn",
        initial_search_limit=10,
        default_top_k=3,
        log_level="INFO",
    )

    try:
        # 处理文档（取消注释以使用）
        # result = rag.process_document("your_document.md")
        # print(f"文档处理结果: {result}")

        # 查询系统
        query = "什么是思维链？"
        result = rag.query(query, return_context=True)

        print(f"查询: {result['query']}")
        print(f"回答: {result['response']}")
        print(f"结果数量: {result['results_count']}")

    except Exception as e:
        print(f"执行失败: {e}")
