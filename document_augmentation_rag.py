import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from LLM import GeminiLLM
from file_reader import file_reader_service, FileReader
from embedding import embedding_service, EmbeddingClient
import re
from milvus_client import milvus_service


@dataclass
class ProcessingConfig:
    """文档处理配置参数"""

    chunk_size: int = 1000  # 文本块大小
    overlap: int = 200  # 重叠长度
    questions_per_chunk: int = 5  # 每个块生成的问题数量
    batch_size: int = 64  # 批处理大小
    collection_name: str = "RAG_learn"  # 向量数据库集合名称


@dataclass
class ProcessingStats:
    """处理统计信息"""

    total_chunks: int = 0  # 总块数
    total_questions: int = 0  # 总问题数
    processing_time: float = 0.0  # 处理时间
    failed_chunks: int = 0  # 失败的块数
    failed_questions: int = 0  # 失败的问题数


class DocumentAugmentationRAG:
    """
    文档增强RAG系统

    这个类提供了完整的文档增强RAG流水线：
    1. 文档读取和文本分块
    2. 使用LLM为每个块生成问题
    3. 批量生成嵌入向量
    4. 向量数据库存储
    5. 语义搜索和上下文准备
    6. 基于检索上下文的响应生成
    """

    def __init__(
        self,
        llm_client: Optional[GeminiLLM] = None,
        file_reader: Optional[FileReader] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        milvus_client=None,
        config: Optional[ProcessingConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化文档增强RAG系统

        Args:
            llm_client: LLM客户端，用于问题生成和响应生成
            file_reader: 文件读取服务
            embedding_client: 嵌入服务，用于向量生成
            milvus_client: 向量数据库客户端
            config: 处理配置参数
            logger: 日志记录器
        """
        # 使用默认服务或提供的服务初始化
        self.llm_client = llm_client or GeminiLLM()
        self.file_reader = file_reader or file_reader_service
        self.embedding_client = embedding_client or embedding_service
        self.milvus_client = milvus_client or milvus_service

        # 配置
        self.config = config or ProcessingConfig()

        # 日志设置
        self.logger = logger or self._setup_logger()

        # 系统提示词
        self.question_generation_prompt = (
            "你是一个从文本中生成相关问题的专家。能够根据用户提供的文本生成可回答的简洁问题，"
            "重点聚焦核心信息和关键概念。"
        )

        self.response_generation_prompt = (
            "你是一个AI助手，严格根据给定的上下文进行回答。"
            "如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"
        )

    def _setup_logger(self) -> logging.Logger:
        """设置默认日志记录器"""
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
        """
        将文本分割成重叠的块

        Args:
            text: 输入文本

        Returns:
            文本块列表

        Raises:
            ValueError: 如果文本为空或配置参数无效
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        if self.config.chunk_size <= 0:
            raise ValueError("块大小必须为正数")

        if self.config.overlap < 0 or self.config.overlap >= self.config.chunk_size:
            raise ValueError("重叠长度必须为非负数且小于块大小")

        chunks = []
        step = self.config.chunk_size - self.config.overlap

        for i in range(0, len(text), step):
            chunk = text[i : i + self.config.chunk_size]
            if chunk.strip():  # 只添加非空块
                chunks.append(chunk)

        self.logger.debug(f"从长度为{len(text)}的文本创建了{len(chunks)}个块")
        return chunks

    def generate_questions(self, text_chunk: str) -> List[str]:
        """
        为给定的文本块生成问题

        Args:
            text_chunk: 要生成问题的文本块

        Returns:
            生成的问题列表

        Raises:
            ValueError: 如果文本块为空
            Exception: 如果LLM生成失败
        """
        if not text_chunk or not text_chunk.strip():
            raise ValueError("文本块不能为空")

        user_prompt = f"""
        请根据以下文本内容生成{self.config.questions_per_chunk}个不同的、仅能通过该文本内容回答的问题：

        {text_chunk}

        请严格按以下格式回复：
        1. 带编号的问题列表(1.,2....)
        2. 仅包含问题
        3. 不要添加任何其他内容
        """

        try:
            response = self.llm_client.generate_text(
                user_prompt, system_instruction=self.question_generation_prompt
            )

            # 使用正则表达式提取问题
            response = response.strip()
            pattern = r"^\d+\.\s*(.*)"
            questions = []

            for line in response.split("\n"):
                line = line.strip()
                if line:
                    match = re.match(pattern, line)
                    if match:
                        question = match.group(1).strip()
                        if question:
                            questions.append(question)

            self.logger.debug(f"为文本块生成了{len(questions)}个问题")
            return questions

        except Exception as e:
            self.logger.error(f"生成问题失败: {e}")
            raise Exception(f"问题生成失败: {e}")

    def _batch_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本的嵌入向量

        Args:
            texts: 要嵌入的文本列表

        Returns:
            嵌入向量列表

        Raises:
            ValueError: 如果文本列表为空
            Exception: 如果嵌入生成失败
        """
        if not texts:
            raise ValueError("文本列表不能为空")

        try:
            embeddings = self.embedding_client.embed_texts(texts)
            self.logger.debug(f"为{len(texts)}个文本生成了嵌入向量")
            return embeddings
        except Exception as e:
            self.logger.error(f"生成嵌入向量失败: {e}")
            raise Exception(f"嵌入向量生成失败: {e}")

    def _store_chunk_data(
        self, chunk: str, embedding: List[float], chunk_index: int
    ) -> None:
        """
        在向量数据库中存储块数据

        Args:
            chunk: 文本块
            embedding: 块的嵌入向量
            chunk_index: 块的索引

        Raises:
            Exception: 如果存储失败
        """
        try:
            self.milvus_client.insert_data(
                collection_name=self.config.collection_name,
                data={
                    "text": chunk,
                    "vector": embedding,
                    "metadata": {"type": "chunk", "index": chunk_index},
                },
            )
        except Exception as e:
            self.logger.error(f"存储块{chunk_index}失败: {e}")
            raise Exception(f"块存储失败: {e}")

    def _store_question_data(
        self,
        question: str,
        embedding: List[float],
        chunk_index: int,
        original_chunk: str,
    ) -> None:
        """
        在向量数据库中存储问题数据

        Args:
            question: 生成的问题
            embedding: 问题的嵌入向量
            chunk_index: 相关块的索引
            original_chunk: 原始文本块

        Raises:
            Exception: 如果存储失败
        """
        try:
            self.milvus_client.insert_data(
                collection_name=self.config.collection_name,
                data={
                    "text": question,
                    "vector": embedding,
                    "metadata": {
                        "type": "question",
                        "chunk_index": chunk_index,
                        "original_chunk": original_chunk,
                    },
                },
            )
        except Exception as e:
            self.logger.error(f"存储问题失败: {e}")
            raise Exception(f"问题存储失败: {e}")

    def process_document(self, file_path: str) -> ProcessingStats:
        """
        通过完整的RAG流水线处理文档

        此方法：
        1. 读取文档
        2. 分块文本
        3. 为每个块生成问题
        4. 批量创建嵌入向量
        5. 将所有内容存储到向量数据库

        Args:
            file_path: 文档文件路径

        Returns:
            包含处理信息的ProcessingStats

        Raises:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果文件格式不支持
            Exception: 如果处理失败
        """
        start_time = time.time()
        stats = ProcessingStats()

        try:
            self.logger.info(f"开始处理文档: {file_path}")

            # 读取文档
            self.logger.info("读取文档...")
            text = self.file_reader.read_file(file_path)

            # 创建文本块
            self.logger.info("创建文本块...")
            text_chunks = self.chunk_text(text)
            stats.total_chunks = len(text_chunks)

            # 批量处理文本块
            self.logger.info(f"处理{len(text_chunks)}个文本块...")

            for i, chunk in enumerate(text_chunks):
                try:
                    # 为此块生成问题
                    questions = self.generate_questions(chunk)

                    # 准备批量嵌入的文本
                    texts_to_embed = [chunk] + questions

                    # 批量生成嵌入向量
                    embeddings = self._batch_embed_texts(texts_to_embed)

                    # 存储块
                    chunk_embedding = embeddings[0]
                    self._store_chunk_data(chunk, chunk_embedding, i)

                    # 存储问题
                    question_embeddings = embeddings[1:]
                    for question, question_embedding in zip(
                        questions, question_embeddings
                    ):
                        self._store_question_data(
                            question, question_embedding, i, chunk
                        )
                        stats.total_questions += 1

                    self.logger.info(f"处理块 {i + 1}/{len(text_chunks)}")

                except Exception as e:
                    self.logger.error(f"处理块{i}失败: {e}")
                    stats.failed_chunks += 1
                    continue

            stats.processing_time = time.time() - start_time
            self.logger.info(
                f"文档处理完成. "
                f"块数: {stats.total_chunks}, 问题数: {stats.total_questions}, "
                f"时间: {stats.processing_time:.2f}秒"
            )

            return stats

        except Exception as e:
            self.logger.error(f"文档处理失败: {e}")
            raise Exception(f"文档处理失败: {e}")

    def search(
        self, query: str, limit: int = 5, collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        使用语义相似度搜索相关文档

        Args:
            query: 搜索查询文本
            limit: 返回的最大结果数
            collection_name: 要搜索的集合（如果为None则使用默认值）

        Returns:
            包含文本和元数据的搜索结果列表

        Raises:
            ValueError: 如果查询为空
            Exception: 如果搜索失败
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")

        collection = collection_name or self.config.collection_name

        try:
            self.logger.debug(f"在集合'{collection}'中搜索: '{query}'")

            search_results = self.milvus_client.search_by_text(
                collection_name=collection,
                text=query,
                limit=limit,
                output_fields=["text", "metadata"],
            )

            self.logger.debug(f"找到{len(search_results)}个结果")
            return search_results

        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            raise Exception(f"搜索失败: {e}")

    def prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        从搜索结果准备用于响应生成的上下文

        此方法结合搜索结果中的块和问题，
        确保即使没有直接检索到的块也会被包含在内（如果被问题引用）。

        Args:
            search_results: 来自向量数据库的搜索结果列表

        Returns:
            组合的上下文字符串
        """
        chunk_indices = set()
        context_chunks = []

        # 第一遍：收集直接的块结果
        for result in search_results:
            metadata = result["entity"]["metadata"]
            if metadata["type"] == "chunk":
                chunk_indices.add(metadata["index"])
                context_chunks.append(result["entity"]["text"])

        # 第二遍：添加问题引用的块
        for result in search_results:
            metadata = result["entity"]["metadata"]
            if metadata["type"] == "question":
                chunk_idx = metadata["chunk_index"]
                if chunk_idx not in chunk_indices:
                    question_text = result["entity"]["text"]
                    original_chunk = metadata["original_chunk"]
                    context_chunks.append(
                        f"(由问题'{question_text}'引用):\n{original_chunk}"
                    )
                    chunk_indices.add(chunk_idx)

        full_context = "\n\n".join(context_chunks)
        self.logger.debug(f"准备了包含{len(context_chunks)}个块的上下文")
        return full_context

    def generate_response(self, query: str, context: str) -> str:
        """
        基于查询和上下文生成响应

        Args:
            query: 用户的问题
            context: 从搜索结果检索的上下文

        Returns:
            生成的响应文本

        Raises:
            ValueError: 如果查询或上下文为空
            Exception: 如果响应生成失败
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")

        if not context or not context.strip():
            raise ValueError("上下文不能为空")

        user_prompt = f"上下文:\n{context}\n\n用户问题：{query}"

        try:
            self.logger.debug("生成响应...")
            response = self.llm_client.generate_text(
                user_prompt, system_instruction=self.response_generation_prompt
            )

            self.logger.debug("响应生成成功")
            return response.strip()

        except Exception as e:
            self.logger.error(f"响应生成失败: {e}")
            raise Exception(f"响应生成失败: {e}")

    def query(
        self, question: str, limit: int = 5, collection_name: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        完整的查询流水线：搜索、准备上下文和生成响应

        Args:
            question: 用户的问题
            limit: 最大搜索结果数
            collection_name: 要搜索的集合（如果为None则使用默认值）

        Returns:
            (响应, 搜索结果, 上下文)的元组

        Raises:
            ValueError: 如果问题为空
            Exception: 如果任何步骤失败
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        try:
            self.logger.info(f"处理查询: '{question}'")

            # 搜索相关文档
            search_results = self.search(question, limit, collection_name)

            # 从结果准备上下文
            context = self.prepare_context(search_results)

            # 生成响应
            response = self.generate_response(question, context)

            self.logger.info("查询处理成功")
            return response, search_results, context

        except Exception as e:
            self.logger.error(f"查询处理失败: {e}")
            raise Exception(f"查询处理失败: {e}")


# 工厂函数和使用示例
def create_document_augmentation_rag(
    chunk_size: int = 1000,
    overlap: int = 200,
    questions_per_chunk: int = 5,
    batch_size: int = 32,
    collection_name: str = "RAG_learn",
    log_level: str = "INFO",
) -> DocumentAugmentationRAG:
    """
    创建具有自定义配置的DocumentAugmentationRAG实例的工厂函数

    Args:
        chunk_size: 文本块大小
        overlap: 块之间的重叠
        questions_per_chunk: 每个块生成的问题数
        batch_size: 嵌入生成的批处理大小
        collection_name: 向量数据库集合名称
        log_level: 日志级别

    Returns:
        配置好的DocumentAugmentationRAG实例
    """
    config = ProcessingConfig(
        chunk_size=chunk_size,
        overlap=overlap,
        questions_per_chunk=questions_per_chunk,
        batch_size=batch_size,
        collection_name=collection_name,
    )

    # 设置日志记录器
    logger = logging.getLogger("DocumentAugmentationRAG")
    logger.setLevel(getattr(logging, log_level.upper()))

    return DocumentAugmentationRAG(config=config, logger=logger)


# 使用示例
if __name__ == "__main__":
    """
    DocumentAugmentationRAG类的使用示例

    演示如何：
    1. 创建RAG实例
    2. 处理文档
    3. 查询系统
    """

    # 使用自定义配置创建RAG实例
    rag = create_document_augmentation_rag(
        chunk_size=1000,
        overlap=200,
        questions_per_chunk=3,
        batch_size=64,
        collection_name="RAG_learn",
        log_level="INFO",
    )

    try:
        # 处理文档（取消注释以使用）
        # stats = rag.process_document("Agent基础.md")
        # print(f"处理完成: {stats}")

        # 查询系统
        query = "思维链(CoT)是什么？举个例子"
        response, search_results, context = rag.query(query, limit=5)

        print(f"查询: {query}")
        print(f"响应: {response}")

    except Exception as e:
        print(f"错误: {e}")
