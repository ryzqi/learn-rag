from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
from milvus_client import milvus_service
import numpy as np
import hashlib
from typing import List, Dict, Any, Optional


class SemanticRag:
    """
    语义RAG（检索增强生成）系统

    该类实现了基于语义相似度的文档分块和检索功能，
    通过计算句子间的语义相似度来智能分割文档，
    并提供语义搜索和问答功能。使用Milvus进行向量存储。
    """

    def __init__(
        self,
        batch_size: int = 64,
        breakpoint_method: str = "percentile",
        breakpoint_threshold: float = 90,
        system_prompt: Optional[str] = None,
        collection_name: str = "RAG_learn",
    ):
        """
        初始化SemanticRag实例

        Args:
            batch_size: 批处理大小，用于嵌入向量生成
            breakpoint_method: 断点计算方法 ("percentile", "standard_deviation", "interquartile")
            breakpoint_threshold: 断点阈值
            system_prompt: 系统提示词，用于指导AI回答
            collection_name: Milvus集合名称，必须指定
        """
        self.llm_client = GeminiLLM()
        self.file_reader = file_reader_service
        self.embedding_client = embedding_service
        self.milvus_client = milvus_service

        self.batch_size = batch_size
        self.breakpoint_method = breakpoint_method
        self.breakpoint_threshold = breakpoint_threshold
        self.collection_name = collection_name

        # 设置默认系统提示词
        self.system_prompt = system_prompt or (
            "你是一个AI助手，严格根据给定的上下文进行回答。"
            "如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"
        )
        # 存储处理后的数据
        self.sentences: List[str] = []
        self.sentence_embeddings: List[List[float]] = []
        self.text_chunks: List[str] = []

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

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _compute_breakpoints(self, similarities: List[float]) -> List[int]:
        """根据相似度列表计算断点"""
        if self.breakpoint_method == "percentile":
            threshold_value = np.percentile(similarities, self.breakpoint_threshold)
        elif self.breakpoint_method == "standard_deviation":
            mean = np.mean(similarities)
            std_dev = np.std(similarities)
            threshold_value = mean - (self.breakpoint_threshold * std_dev)
        elif self.breakpoint_method == "interquartile":
            q1, q3 = np.percentile(similarities, [25, 75])
            threshold_value = q1 - 1.5 * (q3 - q1)
        else:
            raise ValueError(f"不支持的断点计算方法: {self.breakpoint_method}")

        # 找出相似度低于阈值的索引
        return [i for i, sim in enumerate(similarities) if sim < threshold_value]

    def _create_semantic_chunks(self) -> None:
        """创建语义文本块"""
        if len(self.sentence_embeddings) < 2:
            # 如果句子太少，直接作为一个块
            self.text_chunks = ["。".join(self.sentences)]
        else:
            # 计算相邻句子间的相似度
            similarities = []
            for i in range(len(self.sentence_embeddings) - 1):
                try:
                    sim = self._cosine_similarity(
                        self.sentence_embeddings[i], self.sentence_embeddings[i + 1]
                    )
                    similarities.append(sim)
                except Exception:
                    similarities.append(0.5)  # 使用中等相似度作为默认值

            # 计算断点
            breakpoints = self._compute_breakpoints(similarities)

            # 分割文本块
            chunks = []
            start = 0

            for bp in breakpoints:
                chunk = "。".join(self.sentences[start : bp + 1]) + "。"
                chunks.append(chunk)
                start = bp + 1

            # 添加最后一个块
            if start < len(self.sentences):
                chunk = "。".join(self.sentences[start:])
                if chunk.strip():
                    chunks.append(chunk)

            self.text_chunks = chunks

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        处理文档并存储到向量数据库

        Args:
            file_path: 文档文件路径

        Returns:
            处理结果字典
        """
        try:
            # 读取文档内容
            text = self.file_reader.read_file(file_path)

            # 分割句子
            self.sentences = text.split("。")
            if self.sentences and not self.sentences[-1].strip():
                self.sentences.pop(-1)

            # 过滤空句子
            self.sentences = [s.strip() for s in self.sentences if s.strip()]

            if not self.sentences:
                raise ValueError("文档分句后没有有效内容")

            # 批量生成句子嵌入向量
            self.sentence_embeddings = self._batch_embed_texts(self.sentences)

            # 计算语义断点并分块
            self._create_semantic_chunks()

            # 批量生成文本块嵌入向量并存储到Milvus
            chunk_embeddings = self._batch_embed_texts(self.text_chunks)

            # 准备插入数据
            data_to_insert = []
            for i, (chunk, embedding) in enumerate(
                zip(self.text_chunks, chunk_embeddings)
            ):
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
                "sentences_count": len(self.sentences),
                "chunks_count": len(self.text_chunks),
                "file_path": file_path,
                **result,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "file_path": file_path}

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        执行语义搜索

        Args:
            query: 查询文本
            limit: 返回的最相关文本块数量

        Returns:
            最相关的文本块列表
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        try:
            # 使用Milvus的文本搜索功能
            return self.milvus_client.search_by_text(
                collection_name=self.collection_name,
                text=query,
                limit=limit,
                output_fields=["text", "source", "chunk_index"],
                metric_type="COSINE",
                embedding_client=self.embedding_client,
            )

        except Exception as e:
            raise Exception(f"语义搜索失败: {e}")

    def generate_response(self, query: str, context: str) -> str:
        """
        生成基于检索内容的回答

        Args:
            query: 用户查询
            context: 从搜索结果检索的上下文

        Returns:
            AI生成的回答
        """
        if not context or not context.strip():
            return "我没有找到相关的信息来回答这个问题。"

        user_prompt = f"上下文:\n{context}\n\n用户问题：{query}"
        return self.llm_client.generate_text(
            user_prompt, system_instruction=self.system_prompt
        )

    def query(self, question: str, limit: int = 3) -> str:
        """
        完整的查询流水线：搜索、准备上下文和生成响应

        Args:
            question: 用户问题
            limit: 检索的文本块数量

        Returns:
            生成的回答
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        try:
            # 检索相关文本块
            search_results = self.search(question, limit)

            # 构建用户提示
            context_parts = []
            for i, result in enumerate(search_results):
                # 正确访问Milvus搜索结果的结构
                text = (
                    result.get("entity", {}).get("text", "")
                    if "entity" in result
                    else result.get("text", "")
                )
                if text.strip():
                    context_parts.append(f"上下文内容{i + 1}:\n{text}")

            context = "\n\n".join(context_parts)

            # 生成回答
            return self.generate_response(question, context)

        except Exception as e:
            return f"回答生成失败: {e}"

    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """生成文本块的唯一ID"""
        content = f"{file_path}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()


# 示例使用代码（如果直接运行此文件）
if __name__ == "__main__":
    # 创建语义RAG实例
    rag = SemanticRag()

    try:
        # 处理文档（取消注释以使用）
        result = rag.process_document("Agent基础.md")
        print(f"处理结果: {result}")

        # 进行查询
        query = "思维链(CoT)是什么？举个例子"
        response = rag.query(query)

        print(f"查询: {query}")
        print(f"回答: {response}")

    except Exception as e:
        print(f"错误: {e}")
