from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
import numpy as np
from typing import List, Optional


class SemanticRag:
    """
    语义RAG（检索增强生成）系统

    该类实现了基于语义相似度的文档分块和检索功能，
    通过计算句子间的语义相似度来智能分割文档，
    并提供语义搜索和问答功能。
    """

    def __init__(
        self,
        batch_size: int = 64,
        breakpoint_method: str = "percentile",
        breakpoint_threshold: float = 90,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化SemanticRag实例

        Args:
            batch_size: 批处理大小，用于嵌入向量生成
            breakpoint_method: 断点计算方法 ("percentile", "standard_deviation", "interquartile")
            breakpoint_threshold: 断点阈值
            system_prompt: 系统提示词，用于指导AI回答
        """
        self.client = GeminiLLM()
        self.batch_size = batch_size
        self.breakpoint_method = breakpoint_method
        self.breakpoint_threshold = breakpoint_threshold

        # 设置默认系统提示词
        self.system_prompt = system_prompt or (
            "你是一个AI助手，严格根据给定的上下文进行回答。"
            "如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"
        )

        # 存储处理后的数据
        self.sentences: List[str] = []
        self.sentence_embeddings: List[List[float]] = []
        self.text_chunks: List[str] = []
        self.chunk_embeddings: List[List[float]] = []

    def load_document(self, file_path: str) -> None:
        """
        加载并处理文档

        Args:
            file_path: 文档文件路径
        """
        # 读取文档内容
        text = file_reader_service.read_file(file_path)

        # 分割句子
        self.sentences = text.split("。")
        if self.sentences and not self.sentences[-1].strip():
            self.sentences.pop(-1)

        # 生成句子嵌入向量
        self._generate_sentence_embeddings()

        # 计算语义断点并分块
        self._create_semantic_chunks()

    def _generate_sentence_embeddings(self) -> None:
        """生成句子的嵌入向量"""
        self.sentence_embeddings = []

        for i in range(0, len(self.sentences), self.batch_size):
            batch = self.sentences[i : i + self.batch_size]
            if batch:  # 确保批次不为空
                self.sentence_embeddings.extend(embedding_service.embed_texts(batch))

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            vec1: 第一个向量
            vec2: 第二个向量

        Returns:
            余弦相似度值
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _compute_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        根据相似度列表计算断点

        Args:
            similarities: 相似度列表

        Returns:
            断点索引列表
        """
        if self.breakpoint_method == "percentile":
            # 计算相似度分数的第 X 百分位数
            threshold_value = np.percentile(similarities, self.breakpoint_threshold)
        elif self.breakpoint_method == "standard_deviation":
            # 计算相似度分数的均值和标准差
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

    def _split_into_chunks(self, breakpoints: List[int]) -> List[str]:
        """
        根据断点将句子列表分割成文本块

        Args:
            breakpoints: 断点索引列表

        Returns:
            文本块列表
        """
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

        return chunks

    def _create_semantic_chunks(self) -> None:
        """创建语义文本块"""
        if len(self.sentence_embeddings) < 2:
            # 如果句子太少，直接作为一个块
            self.text_chunks = ["。".join(self.sentences)]
        else:
            # 计算相邻句子间的相似度
            similarities = [
                self._cosine_similarity(
                    self.sentence_embeddings[i], self.sentence_embeddings[i + 1]
                )
                for i in range(len(self.sentence_embeddings) - 1)
            ]

            # 计算断点
            breakpoints = self._compute_breakpoints(similarities)

            # 分割文本块
            self.text_chunks = self._split_into_chunks(breakpoints)

        # 为文本块生成嵌入向量
        self.chunk_embeddings = [
            embedding_service.embed_text(chunk) for chunk in self.text_chunks
        ]

    def semantic_search(self, query: str, k: int = 5) -> List[str]:
        """
        执行语义搜索

        Args:
            query: 查询文本
            k: 返回的最相关文本块数量

        Returns:
            最相关的文本块列表
        """
        if not self.text_chunks or not self.chunk_embeddings:
            raise ValueError("请先加载文档")

        query_embedding = embedding_service.embed_text(query)
        similarity_scores = [
            self._cosine_similarity(query_embedding, emb)
            for emb in self.chunk_embeddings
        ]
        top_k = np.argsort(similarity_scores)[-k:][::-1]
        return [self.text_chunks[i] for i in top_k]

    def generate_response(self, query: str, k: int = 3) -> str:
        """
        生成基于检索内容的回答

        Args:
            query: 用户查询
            k: 检索的文本块数量

        Returns:
            AI生成的回答
        """
        # 检索相关文本块
        top_chunks = self.semantic_search(query, k)

        # 构建用户提示
        user_prompt = (
            "\n".join(
                [f"上下文内容{i + 1}:\n{chunk}" for i, chunk in enumerate(top_chunks)]
            )
            + f"\n\n用户问题：{query}"
        )

        # 生成回答
        response = self.client.generate_text(
            user_prompt, system_instruction=self.system_prompt
        )
        return response

    def set_system_prompt(self, prompt: str) -> None:
        """
        设置系统提示词

        Args:
            prompt: 新的系统提示词
        """
        self.system_prompt = prompt


# 为了保持向后兼容性，提供一个使用示例
if __name__ == "__main__":
    # 创建SemanticRag实例
    rag = SemanticRag()

    # 加载文档
    rag.load_document("test.pdf")

    # 执行查询
    query = "什么是'可解释人工智能'，为什么它被认为很重要？"
    response = rag.generate_response(query, k=3)
    print(response)
