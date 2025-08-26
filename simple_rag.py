from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
import numpy as np


class SimpleRAG:
    def __init__(self, chunk_size=500, overlap=100, batch_size=64):
        """
        初始化SimpleRAG类
        Args:
            chunk_size (int): 每个文本块的最大长度，默认1000
            overlap (int): 相邻块之间的重叠长度，默认200
            batch_size (int): 批处理大小，默认64
        """
        self.client = GeminiLLM()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.text_chunks = []
        self.embeddings = []
        self.system_prompt = "你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"

    def chunk_text(self, text, n=None, overlap=None):
        """
        对文本进行分块处理，确保每个块的长度不超过n，并且相邻块之间有overlap的重叠部分。
        Args:
            text (str): 输入的文本字符串。
            n (int): 每个块的最大长度，如果为None则使用实例的chunk_size。
            overlap (int): 相邻块之间的重叠长度，如果为None则使用实例的overlap。
        """
        if n is None:
            n = self.chunk_size
        if overlap is None:
            overlap = self.overlap

        chunks = []
        for i in range(0, len(text), n - overlap):
            chunks.append(text[i : i + n])
        return chunks

    def cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def semantic_search(self, query, k=5):
        """
        语义搜索，返回与查询最相似的k个文本块
        Args:
            query (str): 查询文本
            k (int): 返回的文本块数量，默认5
        """
        if not self.text_chunks or not self.embeddings:
            raise ValueError("请先加载文档并生成嵌入向量")

        # embed_text 返回的是 List[float]，不需要 [0] 索引
        query_embedding = embedding_service.embed_text(query)
        similarity_scores = []

        for i, chunk_embedding in enumerate(self.embeddings):
            similarity_score = self.cosine_similarity(
                np.array(query_embedding), np.array(chunk_embedding)
            )
            # 确保相似度分数是标量值
            if isinstance(similarity_score, np.ndarray):
                similarity_score = similarity_score.item()
            similarity_scores.append((i, similarity_score))

        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [index for index, _ in similarity_scores[:k]]
        return [self.text_chunks[i] for i in top_indices]

    def load_document(self, file_path):
        """
        加载文档并生成文本块和嵌入向量
        Args:
            file_path (str): 文件路径
        """
        text = file_reader_service.read_file(file_path)
        self.text_chunks = self.chunk_text(text)

        # 生成嵌入向量
        self.embeddings = []
        for i in range(0, len(self.text_chunks), self.batch_size):
            batch = self.text_chunks[i : i + self.batch_size]
            if batch:  # 确保批次不为空
                self.embeddings.extend(embedding_service.embed_texts(batch))

    def generate_response(self, user_message, system_prompt=None):
        """
        生成回答
        Args:
            user_message (str): 用户消息
            system_prompt (str): 系统提示，如果为None则使用实例的system_prompt
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        response = self.client.generate_text(
            user_message, system_instruction=system_prompt
        )
        return response

    def query(self, question, k=5, return_chunks=False):
        """
        查询问题并生成回答
        Args:
            question (str): 用户问题
            k (int): 检索的文本块数量，默认5
            return_chunks (bool): 是否返回检索到的文本块，默认False
        """
        # 检索相关文本块
        top_chunks = self.semantic_search(question, k=k)

        # 构建用户提示
        context_parts = []
        for i, chunk in enumerate(top_chunks):
            context_parts.append(f"上下文内容{i + 1}:\n{chunk}")

        user_prompt = "\n\n".join(context_parts) + f"\n\n用户问题：{question}"

        # 生成回答
        response = self.generate_response(user_prompt)

        if return_chunks:
            return response, top_chunks
        return response

    def set_system_prompt(self, prompt):
        """设置系统提示"""
        self.system_prompt = prompt


# 示例用法
if __name__ == "__main__":
    # 创建SimpleRAG实例
    rag = SimpleRAG()

    # 加载文档
    file_path = "test.pdf"
    rag.load_document(file_path)

    # 查询问题
    query = "什么是‘可解释人工智能’，为什么它被认为很重要？"
    response = rag.query(query)

    # 打印回答
    print(response)
