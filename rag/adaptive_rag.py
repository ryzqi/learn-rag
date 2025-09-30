import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Dict, Any
import datetime
import re

from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
from milvus_client import milvus_service


class AdaptiveRAG:
    """
    自适应检索增强生成(RAG)类

    根据查询类型自动选择最适合的检索策略，支持四种查询类型：
    - Factual: 事实性查询，需要精确信息
    - Analytical: 分析性查询，需要多维度分析
    - Opinion: 观点性查询，需要多元观点
    - Contextual: 上下文相关查询，需要情境理解

    Args:
        llm_client: LLM客户端实例 (GeminiLLM)
        embedding_client: 嵌入服务客户端实例 (EmbeddingClient)
        vector_client: 向量数据库客户端实例 (MilvusClient)
        collection_name: 向量数据库集合名称，默认为"RAG_learn"
        default_chunk_size: 默认文本分块大小，默认为1000
        default_overlap: 默认分块重叠大小，默认为200
        default_k: 默认检索文档数量，默认为4
        default_temperature: 默认生成温度，默认为0.2
    """

    def __init__(
        self,
        llm_client: Optional[GeminiLLM] = None,
        embedding_client=None,
        vector_client=None,
        collection_name: str = "RAG_learn",
        default_chunk_size: int = 1000,
        default_overlap: int = 200,
        default_k: int = 4,
        default_temperature: float = 0.2,
    ):
        """
        初始化自适应RAG实例

        Args:
            llm_client: LLM客户端，如果为None则使用默认GeminiLLM
            embedding_client: 嵌入客户端，如果为None则使用默认embedding_service
            vector_client: 向量数据库客户端，如果为None则使用默认milvus_service
            collection_name: 向量数据库集合名称
            default_chunk_size: 默认文本分块大小
            default_overlap: 默认分块重叠大小
            default_k: 默认检索文档数量
            default_temperature: 默认生成温度
        """
        # 依赖注入，支持默认值
        self.llm = llm_client or GeminiLLM()
        self.embedding = embedding_client or embedding_service
        self.vector_db = vector_client or milvus_service

        # 配置参数
        self.collection_name = collection_name
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.default_k = default_k
        self.default_temperature = default_temperature

    def chunk_text(self, text: str, n: int, overlap: int) -> List[str]:
        """
        将文本分割为重叠的块

        Args:
            text: 要分割的文本
            n: 每个块的字符数
            overlap: 块之间的重叠字符数

        Returns:
            文本块列表
        """
        chunks = []
        for i in range(0, len(text), n - overlap):
            # 添加从当前索引到索引 + 块大小的文本块
            chunk = text[i : i + n]
            if chunk:
                chunks.append(chunk)

        return chunks

    def process_document(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        处理文档，分块并存储到向量数据库

        Args:
            file_path: 文档文件路径
            chunk_size: 文本分块大小，如果为None则使用默认值
            overlap: 分块重叠大小，如果为None则使用默认值

        Returns:
            文档分块列表
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap

        text = file_reader_service.read_file(file_path)
        chunks = self.chunk_text(text, chunk_size, overlap)

        # 生成嵌入向量
        if hasattr(self.embedding, "embed_chunks"):
            chunk_embeddings = self.embedding.embed_chunks(chunks)
        else:
            chunk_embeddings = [self.embedding.embed_text(chunk) for chunk in chunks]

        # 存储到向量数据库
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            data = {
                "text": chunk,
                "embedding": embedding,
                "metadata": {
                    "index": i,
                    "source": file_path,
                    "relevance_score": 0.0,
                    "feedback_count": 0,
                    "created_at": datetime.datetime.now().isoformat(),
                },
            }
            self.vector_db.insert_data(self.collection_name, data)

        return chunks

    def _classify_query(self, query: str) -> str:
        """
        对查询进行分类

        Args:
            query: 查询文本

        Returns:
            查询类型 ("Factual", "Analytical", "Opinion", "Contextual")
        """
        system_prompt = """您是专业的查询分类专家。
        请将给定查询严格分类至以下四类中的唯一一项：
        - Factual：需要具体、可验证信息的查询
        - Analytical：需要综合分析或深入解释的查询
        - Opinion：涉及主观问题或寻求多元观点的查询
        - Contextual：依赖用户具体情境的查询

        请仅返回分类名称，不要添加任何解释或额外文本。
        """

        user_prompt = f"对以下查询进行分类: {query}"

        response = self.llm.generate_text(
            prompt=user_prompt, system_instruction=system_prompt
        )

        category = response.strip()

        valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]

        for valid in valid_categories:
            if valid in category:
                return valid

        return "Factual"

    def _factual_retrieval_strategy(
        self, query: str, k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        事实性检索策略

        Args:
            query: 查询文本
            k: 检索文档数量

        Returns:
            检索结果列表
        """
        print("执行事实性检索策略")
        system_prompt = """您是搜索查询优化专家。
        您的任务是重构给定的事实性查询，使其更精确具体以提升信息检索效果。
        重点关注关键实体及其关联关系。

        请仅提供优化后的查询，不要包含任何解释。
        """

        user_prompt = f"请优化此事实性查询: {query}"

        response = self.llm.generate_text(
            prompt=user_prompt, system_instruction=system_prompt, temperature=0
        )

        enhanced_query = response.strip()

        query_embedding = self.embedding.embed_text(enhanced_query)

        initial_results = self.vector_db.search_data(
            self.collection_name,
            query_embedding,
            limit=k * 2,
            metric_type="COSINE",
            output_fields=["text", "metadata", "score"],
        )

        ranked_results = []

        for doc in initial_results:
            relevance_score = self._score_document_relevance(
                enhanced_query, doc["text"]
            )

            ranked_results.append(
                {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "similarity": doc["score"],
                    "relevance_score": relevance_score,
                }
            )

        ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return ranked_results[:k]

    def _analytical_retrieval_strategy(
        self, query: str, k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        分析性检索策略

        Args:
            query: 查询文本
            k: 检索文档数量

        Returns:
            检索结果列表
        """
        print("执行分析性检索策略")

        system_prompt = """您是复杂问题拆解专家。
        请针对给定的分析性查询生成探索不同维度的子问题。
        这些子问题应覆盖主题的广度并帮助获取全面信息。

        请严格生成恰好3个子问题，每个问题单独一行。
        """

        user_prompt = f"请为此分析性查询生成子问题：{query}"

        response = self.llm.generate_text(
            prompt=user_prompt, system_instruction=system_prompt, temperature=0.3
        )

        sub_queries = response.strip().split("\n")
        sub_queries = [q.strip() for q in sub_queries if q.strip()]

        print(f"生成的子问题: {sub_queries}")

        all_results = []
        for sub_query in sub_queries:
            results = self.vector_db.search_by_text(
                self.collection_name,
                sub_query,
                limit=2,
                metric_type="cosine",
                output_fields=["text", "metadata", "score"],
            )
            all_results.extend(results)

        unique_texts = set()
        diverse_results = []

        for result in all_results:
            if result["text"] not in unique_texts:
                unique_texts.add(result["text"])
                diverse_results.append(result)

        if len(diverse_results) < k:
            main_query_embedding = self.embedding.embed_text(query)

            main_results = self.vector_db.search_data(
                self.collection_name,
                main_query_embedding,
                limit=k,
                metric_type="cosine",
                output_fields=["text", "metadata", "score"],
            )

            for result in main_results:
                if result["text"] not in unique_texts:
                    unique_texts.add(result["text"])
                    diverse_results.append(result)

        return diverse_results[:k]

    def _opinion_retrieval_strategy(
        self, query: str, k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        观点性检索策略

        Args:
            query: 查询文本
            k: 检索文档数量

        Returns:
            检索结果列表
        """
        print("执行观点性检索策略")

        system_prompt = """您是主题多视角分析专家。
        针对给定的观点类或意见类查询，请识别人们可能持有的不同立场或观点。

        请严格返回恰好3个不同观点角度，每个角度单独一行。
        """

        user_prompt = f"请识别以下主题的不同观点：{query}"

        response = self.llm.generate_text(
            prompt=user_prompt, system_instruction=system_prompt, temperature=0.3
        )

        viewpoints = response.strip().split("\n")
        viewpoints = [v.strip() for v in viewpoints if v.strip()]

        print(f"生成的观点: {viewpoints}")

        all_results = []
        for viewpoint in viewpoints:
            combined_query = f"{query} {viewpoint}"
            viewpoint_embedding = self.embedding.embed_text(combined_query)
            results = self.vector_db.search_data(
                self.collection_name,
                viewpoint_embedding,
                limit=2,
                metric_type="cosine",
                output_fields=["text", "metadata", "score"],
            )

            for result in results:
                result["viewpoint"] = viewpoint

            all_results.extend(results)

        selected_results = []
        for viewpoint in viewpoints:
            viewpoints_docs = [
                r for r in all_results if r.get("viewpoint") == viewpoint
            ]
            if viewpoints_docs:
                selected_results.append(viewpoints_docs[0])

        remaining_slots = k - len(selected_results)
        if remaining_slots > 0:
            remaining_docs = [r for r in all_results if r not in selected_results]
            remaining_docs.sort(key=lambda x: x["score"], reverse=True)
            selected_results.extend(remaining_docs[:remaining_slots])

        return selected_results[:k]

    def _contextual_retrieval_strategy(
        self, query: str, k: int = 4, user_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        上下文检索策略

        Args:
            query: 查询文本
            k: 检索文档数量
            user_context: 用户提供的上下文，如果为None则自动推断

        Returns:
            检索结果列表
        """
        print("执行上下文检索策略")
        if not user_context:
            system_prompt = """您是理解查询隐含上下文的专家。
            对于给定的查询，请推断可能相关或隐含但未明确说明的上下文信息。
            重点关注有助于回答该查询的背景信息。

            请简要描述推断的隐含上下文。
            """

            user_prompt = f"推断此查询中的隐含背景(上下文)：{query}"

            response = self.llm.generate_text(
                prompt=user_prompt, system_instruction=system_prompt, temperature=0.1
            )

            user_context = response.strip()
            print(f"推断出的上下文: {user_context}")

        system_prompt = """您是上下文整合式查询重构专家。
        根据提供的查询和上下文信息，请重新构建更具体的查询以整合上下文，从而获取更相关的信息。

        请仅返回重新构建的查询，不要包含任何解释。
        """

        user_prompt = f"""
        原始查询：{query}
        关联上下文：{user_context}

        请结合此上下文重新构建查询：
        """

        response = self.llm.generate_text(
            prompt=user_prompt, system_instruction=system_prompt, temperature=0
        )

        contextualized_query = response.strip()

        print(f"重新构建的查询: {contextualized_query}")

        initial_results = self.vector_db.search_by_text(
            self.collection_name,
            query,
            limit=k * 2,
            metric_type="COSINE",
            output_fields=["text", "metadata", "score"],
        )

        ranked_results = []

        for doc in initial_results:
            context_relevance = self._score_document_context_relevance(
                query, user_context, doc["text"]
            )

            ranked_results.append(
                {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "similarity": doc["score"],
                    "context_relevance": context_relevance,
                }
            )

        ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
        return ranked_results[:k]

    def _score_document_relevance(self, query: str, document: str) -> float:
        """
        评估文档与查询的相关性

        Args:
            query: 查询文本
            document: 文档文本

        Returns:
            相关性评分 (0-10)
        """
        system_prompt = """您是文档相关性评估专家。
        请根据文档与查询的匹配程度给出0到10分的评分：
        0 = 完全无关
        10 = 完美契合查询

        请仅返回一个0到10之间的数字评分，不要包含任何其他内容。
        """

        doc_preview = document[:1500] + "..." if len(document) > 1500 else document

        user_prompt = f"""
        查询: {query}

        文档: {doc_preview}

        相关性评分（0-10）：
        """

        response = self.llm.generate_text(
            prompt=user_prompt, system_instruction=system_prompt, temperature=0
        )

        score_text = response.strip()

        match = re.search(r"(\d+(\.\d+)?)", score_text)

        if match:
            score = float(match.group(1))
            return min(10, max(0, score))
        else:
            return 5.0

    def _score_document_context_relevance(
        self, query: str, context: str, document: str
    ) -> float:
        """
        结合上下文评估文档相关性

        Args:
            query: 查询文本
            context: 上下文信息
            document: 文档文本

        Returns:
            上下文相关性评分 (0-10)
        """
        system_prompt = """您是结合上下文评估文档相关性的专家。
        请根据文档在给定上下文中对查询的响应质量，给出0到10分的评分：
        0 = 完全无关
        10 = 在给定上下文中完美契合查询

        请严格仅返回一个0到10之间的数字评分，不要包含任何其他内容。
        """

        # 如果文档过长，则截断文档
        doc_preview = document[:1500] + "..." if len(document) > 1500 else document

        # 包含查询、上下文和文档预览的用户提示
        user_prompt = f"""
        待评估查询：{query}
        关联上下文：{context}

        文档内容预览：
        {doc_preview}

        结合上下文的相关性评分（0-10）：
        """

        response = self.llm.generate_text(
            prompt=user_prompt, system_instruction=system_prompt, temperature=0
        )

        score_text = response.strip()

        match = re.search(r"(\d+(\.\d+)?)", score_text)

        if match:
            score = float(match.group(1))
            return min(10, max(0, score))
        else:
            return 5.0

    def _adaptive_retrieval(
        self, query: str, k: Optional[int] = None, user_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        自适应检索：根据查询类型选择最适合的检索策略

        Args:
            query: 查询文本
            k: 检索文档数量，如果为None则使用默认值
            user_context: 用户上下文，仅在上下文检索策略中使用

        Returns:
            检索结果列表
        """
        k = k or self.default_k
        query_type = self._classify_query(query)
        print(f"查询类型: {query_type}")

        match query_type:
            case "Factual":
                results = self._factual_retrieval_strategy(query, k)
            case "Analytical":
                results = self._analytical_retrieval_strategy(query, k)
            case "Opinion":
                results = self._opinion_retrieval_strategy(query, k)
            case "Contextual":
                results = self._contextual_retrieval_strategy(query, k, user_context)
            case _:
                results = self._factual_retrieval_strategy(query, k)
        return results

    def _generate_response(
        self, query: str, results: List[Dict[str, Any]], query_type: str
    ) -> str:
        """
        根据查询类型和检索结果生成响应

        Args:
            query: 查询文本
            results: 检索结果列表
            query_type: 查询类型

        Returns:
            生成的响应文本
        """
        context = "\n\n---\n\n".join([r["text"] for r in results])

        if query_type == "Factual":
            system_prompt = """您是基于事实信息应答的AI助手。
        请严格根据提供的上下文回答问题，确保信息准确无误。
        若上下文缺乏必要信息，请明确指出信息局限。"""

        elif query_type == "Analytical":
            system_prompt = """您是专业分析型AI助手。
        请基于提供的上下文，对主题进行多维度深度解析：
        - 涵盖不同层面的关键要素（不同方面和视角）
        - 整合多方观点形成系统分析
        若上下文存在信息缺口或空白，请在分析时明确指出信息短缺。"""

        elif query_type == "Opinion":
            system_prompt = """您是观点整合型AI助手。
        请基于提供的上下文，结合以下标准给出不同观点：
        - 全面呈现不同立场观点
        - 保持各观点表述的中立平衡，避免出现偏见
        - 当上下文视角有限时，直接说明"""

        elif query_type == "Contextual":
            system_prompt = """您是情境上下文感知型AI助手。
        请结合查询背景与上下文信息：
        - 建立问题情境与文档内容的关联
        - 当上下文无法完全匹配具体情境时，请明确说明适配性限制"""

        else:
            system_prompt = (
                """您是通用型AI助手。请基于上下文回答问题，若信息不足请明确说明。"""
            )

        user_prompt = f"""
        上下文:
        {context}

        问题: {query}

        请基于上下文提供专业可靠的回答。
        """

        response = self.llm.generate_text(
            prompt=user_prompt,
            system_instruction=system_prompt,
            temperature=self.default_temperature,
        )

        return response

    def query(
        self,
        query_text: str,
        k: Optional[int] = None,
        user_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的自适应RAG查询流程

        Args:
            query_text: 查询文本
            k: 检索文档数量，如果为None则使用默认值
            user_context: 用户上下文信息，仅在上下文查询类型中使用

        Returns:
            包含查询结果的字典，包含以下字段：
            - query: 原始查询
            - query_type: 查询类型
            - retrieved_docs: 检索到的文档列表
            - response: 生成的回答
        """
        print("\n=== RAG WITH ADAPTIVE RETRIEVAL ===")

        k = k or self.default_k
        query_type = self._classify_query(query_text)
        retrieved_docs = self._adaptive_retrieval(query_text, k, user_context)
        response = self._generate_response(query_text, retrieved_docs, query_type)

        result = {
            "query": query_text,
            "query_type": query_type,
            "retrieved_docs": retrieved_docs,
            "response": response,
        }

        return result


if __name__ == "__main__":
    # 使用新的类接口
    adaptive_rag = AdaptiveRAG()
    result = adaptive_rag.query("思维链(CoT)是什么？举个例子")
    print(result)
