import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Dict, Any
from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
from milvus_client import milvus_service
import datetime
import re
import json


class SelfRAGService:
    """
    Self-RAG (Self-Reflective Retrieval-Augmented Generation) 服务类

    实现自反思检索增强生成，包括：
    - 智能判断是否需要检索
    - 文档相关性评估
    - 回答质量和支持度评估
    - 多候选回答评分与选择
    """

    def __init__(
        self,
        llm_client: Optional[GeminiLLM] = None,
        embedding_service=None,
        milvus_service=None,
        file_reader_service=None,
        default_collection: str = "RAG_learn",
        default_chunk_size: int = 1000,
        default_overlap: int = 200,
        max_context_length: int = 2000,
        support_weights: Optional[Dict[str, int]] = None,
    ):
        """
        初始化 Self-RAG 服务

        Args:
            llm_client: LLM 客户端实例，默认创建 GeminiLLM
            embedding_service: 嵌入服务实例
            milvus_service: Milvus 向量数据库服务实例
            file_reader_service: 文件读取服务实例
            default_collection: 默认集合名称
            default_chunk_size: 默认分块大小
            default_overlap: 默认重叠字符数
            max_context_length: 最大上下文长度
            support_weights: 支持度权重配置
        """
        # 依赖注入，如果没有提供则使用默认实例
        self.llm_client = llm_client or GeminiLLM()
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.file_reader_service = file_reader_service

        # 配置参数
        self.default_collection = default_collection
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.max_context_length = max_context_length

        # 支持度权重配置
        self.support_weights = support_weights or {
            "fully supported": 3,
            "partially supported": 1,
            "no support": 0,
        }

        # 系统提示词模板
        self._init_system_prompts()

    def _init_system_prompts(self):
        """初始化系统提示词模板"""
        self.retrieval_prompt = """你是一个判断查询是否需要检索的AI助手。
针对事实性问题、具体信息请求或关于事件、人物、概念的查询，回答"Yes"。
对于观点类、假设性场景或常识性简单查询，回答"No"。
仅回答"Yes"或"No"。"""

        self.relevance_prompt = """你是一个AI助手，任务是判断文档是否与查询相关。
判断文档中是否包含有助于回答查询的信息。
仅回答"Relevant"或"Irrelevant"。"""

        self.support_prompt = """你是一个AI助手，任务是判断回答是否基于给定的上下文。
评估响应中的事实、主张和信息是否由上下文支持。
仅回答以下三个选项之一：
- "Fully supported"（完全支持）：回答所有信息均可从上下文直接得出。
- "Partially supported"（部分支持）：回答中的部分信息由上下文支持，但部分不是。
- "No support"（无支持）：回答中包含大量未在上下文中找到、提及或与上下文矛盾的信息。"""

        self.utility_prompt = """你是一个AI助手，任务是评估一个回答对查询的实用性。
从回答准确性、完整性、正确性和帮助性进行综合评分。
使用1-5级评分标准：
- 1：毫无用处
- 2：稍微有用
- 3：中等有用
- 4：非常有用
- 5：极其有用
仅回答一个从1到5的单个数字，不要过多解释。"""

        self.generation_prompt = (
            """你是一个有帮助的AI助手。请针对查询提供清晰、准确且信息丰富的回答。"""
        )

    def chunk_text(
        self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[str]:
        """
        将文本分割为重叠的块

        Args:
            text: 要分割的文本
            chunk_size: 每个块的字符数，默认使用实例配置
            overlap: 块之间的重叠字符数，默认使用实例配置

        Returns:
            文本块列表

        Raises:
            ValueError: 当文本为空或参数无效时
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap

        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("无效的分块参数")

        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def process_document(
        self,
        file_path: str,
        collection_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        处理文档并存储到向量数据库

        Args:
            file_path: 文件路径
            collection_name: 集合名称，默认使用实例配置
            chunk_size: 分块大小，默认使用实例配置
            overlap: 重叠大小，默认使用实例配置

        Returns:
            文档块列表

        Raises:
            ValueError: 当服务依赖未配置时
            FileNotFoundError: 当文件不存在时
        """
        if not self.file_reader_service:
            raise ValueError("文件读取服务未配置")
        if not self.embedding_service:
            raise ValueError("嵌入服务未配置")
        if not self.milvus_service:
            raise ValueError("Milvus服务未配置")

        collection_name = collection_name or self.default_collection

        # 读取文档
        text = self.file_reader_service.read_file(file_path)

        # 分块
        chunks = self.chunk_text(text, chunk_size, overlap)

        # 生成嵌入向量
        chunk_embeddings = self.embedding_service.embed_chunks(chunks)

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
            self.milvus_service.insert_data(collection_name, data)

        return chunks

    def determine_if_retrieval(self, query: str) -> bool:
        """
        判断查询是否需要检索

        Args:
            query: 用户查询

        Returns:
            是否需要检索的布尔值
        """
        if not query or not query.strip():
            return False

        user_prompt = f"查询: {query}\n\n准确回答此查询是否需要检索？"

        response = self.llm_client.generate_text(
            prompt=user_prompt, system_instruction=self.retrieval_prompt, temperature=0
        )

        answer = response.strip().lower()
        return "yes" in answer

    def evaluate_relevance(self, query: str, context: str) -> str:
        """
        评估文档与查询的相关性

        Args:
            query: 用户查询
            context: 文档上下文

        Returns:
            相关性评估结果（"relevant" 或 "irrelevant"）
        """
        if not query or not context:
            return "irrelevant"

        # 截断过长的上下文
        if len(context) > self.max_context_length:
            context = context[: self.max_context_length] + "... [truncated]"

        user_prompt = f"""查询: {query}
文档内容:
{context}

该文档与查询相关？仅回答"Relevant"或"Irrelevant"。"""

        response = self.llm_client.generate_text(
            prompt=user_prompt, system_instruction=self.relevance_prompt, temperature=0
        )

        return response.strip().lower()

    def assess_support(self, response: str, context: str) -> str:
        """
        评估回答的支持程度

        Args:
            response: 生成的回答
            context: 上下文

        Returns:
            支持程度评估结果
        """
        if not response or not context:
            return "no support"

        # 截断过长的上下文
        if len(context) > self.max_context_length:
            context = context[: self.max_context_length] + "... [truncated]"

        user_prompt = f"""上下文:
{context}

回答:
{response}

该回答与上下文的支持程度如何？仅回答 "Fully supported"、"Partially supported"或 "No support"。"""

        response_eval = self.llm_client.generate_text(
            prompt=user_prompt, system_instruction=self.support_prompt, temperature=0
        )

        return response_eval.strip().lower()

    def rate_utility(self, query: str, response: str) -> int:
        """
        评估回答的实用性

        Args:
            query: 用户查询
            response: 生成的回答

        Returns:
            实用性评分（1-5）
        """
        if not query or not response:
            return 3

        user_prompt = f"""查询: {query}
回答:
{response}

请用1到5分的评分评估该回答的效用，仅用一个1-5的数字评分。"""

        response_eval = self.llm_client.generate_text(
            prompt=user_prompt, system_instruction=self.utility_prompt, temperature=0
        )

        rating = response_eval.strip()
        rating_match = re.search(r"[1-5]", rating)

        return int(rating_match.group(0)) if rating_match else 3

    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        """
        生成回答

        Args:
            query: 用户查询
            context: 可选的上下文信息

        Returns:
            生成的回答
        """
        if not query:
            return ""

        if context:
            user_prompt = f"""上下文:
{context}

查询: {query}

请基于提供的上下文回答该查询。"""
        else:
            user_prompt = f"""查询: {query}

请尽你所能回答该查询。"""

        response = self.llm_client.generate_text(
            prompt=user_prompt,
            system_instruction=self.generation_prompt,
            temperature=0.2,
        )

        return response.strip()

    def _calculate_overall_score(self, support_rating: str, utility_rating: int) -> int:
        """
        计算总体评分

        Args:
            support_rating: 支持程度评级
            utility_rating: 实用性评级

        Returns:
            总体评分
        """
        support_score = self.support_weights.get(support_rating, 0)
        return support_score * 5 + utility_rating

    def self_rag(
        self,
        query: str,
        top_k: int = 3,
        collection_name: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        执行 Self-RAG 流程

        Args:
            query: 用户查询
            top_k: 检索的文档数量
            collection_name: 集合名称，默认使用实例配置
            verbose: 是否输出详细信息

        Returns:
            包含查询、回答和指标的字典

        Raises:
            ValueError: 当必要服务未配置时
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")

        collection_name = collection_name or self.default_collection

        # 判断是否需要检索
        retrieval_needed = self.determine_if_retrieval(query)
        if verbose:
            print(f"是否需要检索: {retrieval_needed}")

        # 初始化指标
        metrics = {
            "retrieval_needed": retrieval_needed,
            "documents_retrieved": 0,
            "relevant_documents": 0,
            "response_support_ratings": [],
            "utility_ratings": [],
        }

        best_response = None
        best_score = -1

        if retrieval_needed:
            if not self.milvus_service:
                raise ValueError("检索需要Milvus服务，但未配置")

            # 检索文档
            results = self.milvus_service.search_by_text(
                collection_name,
                query,
                top_k,
                output_fields=["text", "metadata", "score"],
                metric_type="COSINE",
            )

            metrics["documents_retrieved"] = len(results)
            relevant_contexts = []

            # 评估相关性
            for i, result in enumerate(results):
                context = result["text"]
                relevance = self.evaluate_relevance(query, context)
                if verbose:
                    print(f"文档 {i} 的相关度: {relevance}")

                if relevance == "relevant":
                    relevant_contexts.append(context)

            metrics["relevant_documents"] = len(relevant_contexts)

            # 为每个相关上下文生成和评估回答
            if relevant_contexts:
                for context in relevant_contexts:
                    response = self.generate_response(query, context)

                    # 评估支持程度
                    support_rating = self.assess_support(response, context)
                    if verbose:
                        print(f"支持评级: {support_rating}")
                    metrics["response_support_ratings"].append(support_rating)

                    # 评估实用性
                    utility_rating = self.rate_utility(query, response)
                    if verbose:
                        print(f"效用评级: {utility_rating}/5")
                    metrics["utility_ratings"].append(utility_rating)

                    # 计算总体评分
                    overall_score = self._calculate_overall_score(
                        support_rating, utility_rating
                    )
                    if verbose:
                        print(f"总得分: {overall_score}")

                    # 更新最佳回答
                    if overall_score > best_score:
                        best_response = response
                        best_score = overall_score
                        if verbose:
                            print("找到最佳响应")

            # 如果没有相关上下文或评分过低，生成无上下文回答
            if not relevant_contexts or best_score <= 0:
                best_response = self.generate_response(query)
        else:
            # 不需要检索，直接生成回答
            best_response = self.generate_response(query)

        # 完善指标
        metrics["best_score"] = best_score
        metrics["used_retrieval"] = retrieval_needed and best_score > 0

        return {
            "query": query,
            "response": best_response,
            "metrics": metrics,
        }


# 服务实例
self_rag_service = SelfRAGService(
    embedding_service=embedding_service,
    milvus_service=milvus_service,
    file_reader_service=file_reader_service,
)


def run_self_rag():
    """运行Self-RAG示例"""
    query = "思维链是什么？举个例子"

    result = self_rag_service.self_rag(query)
    print(result["response"])
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_self_rag()
