"""
CRAG (Corrective Retrieval-Augmented Generation) 实现

纠错型RAG系统，通过动态评估检索文档的相关性，并根据评估结果自动选择最佳知识来源策略。

核心特性：
- 检索文档相关性评估（0-1分数）
- 三级策略切换：高相关性（文档only）、低相关性（网络搜索only）、中等相关性（混合+提炼）
- 查询重写优化
- 知识提炼避免冗余
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import re
from typing import List, Dict, Any, Optional, Tuple

from LLM import GeminiLLM
from embedding import embedding_service
from milvus_client import milvus_service
from file_reader import file_reader_service


class CRAG:
    """
    纠错型检索增强生成（Corrective RAG）类

    通过评估检索文档的相关性，动态选择知识来源策略：
    - 高相关性(>0.7)：直接使用文档内容
    - 低相关性(<0.3)：使用网络搜索
    - 中等相关性(0.3-0.7)：结合文档与网络搜索，并提炼关键信息

    Args:
        llm_client: LLM客户端实例 (GeminiLLM)
        embedding_client: 嵌入服务客户端实例 (EmbeddingClient)
        vector_client: 向量数据库客户端实例 (MilvusClient)
        collection_name: 向量数据库集合名称，默认为"RAG_learn"
        default_chunk_size: 默认文本分块大小，默认为1000
        default_overlap: 默认分块重叠大小，默认为200
        default_k: 默认检索文档数量，默认为3
        high_relevance_threshold: 高相关性阈值，默认为0.7
        low_relevance_threshold: 低相关性阈值，默认为0.3
        max_context_length: 最大上下文长度，用于截断过长文档，默认为1500
        enable_web_search: 是否启用网络搜索，默认为True
    """

    def __init__(
        self,
        llm_client: Optional[GeminiLLM] = None,
        embedding_client=None,
        vector_client=None,
        collection_name: str = "RAG_learn",
        default_chunk_size: int = 1000,
        default_overlap: int = 200,
        default_k: int = 3,
        high_relevance_threshold: float = 0.7,
        low_relevance_threshold: float = 0.3,
        max_context_length: int = 1500,
        enable_web_search: bool = True,
    ):
        """
        初始化CRAG实例

        Args:
            llm_client: LLM客户端，如果为None则使用默认GeminiLLM
            embedding_client: 嵌入客户端，如果为None则使用默认embedding_service
            vector_client: 向量数据库客户端，如果为None则使用默认milvus_service
            collection_name: 向量数据库集合名称
            default_chunk_size: 默认文本分块大小
            default_overlap: 默认分块重叠大小
            default_k: 默认检索文档数量
            high_relevance_threshold: 高相关性阈值
            low_relevance_threshold: 低相关性阈值
            max_context_length: 最大上下文长度
            enable_web_search: 是否启用网络搜索
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
        self.high_relevance_threshold = high_relevance_threshold
        self.low_relevance_threshold = low_relevance_threshold
        self.max_context_length = max_context_length
        self.enable_web_search = enable_web_search

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

        # 读取文件
        text = file_reader_service.read_file(file_path)

        # 分块
        chunks = self.chunk_text(text, chunk_size, overlap)

        # 生成嵌入向量
        print("正在为文本块生成嵌入...")
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

        print(f"已处理并存储 {len(chunks)} 个文本块")
        return chunks

    def evaluate_document_relevance(self, query: str, document: str) -> float:
        """
        评估文档与查询的相关性

        使用LLM对文档内容与查询的匹配程度进行评分。

        Args:
            query: 用户查询
            document: 文档文本

        Returns:
            相关性评分（0到1之间的浮点数）
            - 0: 完全不相关
            - 1: 完全相关
            - 失败时返回0.5（中等相关性）
        """
        system_prompt = """你是一位评估文档相关性的专家。
请在0到1的范围内对给定文档与查询的相关性进行评分。
0表示完全不相关，1表示完全相关。
仅返回一个介于0和1之间的浮点数评分，不要过多解释与生成。"""

        # 截断过长的文档
        doc_preview = (
            document[: self.max_context_length] + "..."
            if len(document) > self.max_context_length
            else document
        )

        user_prompt = f"查询：{query}\n\n文档：{doc_preview}"

        try:
            response = self.llm.generate_text(
                prompt=user_prompt,
                system_instruction=system_prompt,
                temperature=0,
                max_tokens=5,
            )

            # 提取评分
            score_text = response.strip()
            score_match = re.search(r"(\d+(\.\d+)?)", score_text)
            if score_match:
                score = float(score_match.group(1))
                # 确保评分在0-1范围内
                return min(1.0, max(0.0, score))

            # 解析失败，返回默认值
            return 0.5

        except Exception as e:
            print(f"评估文档相关性时出错：{e}")
            return 0.5

    def rewrite_search_query(self, query: str) -> str:
        """
        将查询重写为更适合网络搜索的形式

        优化查询以提升搜索引擎的检索效果。

        Args:
            query: 原始查询语句

        Returns:
            重写后的查询语句
        """
        system_prompt = """你是一位编写高效搜索查询的专家。
请将给定的查询重写为更适合搜索引擎的形式。
重点使用关键词和事实，去除不必要的词语，使查询更简洁明确。
仅返回重写后的查询，不要添加任何解释。"""

        try:
            response = self.llm.generate_text(
                prompt=f"原始查询：{query}\n\n重写后的查询：",
                system_instruction=system_prompt,
                temperature=0.3,
                max_tokens=50,
            )

            return response.strip()

        except Exception as e:
            print(f"重写搜索查询时出错：{e}")
            return query

    def perform_web_search(self, query: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        使用重写后的查询执行网络搜索

        Args:
            query: 用户原始查询语句

        Returns:
            元组：(搜索结果文本, 来源元数据列表)
            如果搜索失败或未启用，返回空字符串和空列表
        """
        if not self.enable_web_search:
            print("网络搜索未启用")
            return "", []

        try:
            # 重写查询以提升搜索效果
            rewritten_query = self.rewrite_search_query(query)
            print(f"重写后的搜索查询：{rewritten_query}")

            # 这里使用简化的实现
            # 在实际环境中，可以集成WebSearch工具或其他搜索API
            # 由于网络搜索在某些环境中不可用，这里提供降级机制

            # 模拟搜索结果（实际实现中应调用真实的搜索API）
            print("注意：当前使用简化的网络搜索实现")
            print("在生产环境中，应集成真实的搜索API（如Google Search API、Bing API等）")

            # 返回空结果，表示搜索不可用
            return "", []

        except Exception as e:
            print(f"执行网络搜索时出错：{e}")
            return "", []

    def refine_knowledge(self, text: str) -> str:
        """
        从文本中提取并精炼关键信息

        用于避免文档和网络搜索结果的冗余重复。

        Args:
            text: 要精炼的输入文本

        Returns:
            精炼后的关键要点
        """
        if not text or not text.strip():
            return ""

        system_prompt = """请从以下文本中提取关键信息，并以清晰简洁的项目符号列表形式呈现。
重点关注最相关和最重要的事实与细节。
你的回答应格式化为一个项目符号列表，每一项以"• "开头，换行分隔。"""

        try:
            response = self.llm.generate_text(
                prompt=f"要提炼的文本内容：\n\n{text}",
                system_instruction=system_prompt,
                temperature=0.3,
            )

            return response.strip()

        except Exception as e:
            print(f"精炼知识时出错：{e}")
            return text

    def generate_response(
        self, query: str, knowledge: str, sources: List[Dict[str, str]]
    ) -> str:
        """
        根据查询内容和提供的知识生成回答

        Args:
            query: 用户的查询内容
            knowledge: 用于生成回答的知识内容
            sources: 来源列表，每个来源包含title和url

        Returns:
            生成的回答文本
        """
        # 将来源格式化为可用于提示的内容
        sources_text = ""
        for source in sources:
            title = source.get("title", "未知来源")
            url = source.get("url", "")
            if url:
                sources_text += f"- {title}: {url}\n"
            else:
                sources_text += f"- {title}\n"

        system_prompt = """你是一个乐于助人的AI助手。请根据提供的知识内容，生成一个全面且有信息量的回答。
在回答中包含所有相关信息，同时保持语言清晰简洁。
如果知识内容不能完全回答问题，请指出这一限制。
最后在回答末尾注明引用来源。"""

        user_prompt = f"""查询内容：{query}

知识内容：
{knowledge}

引用来源：
{sources_text}

请根据以上信息，提供一个有帮助的回答，并在最后列出引用来源。"""

        try:
            response = self.llm.generate_text(
                prompt=user_prompt,
                system_instruction=system_prompt,
                temperature=0.2,
            )

            return response.strip()

        except Exception as e:
            print(f"生成回答时出错: {e}")
            return "抱歉，在尝试回答您的问题时遇到了错误。"

    def crag_process(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        执行纠错性检索增强生成（Corrective RAG）流程

        核心流程：
        1. 创建查询嵌入并检索文档
        2. 评估文档相关性
        3. 根据最高相关性得分确定操作策略
        4. 根据情况执行相应的知识获取策略
        5. 生成最终回答

        Args:
            query: 用户查询内容
            k: 初始要检索的文档数量，如果为None则使用默认值

        Returns:
            处理结果字典，包括：
            - query: 原始查询
            - response: 生成的回答
            - strategy: 使用的策略（"document_only", "web_only", "hybrid"）
            - max_relevance: 最高相关性分数
            - sources: 来源列表
            - retrieved_docs: 检索到的文档（用于调试）
        """
        print(f"\n=== 正在使用 CRAG 处理查询：{query} ===\n")

        k = k or self.default_k

        # 步骤 1: 创建查询嵌入并检索文档
        print("正在检索初始文档...")
        query_embedding = self.embedding.embed_text(query)
        retrieved_docs = self.vector_db.search_data(
            self.collection_name,
            query_embedding,
            limit=k,
            metric_type="COSINE",
            output_fields=["text", "metadata", "score"],
        )

        if not retrieved_docs:
            print("警告：未检索到任何文档")
            # 如果启用网络搜索，尝试使用网络搜索
            if self.enable_web_search:
                print("尝试使用网络搜索...")
                web_results, web_sources = self.perform_web_search(query)
                if web_results:
                    final_knowledge = self.refine_knowledge(web_results)
                    sources = web_sources
                    strategy = "web_only"
                else:
                    # 网络搜索也失败，使用LLM基础知识
                    final_knowledge = ""
                    sources = []
                    strategy = "llm_only"
            else:
                final_knowledge = ""
                sources = []
                strategy = "llm_only"

            response = self.generate_response(query, final_knowledge, sources)
            return {
                "query": query,
                "response": response,
                "strategy": strategy,
                "max_relevance": 0.0,
                "sources": sources,
                "retrieved_docs": [],
            }

        # 步骤 2: 评估文档相关性
        print("正在评估文档的相关性...")
        relevance_scores = []
        for doc in retrieved_docs:
            score = self.evaluate_document_relevance(query, doc["text"])
            relevance_scores.append(score)
            doc["relevance"] = score
            print(f"文档得分为 {score:.2f} 的相关性")

        # 步骤 3: 根据最高相关性得分确定操作策略
        max_score = max(relevance_scores) if relevance_scores else 0
        best_doc_idx = relevance_scores.index(max_score) if relevance_scores else -1

        # 记录来源用于引用
        sources = []
        final_knowledge = ""
        strategy = ""

        # 步骤 4: 根据情况执行相应的知识获取策略
        if max_score > self.high_relevance_threshold:
            # 情况 1: 高相关性 - 直接使用文档内容
            print(f"高相关性 ({max_score:.2f}) - 直接使用文档内容")
            strategy = "document_only"
            best_doc = retrieved_docs[best_doc_idx]["text"]
            final_knowledge = best_doc
            sources.append({"title": "文档", "url": ""})

        elif max_score < self.low_relevance_threshold:
            # 情况 2: 低相关性 - 使用网络搜索
            print(f"低相关性 ({max_score:.2f}) - 进行网络搜索")
            strategy = "web_only"
            web_results, web_sources = self.perform_web_search(query)

            if web_results:
                final_knowledge = self.refine_knowledge(web_results)
                sources.extend(web_sources)
            else:
                # 网络搜索失败，降级使用最佳文档
                print("网络搜索失败，降级使用文档内容")
                strategy = "document_fallback"
                final_knowledge = retrieved_docs[best_doc_idx]["text"]
                sources.append({"title": "文档（降级）", "url": ""})

        else:
            # 情况 3: 中等相关性 - 结合文档与网络搜索结果
            print(f"中等相关性 ({max_score:.2f}) - 结合文档与网络搜索")
            strategy = "hybrid"
            best_doc = retrieved_docs[best_doc_idx]["text"]
            refined_doc = self.refine_knowledge(best_doc)

            # 获取网络搜索结果
            web_results, web_sources = self.perform_web_search(query)

            if web_results:
                refined_web = self.refine_knowledge(web_results)
                # 合并知识
                final_knowledge = f"来自文档的内容:\n{refined_doc}\n\n来自网络搜索的内容:\n{refined_web}"
                sources.append({"title": "文档", "url": ""})
                sources.extend(web_sources)
            else:
                # 网络搜索失败，仅使用文档
                print("网络搜索失败，仅使用文档内容")
                strategy = "document_only"
                final_knowledge = refined_doc
                sources.append({"title": "文档", "url": ""})

        # 步骤 5: 生成最终响应
        print("正在生成最终响应...")
        response = self.generate_response(query, final_knowledge, sources)

        # 返回完整的处理结果
        return {
            "query": query,
            "response": response,
            "strategy": strategy,
            "max_relevance": max_score,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
        }

    def query(
        self, query_text: str, k: Optional[int] = None, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        执行CRAG查询（用户友好的主入口方法）

        Args:
            query_text: 查询文本
            k: 检索文档数量，如果为None则使用默认值
            verbose: 是否输出详细信息

        Returns:
            包含查询结果的字典，包含以下字段：
            - query: 原始查询
            - response: 生成的回答
            - strategy: 使用的策略
            - max_relevance: 最高相关性分数
            - sources: 来源列表
        """
        if not verbose:
            # 临时禁用print输出
            import sys
            import os
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        try:
            result = self.crag_process(query_text, k)
            return result
        finally:
            if not verbose:
                sys.stdout.close()
                sys.stdout = original_stdout


# 使用示例
if __name__ == "__main__":
    # 初始化CRAG
    crag = CRAG()

    # 处理文档
    print("处理文档示例...")
    # chunks = crag.process_document("path/to/your/document.pdf")

    # 执行查询
    print("\n查询示例...")
    result = crag.query(
        "机器学习与传统编程有何不同？",
        k=3,
        verbose=True
    )

    print("\n=== 查询结果 ===")
    print(f"查询: {result['query']}")
    print(f"策略: {result['strategy']}")
    print(f"最高相关性: {result['max_relevance']:.2f}")
    print(f"\n回答:\n{result['response']}")
    print(f"\n来源:")
    for source in result['sources']:
        print(f"  - {source['title']}")