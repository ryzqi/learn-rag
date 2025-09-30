"""
HyDE RAG (Hypothetical Document Embeddings RAG) 系统

该模块实现了 HyDE RAG 技术，通过生成假设文档来改进检索效果。
不同于传统 RAG 的查询-文档匹配，HyDE 使用文档-文档匹配提高检索精度。
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
from milvus_client import milvus_service


@dataclass
class HyDEConfig:
    """HyDE RAG 配置类"""

    chunk_size: int = 1000  # 文本分块大小
    chunk_overlap: int = 200  # 分块重叠
    num_hypothetical_docs: int = 3  # 生成假设文档数量 (1-5)
    top_k_per_doc: int = 3  # 每个假设文档检索结果数
    final_top_k: int = 5  # 最终返回结果数
    hypothetical_doc_temperature: float = 0.3  # 假设文档生成温度
    response_temperature: float = 0.2  # 最终答案生成温度
    max_hypothetical_doc_length: int = 500  # 假设文档最大长度（字符）


class HyDERAG:
    """HyDE RAG 系统

    使用假设文档嵌入(Hypothetical Document Embeddings)技术改进检索效果。
    通过生成假设文档并用其进行检索，实现更准确的文档-文档语义匹配。
    """

    def __init__(
        self,
        llm_client: Optional[GeminiLLM] = None,
        embedding_client=None,
        milvus_client=None,
        collection_name: str = "RAG_learn",
        config: Optional[HyDEConfig] = None,
    ):
        """初始化 HyDE RAG 系统

        Args:
            llm_client: LLM 客户端，用于生成假设文档和最终答案
            embedding_client: 嵌入服务，用于向量生成
            milvus_client: 向量数据库客户端
            collection_name: Milvus 集合名称
            config: HyDE 配置参数
        """
        # 使用默认服务或提供的服务初始化
        self.llm_client = llm_client or GeminiLLM()
        self.embedding_client = embedding_client or embedding_service
        self.milvus_client = milvus_client or milvus_service

        # 配置
        self.collection_name = collection_name
        self.config = config or HyDEConfig()

    def generate_hypothetical_documents(
        self, query: str, num_docs: Optional[int] = None
    ) -> List[str]:
        """生成假设文档

        基于查询生成多个假设的文档片段，这些文档片段模拟真实文档的语义和结构。

        Args:
            query: 用户查询
            num_docs: 生成假设文档的数量，如果为 None 则使用配置中的默认值

        Returns:
            假设文档列表

        Raises:
            ValueError: 当查询为空时
            Exception: 当生成失败时
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")

        num_docs = num_docs or self.config.num_hypothetical_docs

        system_prompt = """你是一个专业的文档生成助手。你的任务是基于用户的查询，生成一个假设的知识文档片段。
这个片段应该包含回答该查询所需的信息，并模仿专业文档的写作风格。
请包含详细的解释和示例（如果适用）。"""

        user_prompt = f"""请为以下查询生成一个假设的文档片段，该片段应该：
1. 直接包含回答查询所需的关键信息
2. 使用清晰、专业的语言
3. 包含相关的概念解释和示例（如果适用）
4. 长度适中（200-{self.config.max_hypothetical_doc_length}字）

查询：{query}

假设文档："""

        hypothetical_docs = []

        try:
            for i in range(num_docs):
                # 第一个文档使用较低温度确保准确性，后续文档增加温度以获得多样性
                temperature = (
                    0.1 if i == 0 else self.config.hypothetical_doc_temperature
                )

                doc = self.llm_client.generate_text(
                    user_prompt,
                    system_instruction=system_prompt,
                    temperature=temperature,
                )

                # 验证生成的文档不为空
                if doc and doc.strip():
                    # 限制文档长度
                    if len(doc) > self.config.max_hypothetical_doc_length:
                        doc = doc[: self.config.max_hypothetical_doc_length]
                    hypothetical_docs.append(doc.strip())

            if not hypothetical_docs:
                raise Exception("未能生成有效的假设文档")

            return hypothetical_docs

        except Exception as e:
            raise Exception(f"生成假设文档失败: {e}")

    def search_with_hypothetical_docs(
        self, hypothetical_docs: List[str], top_k_per_doc: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """使用假设文档进行检索

        对每个假设文档进行向量检索，返回所有检索结果。

        Args:
            hypothetical_docs: 假设文档列表
            top_k_per_doc: 每个假设文档检索的结果数，如果为 None 则使用配置中的默认值

        Returns:
            嵌套列表，每个子列表包含一个假设文档的检索结果

        Raises:
            ValueError: 当假设文档列表为空时
            Exception: 当检索失败时
        """
        if not hypothetical_docs:
            raise ValueError("假设文档列表不能为空")

        top_k_per_doc = top_k_per_doc or self.config.top_k_per_doc

        all_results = []

        try:
            for doc in hypothetical_docs:
                # 使用假设文档进行检索
                results = self.milvus_client.search_by_text(
                    collection_name=self.collection_name,
                    text=doc,
                    limit=top_k_per_doc,
                    output_fields=["text", "source", "chunk_index"],
                    metric_type="COSINE",
                    embedding_client=self.embedding_client,
                )
                all_results.append(results)

            return all_results

        except Exception as e:
            raise Exception(f"使用假设文档检索失败: {e}")

    def aggregate_search_results(
        self, all_results: List[List[Dict[str, Any]]], final_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """聚合搜索结果

        将多个假设文档的检索结果进行聚合、去重和排序。

        Args:
            all_results: 嵌套列表，包含多个检索结果列表
            final_k: 最终返回的结果数量，如果为 None 则使用配置中的默认值

        Returns:
            聚合后的检索结果列表

        Raises:
            ValueError: 当结果列表为空时
        """
        if not all_results:
            raise ValueError("结果列表不能为空")

        final_k = final_k or self.config.final_top_k

        # 使用字典进行去重，键为文本内容，值为结果
        seen_texts = {}

        for results in all_results:
            for result in results:
                # 提取文本和距离，兼容不同的 Milvus 结果格式
                text = None
                distance = None

                if isinstance(result, dict):
                    if "entity" in result:
                        # Milvus 2.x 格式
                        text = result.get("entity", {}).get("text")
                        distance = result.get("distance", float("inf"))
                    else:
                        # 简化格式
                        text = result.get("text")
                        distance = result.get(
                            "distance", result.get("score", float("inf"))
                        )

                if text and text.strip():
                    # 对于 COSINE 距离，距离越小相似度越高
                    # 如果文本已存在，保留距离更小的结果
                    if text not in seen_texts or distance < seen_texts[text].get(
                        "distance", float("inf")
                    ):
                        result_copy = (
                            result.copy() if isinstance(result, dict) else result
                        )
                        if isinstance(result_copy, dict):
                            result_copy["distance"] = distance
                        seen_texts[text] = result_copy

        # 按距离排序（距离越小越相似）
        sorted_results = sorted(
            seen_texts.values(), key=lambda x: x.get("distance", float("inf"))
        )

        # 返回前 final_k 个结果
        return sorted_results[:final_k]

    def generate_response(self, query: str, context: str) -> str:
        """基于上下文生成回答

        Args:
            query: 用户查询
            context: 检索到的上下文信息

        Returns:
            AI 生成的回答

        Raises:
            ValueError: 当查询或上下文为空时
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")

        if not context or not context.strip():
            return "没有找到相关信息来回答这个问题。"

        system_prompt = """你是一个AI助手，专门基于提供的上下文回答问题。
请仔细阅读上下文，并给出准确、详细的答案。
如果上下文不包含足够信息，请明确说明。"""

        user_prompt = f"""上下文信息:
{context}

问题: {query}

请基于上述上下文提供详细的答案。
"""

        try:
            response = self.llm_client.generate_text(
                user_prompt,
                system_instruction=system_prompt,
                temperature=self.config.response_temperature,
            )
            return response.strip()
        except Exception as e:
            raise Exception(f"响应生成失败: {e}")

    def query(
        self,
        question: str,
        num_hypothetical_docs: Optional[int] = None,
        top_k_per_doc: Optional[int] = None,
        final_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """执行完整的 HyDE RAG 查询流程

        Args:
            question: 用户查询
            num_hypothetical_docs: 生成假设文档的数量
            top_k_per_doc: 每个假设文档检索的结果数
            final_k: 最终返回的结果数

        Returns:
            包含查询结果的字典，包含以下字段：
            - query: 原始查询
            - hypothetical_documents: 生成的假设文档列表
            - retrieved_docs: 检索到的文档列表
            - context: 格式化的上下文字符串
            - response: 最终答案

        Raises:
            ValueError: 当查询为空时
            Exception: 当处理失败时
        """
        if not question or not question.strip():
            raise ValueError("查询不能为空")

        try:
            print("\n=== HyDE RAG 查询流程 ===")
            print(f"原始查询: {question}")

            # 步骤1: 生成假设文档
            print("\n步骤1: 生成假设文档...")
            hypothetical_docs = self.generate_hypothetical_documents(
                question, num_hypothetical_docs
            )
            print(f"生成了 {len(hypothetical_docs)} 个假设文档")
            for i, doc in enumerate(hypothetical_docs, 1):
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"  假设文档 {i}: {preview}")

            # 步骤2: 使用假设文档进行检索
            print("\n步骤2: 使用假设文档进行检索...")
            all_results = self.search_with_hypothetical_docs(
                hypothetical_docs, top_k_per_doc
            )
            total_retrieved = sum(len(results) for results in all_results)
            print(f"共检索到 {total_retrieved} 个结果")

            # 步骤3: 聚合搜索结果
            print("\n步骤3: 聚合和去重结果...")
            retrieved_docs = self.aggregate_search_results(all_results, final_k)
            print(f"聚合后得到 {len(retrieved_docs)} 个唯一文档")

            # 步骤4: 准备上下文
            print("\n步骤4: 准备上下文...")
            context_parts = []
            for i, result in enumerate(retrieved_docs):
                # 兼容不同的 Milvus 结果格式
                text = ""
                if isinstance(result, dict):
                    if "entity" in result:
                        # Milvus 2.x 格式
                        text = result.get("entity", {}).get("text", "")
                    else:
                        # 简化格式
                        text = result.get("text", "")

                if text and text.strip():
                    context_parts.append(f"文档片段 {i + 1}:\n{text}")

            context = "\n\n".join(context_parts)

            if not context.strip():
                return {
                    "query": question,
                    "hypothetical_documents": hypothetical_docs,
                    "retrieved_docs": [],
                    "context": "",
                    "response": "没有找到相关信息",
                }

            # 步骤5: 生成最终答案
            print("\n步骤5: 生成最终答案...")
            response = self.generate_response(question, context)

            print("\n=== 查询完成 ===\n")

            return {
                "query": question,
                "hypothetical_documents": hypothetical_docs,
                "retrieved_docs": retrieved_docs,
                "context": context,
                "response": response,
            }

        except Exception as e:
            raise Exception(f"HyDE RAG 查询失败: {e}")

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
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

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
            print(f"读取文件: {file_path}")
            text = file_reader_service.read_file(file_path)

            print("分块文本...")
            chunks = self._chunk_text(text)
            print(f"生成 {len(chunks)} 个文本块")

            print("生成嵌入向量...")
            chunk_embeddings = self.embedding_client.embed_texts(chunks)

            print("存储到向量数据库...")
            data_to_insert = []
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                data_to_insert.append(
                    {
                        "vector": embedding,
                        "text": chunk,
                        "source": file_path,
                        "chunk_index": i,
                    }
                )

            result = self.milvus_client.insert_data(
                self.collection_name, data_to_insert
            )

            print(f"文档处理完成，共处理 {len(chunks)} 个文本块")

            return {
                "success": True,
                "file_path": file_path,
                "chunks_count": len(chunks),
                **result,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "file_path": file_path}


# 创建全局服务实例
hyde_rag_service = HyDERAG()


if __name__ == "__main__":
    # 示例用法
    try:
        # 创建 HyDE RAG 实例
        hyde_rag = HyDERAG(collection_name="RAG_learn")

        # 处理文档（如果需要）
        # result = hyde_rag.process_document("Agent基础.md")
        # print(f"文档处理结果: {result}")

        # 执行查询
        query = "思维链(CoT)是什么？举个例子"
        result = hyde_rag.query(query)

        print(f"\n{'=' * 60}")
        print(f"原始查询: {result['query']}")
        print(f"\n生成的假设文档数量: {len(result['hypothetical_documents'])}")
        print(f"检索到的文档数量: {len(result['retrieved_docs'])}")
        print(f"\n最终答案:\n{result['response']}")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"执行失败: {e}")
