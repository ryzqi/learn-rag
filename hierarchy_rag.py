from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
from milvus_client import milvus_service
import hashlib
import fitz  # PyMuPDF for PDF processing
from typing import List, Dict, Any
import os
import gc


class HierarchyRAG:
    """
    层次化检索增强生成(Hierarchy RAG)类

    实现两级检索系统：首先通过摘要识别相关文档部分，然后从这些部分检索具体细节。
    解决传统RAG在处理长文档时上下文信息丢失和检索效率低下的问题。

    主要特点：
    1. 页面级摘要：为文档的每一页生成简洁摘要
    2. 详细文本块：对每一页创建详细的重叠文本块
    3. 两级检索：先搜索摘要确定相关页面，再在相关页面中搜索详细块
    4. 上下文保持：在保留具体细节的同时保持更大的上下文信息
    """

    def __init__(
        self,
        llm_client=None,
        embedding_client=None,
        vector_client=None,
        file_reader_client=None,
        summary_collection: str = "hierarchy_summaries",
        detail_collection: str = "hierarchy_details",
        chunk_size: int = 1000,
        overlap: int = 200,
        k_summaries: int = 3,
        k_chunks: int = 5,
        batch_size: int = 64,
        temperature: float = 0.2,
    ):
        """
        初始化层次化RAG实例

        Args:
            llm_client: LLM客户端，如果为None则使用默认GeminiLLM
            embedding_client: 嵌入客户端，如果为None则使用默认embedding_service
            vector_client: 向量数据库客户端，如果为None则使用默认milvus_service
            file_reader_client: 文件读取客户端，如果为None则使用默认file_reader_service
            summary_collection: 摘要向量数据库集合名称
            detail_collection: 详细向量数据库集合名称
            chunk_size: 详细文本块大小
            overlap: 文本块重叠大小
            k_summaries: 检索的摘要数量
            k_chunks: 每个摘要检索的详细块数量
            batch_size: 批处理大小
            temperature: 生成温度
        """
        # 参数验证
        if chunk_size <= 0:
            raise ValueError("文本块大小必须大于0")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("重叠大小必须大于等于0且小于文本块大小")
        if k_summaries <= 0 or k_chunks <= 0:
            raise ValueError("检索数量必须大于0")
        if batch_size <= 0:
            raise ValueError("批处理大小必须大于0")

        # 依赖注入，支持默认值
        self.llm_client = llm_client or GeminiLLM()
        self.embedding_client = embedding_client or embedding_service
        self.vector_client = vector_client or milvus_service
        self.file_reader = file_reader_client or file_reader_service

        # 配置参数
        self.summary_collection = summary_collection
        self.detail_collection = detail_collection
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.k_summaries = k_summaries
        self.k_chunks = k_chunks
        self.batch_size = batch_size
        self.temperature = temperature

        # 系统提示词
        self.system_prompt = (
            "你是一个AI助手，严格根据给定的上下文进行回答。"
            "如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"
        )

        # 摘要生成提示词
        self.summary_prompt = (
            "你是一个专业的摘要生成系统。请对提供的文本创建一个详细的摘要。"
            "重点捕捉主要内容、关键信息和重要事实。"
            "你的摘要应足够全面，能够让人理解该页面包含的内容，但要比原文更简洁。"
        )

    def extract_pages_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        从PDF文件中按页提取文本内容

        Args:
            pdf_path: PDF文件路径

        Returns:
            包含文本内容和元数据的页面列表

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持或损坏
            PermissionError: 文件访问权限不足
        """
        # 验证文件路径
        if not pdf_path or not pdf_path.strip():
            raise ValueError("PDF文件路径不能为空")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        pages = []
        pdf_doc = None

        try:
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)

            for page_num in range(total_pages):
                page = pdf_doc[page_num]
                text = page.get_text()

                # 跳过文本很少的页面
                if len(text.strip()) > 50:
                    page_data = {
                        "text": text,
                        "metadata": {
                            "source": pdf_path,
                            "page": page_num + 1,  # 页码从1开始
                            "total_pages": total_pages,
                        },
                    }
                    pages.append(page_data)

        except fitz.FileDataError as e:
            raise ValueError(f"PDF文件损坏或无法读取: {e}")
        except fitz.EmptyFileError as e:
            raise ValueError(f"PDF文件为空: {e}")
        except PermissionError:
            raise PermissionError(f"无权限访问PDF文件: {pdf_path}")
        except Exception as e:
            raise RuntimeError(f"PDF提取意外失败: {e}")
        finally:
            if pdf_doc:
                pdf_doc.close()
            gc.collect()

        return pages

    def generate_page_summary(self, page_text: str) -> str:
        """
        为页面文本生成摘要

        Args:
            page_text: 页面原始文本

        Returns:
            生成的摘要文本
        """
        if not page_text or not page_text.strip():
            raise ValueError("页面文本不能为空")

        # 如果文本过长，截断到合理长度
        max_tokens = 6000
        truncated_text = (
            page_text[:max_tokens] if len(page_text) > max_tokens else page_text
        )

        user_prompt = f"请总结以下文本：\n\n{truncated_text}"

        try:
            summary = self.llm_client.generate_text(
                user_prompt,
                system_instruction=self.summary_prompt,
                temperature=self.temperature,
            )

            return summary

        except Exception as e:
            raise Exception(f"摘要生成失败: {e}")

    def chunk_page_text(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        将页面文本分割为重叠的详细块

        Args:
            text: 输入文本
            metadata: 页面元数据

        Returns:
            详细文本块列表
        """
        if not text or not text.strip():
            return []

        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(text), step):
            chunk_text = text[i : i + self.chunk_size]

            # 跳过很小的块
            if chunk_text and len(chunk_text.strip()) > 50:
                chunk_metadata = metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": len(chunks),
                        "start_char": i,
                        "end_char": i + len(chunk_text),
                        "is_summary": False,
                    }
                )

                chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        return chunks

    def _batch_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        if not texts:
            return []

        try:
            # 如果文本数量超过批处理大小，分批处理
            if len(texts) <= self.batch_size:
                return self.embedding_client.embed_texts(texts)

            # 分批处理大量文本
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_embeddings = self.embedding_client.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            raise Exception(f"嵌入向量生成失败: {e}")

    def process_document_hierarchically(self, file_path: str) -> Dict[str, Any]:
        """
        层次化处理文档：创建页面摘要和详细文本块

        Args:
            file_path: 文档文件路径

        Returns:
            处理结果字典
        """
        try:
            print(f"开始处理文档: {file_path}")

            # 提取页面
            pages_data = self.extract_pages_from_pdf(file_path)
            summaries = []
            detailed_chunks = []

            print(f"总共提取了 {len(pages_data)} 页")

            # 处理每一页
            for i, page_data in enumerate(pages_data):
                # 生成摘要
                summary_text = self.generate_page_summary(page_data["text"])
                summary_metadata = page_data["metadata"].copy()
                summary_metadata.update({"is_summary": True})

                summaries.append({"text": summary_text, "metadata": summary_metadata})

                # 创建详细文本块
                page_chunks = self.chunk_page_text(
                    page_data["text"], page_data["metadata"]
                )
                detailed_chunks.extend(page_chunks)

                if (i + 1) % 5 == 0:
                    print(f"已处理 {i + 1} 页...")

            print(
                f"创建了 {len(summaries)} 个摘要和 {len(detailed_chunks)} 个详细文本块"
            )

            # 生成嵌入向量
            print("正在为摘要生成嵌入向量...")
            summary_texts = [summary["text"] for summary in summaries]
            summary_embeddings = self._batch_embed_texts(summary_texts)

            print("正在为详细块生成嵌入向量...")
            chunk_texts = [chunk["text"] for chunk in detailed_chunks]
            chunk_embeddings = self._batch_embed_texts(chunk_texts)

            # 存储摘要到向量数据库
            print("正在存储摘要到向量数据库...")
            summary_data_list = []
            for i, (summary, embedding) in enumerate(
                zip(summaries, summary_embeddings)
            ):
                summary_id = self._generate_summary_id(
                    file_path, summary["metadata"]["page"]
                )
                summary_data = {
                    "id": summary_id,
                    "vector": embedding,
                    "text": summary["text"],
                    "source": file_path,
                    "page": summary["metadata"]["page"],
                    "is_summary": True,
                }
                summary_data_list.append(summary_data)

            summary_result = self.vector_client.insert_data(
                self.summary_collection, summary_data_list
            )

            # 存储详细块到向量数据库
            print("正在存储详细块到向量数据库...")
            chunk_data_list = []
            for i, (chunk, embedding) in enumerate(
                zip(detailed_chunks, chunk_embeddings)
            ):
                chunk_id = self._generate_chunk_id(
                    file_path, chunk["metadata"]["page"], i
                )
                chunk_data = {
                    "id": chunk_id,
                    "vector": embedding,
                    "text": chunk["text"],
                    "source": file_path,
                    "page": chunk["metadata"]["page"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "is_summary": False,
                }
                chunk_data_list.append(chunk_data)

            chunk_result = self.vector_client.insert_data(
                self.detail_collection, chunk_data_list
            )

            result = {
                "success": True,
                "file_path": file_path,
                "pages_count": len(pages_data),
                "summaries_count": len(summaries),
                "chunks_count": len(detailed_chunks),
                "summary_insert_result": summary_result,
                "chunk_insert_result": chunk_result,
            }

            print(f"文档处理完成: {file_path}")
            return result

        except Exception as e:
            return {"success": False, "error": str(e), "file_path": file_path}

    def hierarchical_search(
        self, query: str, k_summaries: int = None, k_chunks: int = None
    ) -> List[Dict[str, Any]]:
        """
        执行层次化检索：首先搜索摘要，然后在相关页面中搜索详细块

        Args:
            query: 查询文本
            k_summaries: 要检索的摘要数量，如果为None则使用默认值
            k_chunks: 每个摘要要检索的块数量，如果为None则使用默认值

        Returns:
            检索到的带有相关性分数的块列表
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        k_summaries = k_summaries or self.k_summaries
        k_chunks = k_chunks or self.k_chunks

        try:
            # 首先，检索相关的摘要
            summary_results = self.vector_client.search_by_text(
                collection_name=self.summary_collection,
                text=query,
                limit=k_summaries,
                output_fields=["text", "source", "page"],
                metric_type="COSINE",
                embedding_client=self.embedding_client,
            )

            print(f"检索到 {len(summary_results)} 个相关摘要")

            # 标准化摘要结果并收集相关页面
            normalized_summaries = [
                self._normalize_search_result(result) for result in summary_results
            ]
            relevant_pages = [
                summary["page"]
                for summary in normalized_summaries
                if summary["page"] is not None
            ]

            if not relevant_pages:
                return []

            # 在相关页面中搜索详细块
            detailed_results = []

            for page in relevant_pages:
                page_results = self.vector_client.search_by_text(
                    collection_name=self.detail_collection,
                    text=query,
                    limit=k_chunks,
                    output_fields=["text", "source", "page", "chunk_index"],
                    metric_type="COSINE",
                    embedding_client=self.embedding_client,
                    filter_expression=f"page == {page}",
                )

                # 标准化详细结果
                normalized_page_results = [
                    self._normalize_search_result(result) for result in page_results
                ]
                detailed_results.extend(normalized_page_results)

            # 按相似度分数排序（距离越小越相似）
            detailed_results.sort(key=lambda x: x.get("distance", 1.0))

            print(f"从相关页面检索到 {len(detailed_results)} 个详细块")

            # 为每个详细结果添加对应的摘要信息
            summary_by_page = {
                summary["page"]: summary["text"] for summary in normalized_summaries
            }
            for result in detailed_results:
                page_num = result["page"]
                if page_num in summary_by_page:
                    result["summary"] = summary_by_page[page_num]

            return detailed_results[: k_chunks * k_summaries]

        except Exception as e:
            raise Exception(f"层次化搜索失败: {e}")

    def generate_response(self, query: str, context: str) -> str:
        """
        基于查询和上下文生成回答

        Args:
            query: 用户查询
            context: 检索到的上下文

        Returns:
            生成的回答
        """
        if not context or not context.strip():
            return "我没有找到相关的信息来回答这个问题。"

        # 限制上下文长度以避免token限制
        max_context_length = 8000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        user_prompt = f"上下文:\n{context}\n\n用户问题：{query}"

        try:
            response = self.llm_client.generate_text(
                user_prompt,
                system_instruction=self.system_prompt,
                temperature=self.temperature,
            )

            return response

        except Exception as e:
            return f"生成回答时出错: {e}"

    def query(
        self, question: str, k_summaries: int = None, k_chunks: int = None
    ) -> str:
        """
        完整的层次化查询流水线：搜索、准备上下文和生成响应

        Args:
            question: 用户的问题
            k_summaries: 最大搜索摘要数
            k_chunks: 每个摘要的最大块数

        Returns:
            生成的回答
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        try:
            # 执行层次化搜索
            search_results = self.hierarchical_search(question, k_summaries, k_chunks)

            if not search_results:
                return "我没有找到相关的信息来回答这个问题。"

            # 准备上下文
            context_parts = []
            for i, result in enumerate(search_results):
                # 使用标准化结果
                text = result.get("text", "")
                page_num = result.get("page", "未知")
                score = result.get("score", 0.0)

                if text.strip():
                    context_parts.append(
                        f"页面{page_num}(相似度:{score:.3f})：\n{text}"
                    )

                # 限制上下文数量以控制token使用
                if len(context_parts) >= 10:
                    break

            context = "\n\n".join(context_parts)

            # 生成响应
            response = self.generate_response(question, context)

            return response

        except Exception as e:
            return f"查询处理失败: {e}"

    def _generate_summary_id(self, file_path: str, page: int) -> str:
        """生成页面摘要的唯一ID"""
        content = f"{file_path}_page_{page}_summary"
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_chunk_id(self, file_path: str, page: int, chunk_index: int) -> str:
        """生成文本块的唯一ID"""
        content = f"{file_path}_page_{page}_chunk_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()

    def _normalize_search_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        标准化搜索结果格式，处理不同的Milvus结果结构

        Args:
            result: 原始搜索结果

        Returns:
            标准化的结果字典
        """
        if "entity" in result:
            entity = result["entity"]
            return {
                "text": entity.get("text", ""),
                "source": entity.get("source", ""),
                "page": entity.get("page"),
                "chunk_index": entity.get("chunk_index"),
                "distance": result.get("distance", 1.0),
                "score": 1.0 - result.get("distance", 1.0),
            }
        else:
            return {
                "text": result.get("text", ""),
                "source": result.get("source", ""),
                "page": result.get("page"),
                "chunk_index": result.get("chunk_index"),
                "distance": result.get("distance", 1.0),
                "score": 1.0 - result.get("distance", 1.0),
            }


# 使用示例
if __name__ == "__main__":
    print("层次化RAG系统初始化...")

    # 创建层次化RAG实例
    hierarchy_rag = HierarchyRAG(
        summary_collection="hierarchy_summaries",
        detail_collection="hierarchy_details",
        chunk_size=1000,
        overlap=200,
        k_summaries=3,
        k_chunks=5,
        batch_size=64,
        temperature=0.2,
    )

    print("层次化RAG系统初始化完成！")
    print(f"- 摘要集合: {hierarchy_rag.summary_collection}")
    print(f"- 详细集合: {hierarchy_rag.detail_collection}")
    print(f"- 文本块大小: {hierarchy_rag.chunk_size}")
    print(f"- 重叠大小: {hierarchy_rag.overlap}")
    print(f"- 摘要检索数: {hierarchy_rag.k_summaries}")
    print(f"- 块检索数: {hierarchy_rag.k_chunks}")

    print("\n使用示例:")
    print("# 处理文档")
    print(
        "# result = hierarchy_rag.process_document_hierarchically('path/to/document.pdf')"
    )
    print("# print(f'处理结果: {result}')")
    print("")
    print("# 查询")
    print("# response = hierarchy_rag.query('什么是深度学习？')")
    print("# print(f'查询回答: {response}')")
