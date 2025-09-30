import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import datetime
import json

from LLM import GeminiLLM
from file_reader import FileReader
from embedding import EmbeddingClient
from milvus_client import milvus_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeedbackData:
    """反馈数据结构"""

    query: str
    response: str
    relevance: float  # 1-5 scale
    quality: float  # 1-5 scale
    comments: str
    timestamp: str


class FeedbackLoopRAG:
    """
    反馈循环RAG系统客户端

    实现一个具备反馈循环机制的RAG系统，通过持续学习实现性能迭代优化。
    系统将收集并整合用户反馈数据，使每次交互都能提升回答的相关性与质量。

    核心功能：
    - 记忆有效/无效的交互模式
    - 动态调整文档相关性权重
    - 将优质问答对整合至知识库
    - 通过用户互动持续增强智能水平
    """

    def __init__(
        self,
        llm_client: Optional[GeminiLLM] = None,
        file_reader: Optional[FileReader] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        collection_name: str = "RAG_learn",
        default_chunk_size: int = 1000,
        default_chunk_overlap: int = 200,
        feedback_file: str = "feedback_data.json",
    ):
        """
        初始化反馈循环RAG系统

        Args:
            llm_client: LLM客户端实例
            file_reader: 文件读取器实例
            embedding_client: 嵌入客户端实例
            collection_name: Milvus集合名称
            default_chunk_size: 默认文本块大小
            default_chunk_overlap: 默认文本块重叠大小
            feedback_file: 反馈数据存储文件路径
        """
        # 初始化服务客户端
        self.llm_client = llm_client or GeminiLLM()
        self.file_reader = file_reader or FileReader()
        self.embedding_client = embedding_client or EmbeddingClient()

        # 配置参数
        self.collection_name = collection_name
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.feedback_file = feedback_file

        # 验证必要组件
        if not all([self.llm_client, self.file_reader, self.embedding_client]):
            raise ValueError("必须提供所有必要的客户端实例")

        logger.info(f"FeedbackLoopRAG initialized with collection: {collection_name}")

    def chunk_text(
        self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[str]:
        """
        将文本分割为重叠的块

        Args:
            text: 要分割的文本
            chunk_size: 每个块的字符数，默认使用初始化时的值
            overlap: 块之间的重叠字符数，默认使用初始化时的值

        Returns:
            文本块列表

        Raises:
            ValueError: 当文本为空或参数无效时
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_chunk_overlap

        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("块大小必须大于0，重叠必须非负且小于块大小")

        chunks = []
        text_len = len(text)

        for i in range(0, text_len, chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():  # 只添加非空块
                chunks.append(chunk.strip())

        logger.info(f"Text chunked into {len(chunks)} pieces")
        return chunks

    def process_document(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        """
        处理文档并存储到向量数据库

        Args:
            file_path: 文档文件路径
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小

        Returns:
            处理后的文本块列表

        Raises:
            FileNotFoundError: 当文件不存在时
            ValueError: 当文件处理失败时
        """
        try:
            logger.info(f"Processing document: {file_path}")

            # 检查文件是否存在
            if not Path(file_path).exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

            # 读取文件
            logger.info("提取文本")
            text = self.file_reader.read_file(file_path)

            if not text or not text.strip():
                raise ValueError(f"文件内容为空: {file_path}")

            # 分块
            logger.info("分块")
            chunks = self.chunk_text(text, chunk_size, chunk_overlap)

            if not chunks:
                raise ValueError("文档分块后没有有效内容")

            # 生成嵌入
            logger.info(f"生成嵌入，共{len(chunks)}个块")
            chunk_embeddings = self.embedding_client.embed_texts(chunks)

            # 存储到向量数据库
            logger.info("存储到向量数据库")
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
                milvus_service.insert_data(self.collection_name, data)

            logger.info(f"文档处理完成，共处理{len(chunks)}个文本块")
            return chunks

        except Exception as e:
            logger.error(f"处理文档时出错: {e}")
            raise

    def get_user_feedback(
        self,
        query: str,
        response: str,
        relevance: float,
        quality: float,
        comments: str = "",
    ) -> FeedbackData:
        """
        收集用户反馈

        Args:
            query: 用户查询
            response: 系统响应
            relevance: 相关性评分 (1-5)
            quality: 质量评分 (1-5)
            comments: 额外评论

        Returns:
            FeedbackData对象

        Raises:
            ValueError: 当评分超出范围时
        """
        if not (1 <= relevance <= 5) or not (1 <= quality <= 5):
            raise ValueError("评分必须在1-5之间")

        if not query.strip() or not response.strip():
            raise ValueError("查询和响应不能为空")

        feedback = FeedbackData(
            query=query.strip(),
            response=response.strip(),
            relevance=relevance,
            quality=quality,
            comments=comments.strip(),
            timestamp=datetime.datetime.now().isoformat(),
        )

        logger.info(f"收集到用户反馈: 相关性={relevance}, 质量={quality}")
        return feedback

    def store_feedback(
        self, feedback: FeedbackData, feedback_file: Optional[str] = None
    ) -> None:
        """
        存储反馈数据到文件

        Args:
            feedback: 反馈数据对象
            feedback_file: 反馈文件路径，为None时使用默认路径

        Raises:
            IOError: 当文件写入失败时
        """
        file_path = feedback_file or self.feedback_file

        try:
            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            # 将反馈数据转换为字典并写入文件
            feedback_dict = {
                "query": feedback.query,
                "response": feedback.response,
                "relevance": feedback.relevance,
                "quality": feedback.quality,
                "comments": feedback.comments,
                "timestamp": feedback.timestamp,
            }

            with open(file_path, "a", encoding="utf-8") as f:
                json.dump(feedback_dict, f, ensure_ascii=False)
                f.write("\n")

            logger.info(f"反馈数据已存储到: {file_path}")

        except Exception as e:
            logger.error(f"存储反馈数据时出错: {e}")
            raise IOError(f"无法存储反馈数据: {e}")

    def load_feedback_data(
        self, feedback_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        从文件加载历史反馈数据

        Args:
            feedback_file: 反馈文件路径，为None时使用默认路径

        Returns:
            反馈数据列表
        """
        file_path = feedback_file or self.feedback_file
        feedback_data = []

        try:
            if not Path(file_path).exists():
                logger.warning(f"反馈文件不存在: {file_path}，将返回空列表")
                return feedback_data

            with open(file_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            feedback = json.loads(line)
                            # 验证必要字段
                            required_fields = [
                                "query",
                                "response",
                                "relevance",
                                "quality",
                                "timestamp",
                            ]
                            if all(field in feedback for field in required_fields):
                                feedback_data.append(feedback)
                            else:
                                logger.warning(f"第{line_no}行缺少必要字段，跳过")
                        except json.JSONDecodeError as e:
                            logger.warning(f"第{line_no}行JSON解析失败: {e}，跳过")

            logger.info(f"成功加载{len(feedback_data)}条反馈数据")
            return feedback_data

        except Exception as e:
            logger.error(f"加载反馈数据时出错: {e}")
            return []

    def assess_feedback_relevance(
        self, query: str, doc_text: str, feedback: Dict[str, Any]
    ) -> bool:
        """
        评估历史反馈与当前查询和文档的相关性

        Args:
            query: 当前查询
            doc_text: 文档内容
            feedback: 历史反馈数据

        Returns:
            是否相关的布尔值
        """
        try:
            system_prompt = """您是专门用于判断历史反馈与当前查询及文档相关性的AI系统。
请仅回答 'yes' 或 'no'。您的任务是严格判断相关性，无需提供任何解释。"""

            user_prompt = f"""
当前查询: {query}
收到反馈的历史查询: {feedback["query"]}
文档内容: {doc_text[:500]}... [截断]
收到反馈的历史响应: {feedback["response"][:500]}... [截断]

该历史反馈是否与当前查询及文档相关？(yes/no)
"""

            response = self.llm_client.generate_text(
                prompt=user_prompt, system_instruction=system_prompt
            )

            answer = response.strip().lower()
            is_relevant = "yes" in answer

            logger.debug(f"相关性评估结果: {is_relevant}")
            return is_relevant

        except Exception as e:
            logger.error(f"评估反馈相关性时出错: {e}")
            return False

    def adjust_relevance_scores(
        self,
        query: str,
        results: List[Dict[str, Any]],
        feedback_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        基于历史反馈调整检索结果的相关性分数

        Args:
            query: 当前查询
            results: 检索结果列表
            feedback_data: 历史反馈数据

        Returns:
            调整后的检索结果列表
        """
        if not feedback_data:
            logger.info("没有历史反馈数据，跳过分数调整")
            return results

        logger.info(f"基于{len(feedback_data)}条反馈历史调整相关性分数...")

        try:
            for i, result in enumerate(results):
                document_text = result.get("text", "")
                relevant_feedback = []

                # 找到与当前文档和查询相关的反馈
                for feedback in feedback_data:
                    if self.assess_feedback_relevance(query, document_text, feedback):
                        relevant_feedback.append(feedback)

                if relevant_feedback:
                    # 计算平均相关性分数
                    avg_relevance = sum(
                        f["relevance"] for f in relevant_feedback
                    ) / len(relevant_feedback)
                    avg_quality = sum(f["quality"] for f in relevant_feedback) / len(
                        relevant_feedback
                    )

                    # 计算调整因子 (0.5-1.5之间)
                    relevance_modifier = 0.5 + (avg_relevance / 5.0)
                    quality_modifier = 0.5 + (avg_quality / 5.0)
                    combined_modifier = (relevance_modifier + quality_modifier) / 2

                    # 获取原始分数 - 修复：从正确位置获取分数
                    original_score = result.get("score", 0.0)
                    if original_score == 0.0:
                        # 如果score为0，尝试从distance转换（COSINE距离转相似度）
                        distance = result.get("distance", 1.0)
                        if distance is not None:
                            original_score = 1.0 - distance  # COSINE距离转相似度

                    # 计算调整后的分数
                    adjusted_score = original_score * combined_modifier

                    # 更新结果 - 确保结构一致性
                    if "entity" not in result:
                        result["entity"] = {}
                    if "metadata" not in result["entity"]:
                        result["entity"]["metadata"] = {}

                    result["original_similarity"] = original_score
                    result["score"] = adjusted_score  # 更新顶级score
                    result["entity"]["score"] = adjusted_score  # 也更新entity中的score以保持兼容性
                    result["entity"]["metadata"]["relevance_score"] = adjusted_score
                    result["feedback_applied"] = True
                    result["entity"]["metadata"]["feedback_count"] = len(
                        relevant_feedback
                    )
                    result["entity"]["metadata"]["avg_feedback_relevance"] = (
                        avg_relevance
                    )
                    result["entity"]["metadata"]["avg_feedback_quality"] = avg_quality

                    logger.info(
                        f"文档 {i + 1}: 基于 {len(relevant_feedback)} 条反馈，"
                        f"分数从 {original_score:.4f} 调整为 {adjusted_score:.4f}"
                    )

            # 按调整后的分数重新排序 - 修复：从正确位置获取分数进行排序
            results.sort(
                key=lambda x: x.get("score", x.get("entity", {}).get("score", 0)), reverse=True
            )

            return results

        except Exception as e:
            logger.error(f"调整相关性分数时出错: {e}")
            return results

    def search_with_feedback(
        self, query: str, k: int = 5, feedback_file: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        执行带反馈优化的检索和生成

        Args:
            query: 查询文本
            k: 返回结果数量
            feedback_file: 反馈文件路径

        Returns:
            (检索结果列表, 生成的回答)

        Raises:
            ValueError: 当查询为空时
        """
        if not query.strip():
            raise ValueError("查询不能为空")

        try:
            logger.info(f"开始带反馈优化的检索，查询: {query[:50]}...")

            # 生成查询嵌入
            query_embedding = self.embedding_client.embed_text(query)

            # 执行初始检索 - 修复：添加output_fields确保返回text字段
            results = milvus_service.search_data(
                collection_name=self.collection_name,
                data=query_embedding,
                limit=k * 2,  # 获取更多结果以便筛选
                metric_type="COSINE",  # 使用 COSINE 相似度
                output_fields=["text", "metadata"]  # 确保返回text和metadata字段
            )

            if not results:
                logger.warning("未找到相关文档")
                return [], "抱歉，未找到相关信息。"

            # 调试：检查检索结果的结构
            logger.debug(f"检索结果数量: {len(results)}")
            for i, result in enumerate(results[:2]):  # 只检查前2个结果
                logger.debug(f"结果 {i+1} 结构: {list(result.keys())}")
                logger.debug(f"结果 {i+1} text字段: {result.get('text', 'NOT_FOUND')[:100]}...")

            # 加载历史反馈并调整分数
            feedback_data = self.load_feedback_data(feedback_file)
            adjusted_results = self.adjust_relevance_scores(
                query, results, feedback_data
            )

            # 取前k个结果
            final_results = adjusted_results[:k]

            # 生成回答 - 修复：添加调试信息确保context不为空
            context_parts = []
            for i, result in enumerate(final_results):
                text = result.get("text", "")
                if text.strip():
                    context_parts.append(text.strip())
                    logger.debug(f"添加到上下文的文本 {i+1}: {text[:100]}...")
                else:
                    logger.warning(f"结果 {i+1} 的text字段为空或不存在")

            context = "\n\n".join(context_parts)
            
            if not context.strip():
                logger.error("上下文为空！检索结果中没有有效的text字段")
                return final_results, "抱歉，检索到的文档内容为空，无法生成回答。"

            generation_prompt = f"""
基于以下上下文信息回答问题：

上下文：
{context}

问题：{query}

请提供准确、有用的回答：
"""

            answer = self.llm_client.generate_text(generation_prompt)

            logger.info(f"检索完成，返回{len(final_results)}个结果")
            logger.info(f"生成的上下文长度: {len(context)} 字符")
            return final_results, answer

        except Exception as e:
            logger.error(f"带反馈检索时出错: {e}")
            raise

    def get_feedback_statistics(
        self, feedback_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取反馈数据统计信息

        Args:
            feedback_file: 反馈文件路径

        Returns:
            统计信息字典
        """
        feedback_data = self.load_feedback_data(feedback_file)

        if not feedback_data:
            return {"total_count": 0}

        relevance_scores = [f["relevance"] for f in feedback_data]
        quality_scores = [f["quality"] for f in feedback_data]

        stats = {
            "total_count": len(feedback_data),
            "avg_relevance": sum(relevance_scores) / len(relevance_scores),
            "avg_quality": sum(quality_scores) / len(quality_scores),
            "min_relevance": min(relevance_scores),
            "max_relevance": max(relevance_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
        }

        logger.info(f"反馈统计: {stats}")
        return stats


# 创建默认实例
# 示例用法
if __name__ == "__main__":
    import os

    # 创建FeedbackLoopRAG实例
    rag = FeedbackLoopRAG(
        collection_name="RAG_learn", feedback_file="demo_feedback.json"
    )

    print("=== 反馈循环RAG系统演示 ===")

    try:
        # 步骤1：处理文档并建立知识库
        print("\n1. 处理文档并建立知识库...")
        # result = rag.process_document("Agent基础.md")
        # print(f"文档处理完成，共生成 {len(result)} 个文本块")

        # 步骤2：执行初始查询（无历史反馈）
        print("\n2. 执行初始查询（无历史反馈优化）...")
        query1 = "思维链是什么？举个例子"

        results1, answer1 = rag.search_with_feedback(query1, k=3)
        print(f"查询: {query1}")
        print(f"回答: {answer1[:200]}...")
        print(f"检索到 {len(results1)} 个相关文档块")

        # 步骤3：收集用户反馈（模拟高质量反馈）
        print("\n3. 收集用户反馈...")
        feedback1 = rag.get_user_feedback(
            query=query1,
            response=answer1,
            relevance=4.5,  # 高相关性
            quality=4.2,  # 高质量
            comments="回答很详细，例子很有帮助",
        )
        rag.store_feedback(feedback1)
        print("[OK] 反馈已存储")

        # 步骤4：执行相似查询（带反馈优化）
        print("\n4. 执行相似查询（带反馈优化）...")
        query2 = "能详细解释一下思维链的工作原理吗？"

        results2, answer2 = rag.search_with_feedback(query2, k=3)
        print(f"查询: {query2}")
        print(f"回答: {answer2[:200]}...")

        # 检查是否应用了反馈优化
        feedback_applied_count = sum(
            1 for r in results2 if r.get("feedback_applied", False)
        )
        print(f"[OK] 基于历史反馈优化了 {feedback_applied_count} 个检索结果")


        # 步骤7：显示反馈统计信息
        print("\n7. 反馈统计信息...")
        stats = rag.get_feedback_statistics()
        if stats["total_count"] > 0:
            print(f"总反馈条数: {stats['total_count']}")
            print(f"平均相关性评分: {stats['avg_relevance']:.2f}")
            print(f"平均质量评分: {stats['avg_quality']:.2f}")
            print(
                f"相关性评分范围: {stats['min_relevance']:.1f} - {stats['max_relevance']:.1f}"
            )
            print(
                f"质量评分范围: {stats['min_quality']:.1f} - {stats['max_quality']:.1f}"
            )


        # 最终统计
        final_stats = rag.get_feedback_statistics()
        print(f"\n[OK] 系统持续学习完成！总反馈数量: {final_stats['total_count']}")

        # 清理演示文件（可选）
        cleanup = input("\n是否删除演示产生的反馈文件? (y/n): ").lower().strip()
        if cleanup == "y":
            try:
                os.remove("demo_feedback.json")
                print("[OK] 演示文件已清理")
            except FileNotFoundError:
                print("[OK] 无需清理（文件不存在）")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


# 创建默认实例
feedback_rag_service = FeedbackLoopRAG()

# 示例用法
if __name__ == "__main__":
    import os

    # 创建FeedbackLoopRAG实例
    rag = FeedbackLoopRAG(
        collection_name="RAG_learn", feedback_file="demo_feedback.json"
    )

    print("=== 反馈循环RAG系统演示 ===")

    try:
        # 步骤1：处理文档并建立知识库
        print("\n1. 处理文档并建立知识库...")
        # result = rag.process_document("Agent基础.md")
        # print(f"文档处理完成，共生成 {len(result)} 个文本块")

        # 步骤2：执行初始查询（无历史反馈）
        print("\n2. 执行初始查询（无历史反馈优化）...")
        query1 = "思维链是什么？举个例子"

        results1, answer1 = rag.search_with_feedback(query1, k=3)
        print(f"查询: {query1}")
        print(f"回答: {answer1}")
        print(f"检索到 {len(results1)} 个相关文档块")

        # 步骤3：收集用户反馈（模拟高质量反馈）
        print("\n3. 收集用户反馈...")
        feedback1 = rag.get_user_feedback(
            query=query1,
            response=answer1,
            relevance=4.5,  # 高相关性
            quality=4.2,  # 高质量
            comments="回答很详细，例子很有帮助",
        )
        rag.store_feedback(feedback1)
        print("✓ 反馈已存储")

        # 步骤4：执行相似查询（带反馈优化）
        print("\n4. 执行相似查询（带反馈优化）...")
        query2 = "能详细解释一下思维链的工作原理吗？"

        results2, answer2 = rag.search_with_feedback(query2, k=3)
        print(f"查询: {query2}")
        print(f"回答: {answer2}")

        # 检查是否应用了反馈优化
        feedback_applied_count = sum(
            1 for r in results2 if r.get("feedback_applied", False)
        )
        print(f"✓ 基于历史反馈优化了 {feedback_applied_count} 个检索结果")


        # 步骤7：显示反馈统计信息
        print("\n7. 反馈统计信息...")
        stats = rag.get_feedback_statistics()
        if stats["total_count"] > 0:
            print(f"总反馈条数: {stats['total_count']}")
            print(f"平均相关性评分: {stats['avg_relevance']:.2f}")
            print(f"平均质量评分: {stats['avg_quality']:.2f}")
            print(
                f"相关性评分范围: {stats['min_relevance']:.1f} - {stats['max_relevance']:.1f}"
            )
            print(
                f"质量评分范围: {stats['min_quality']:.1f} - {stats['max_quality']:.1f}"
            )


        # 最终统计
        final_stats = rag.get_feedback_statistics()
        print(f"\n✓ 系统持续学习完成！总反馈数量: {final_stats['total_count']}")

        # 清理演示文件（可选）
        cleanup = input("\n是否删除演示产生的反馈文件? (y/n): ").lower().strip()
        if cleanup == "y":
            try:
                os.remove("demo_feedback.json")
                print("✓ 演示文件已清理")
            except FileNotFoundError:
                print("✓ 无需清理（文件不存在）")

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
