import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import datetime
import json

from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
from milvus_client import milvus_service
import re


client = GeminiLLM()


def chunk_text(text, n, overlap) -> list[Any]:
    """
    将文本分割为重叠的块

    Args:
    text (str): 要分割的文本
    n (int): 每个块的字符数
    overlap (int): 块之间的重叠字符数

    Returns:
    List[str]: 文本块列表
    """
    chunks = []  #
    for i in range(0, len(text), n - overlap):
        # 添加从当前索引到索引 + 块大小的文本块
        chunk = text[i : i + n]
        if chunk:
            chunks.append(chunk)

    return chunks


def process_document(file_path, chunk_size=1000, overlap=200):
    text = file_reader_service.read_file(file_path)

    chunks = chunk_text(text, chunk_size, overlap)

    chunk_embeddings = embedding_service.embed_chunks(chunks)

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
        milvus_service.insert_data("RAG_learn", data)

    return chunks


def classify_query(query):
    system_prompt = """您是专业的查询分类专家。
        请将给定查询严格分类至以下四类中的唯一一项：
        - Factual：需要具体、可验证信息的查询
        - Analytical：需要综合分析或深入解释的查询
        - Opinion：涉及主观问题或寻求多元观点的查询
        - Contextual：依赖用户具体情境的查询

        请仅返回分类名称，不要添加任何解释或额外文本。
    """

    user_prompt = f"对以下查询进行分类: {query}"

    response = client.generate_text(
        prompt=user_prompt, system_instruction=system_prompt
    )

    category = response.strip()

    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]

    for valid in valid_categories:
        if valid in category:
            return valid

    return "Factual"


def factual_retrieval_strategy(query, k=4):
    print("执行事实性检索策略")
    system_prompt = """您是搜索查询优化专家。
        您的任务是重构给定的事实性查询，使其更精确具体以提升信息检索效果。
        重点关注关键实体及其关联关系。

        请仅提供优化后的查询，不要包含任何解释。
    """

    user_prompt = f"请优化此事实性查询: {query}"

    response = client.generate_text(
        prompt=user_prompt, system_instruction=system_prompt, temperature=0
    )

    enhanced_query = response.strip()

    query_embedding = embedding_service.embed_text(enhanced_query)

    initial_results = milvus_service.search_data(
        "RAG_learn",
        query_embedding,
        limit=k * 2,
        metric_type="COSINE",
        output_fields=["text", "metadata", "score"],
    )


    ranked_results = []

    for doc in initial_results:
        relevance_score = score_document_relevance(
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


def analytical_retrieval_strategy(query, k=4):
    print("执行分析性检索策略")

    system_prompt = """您是复杂问题拆解专家。
    请针对给定的分析性查询生成探索不同维度的子问题。
    这些子问题应覆盖主题的广度并帮助获取全面信息。

    请严格生成恰好3个子问题，每个问题单独一行。
    """

    user_prompt = f"请为此分析性查询生成子问题：{query}"

    response = client.generate_text(
        prompt=user_prompt, system_instruction=system_prompt, temperature=0.3
    )

    sub_queries = response.strip().split("\n")
    sub_queries = [q.strip() for q in sub_queries if q.strip()]

    print(f"生成的子问题: {sub_queries}")

    all_results = []
    for sub_query in sub_queries:
        results = milvus_service.search_by_text(
            "RAG_learn",
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
        main_query_embedding = embedding_service.embed_text(query)

        main_results = milvus_service.search_data(
            "RAG_learn",
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


def opinion_retrieval_strategy(query, k=4):
    print("执行观点性检索策略")

    system_prompt = """您是主题多视角分析专家。
        针对给定的观点类或意见类查询，请识别人们可能持有的不同立场或观点。

        请严格返回恰好3个不同观点角度，每个角度单独一行。
    """

    user_prompt = f"请识别以下主题的不同观点：{query}"

    response = client.generate_text(
        prompt=user_prompt, system_instruction=system_prompt, temperature=0.3
    )

    viewpoints = response.strip().split("\n")
    viewpoints = [v.strip() for v in viewpoints if v.strip()]

    print(f"生成的观点: {viewpoints}")

    all_results = []
    for viewpoint in viewpoints:
        combined_query = f"{query} {viewpoint}"
        viewpoint_embedding = embedding_service.embed_text(combined_query)
        results = milvus_service.search_data(
            "RAG_learn",
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
        viewpoints_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
        if viewpoints_docs:
            selected_results.append(viewpoints_docs[0])

    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["score"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])

    return selected_results[:k]


def contextual_retrieval_strategy(query, k=4, user_context=None):
    print("执行上下文检索策略")
    if not user_context:
        system_prompt = """您是理解查询隐含上下文的专家。
        对于给定的查询，请推断可能相关或隐含但未明确说明的上下文信息。
        重点关注有助于回答该查询的背景信息。

        请简要描述推断的隐含上下文。
        """

        user_prompt = f"推断此查询中的隐含背景(上下文)：{query}"

        response = client.generate_text(
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

    response = client.generate_text(
        prompt=user_prompt, system_instruction=system_prompt, temperature=0
    )

    contextualized_query = response.strip()

    print(f"重新构建的查询: {contextualized_query}")

    initial_results = milvus_service.search_by_text(
        "RAG_learn",
        query,
        limit=k * 2,
        metric_type="COSINE",
        output_fields=["text", "metadata", "score"],
    )

    ranked_results = []

    for doc in initial_results:
        context_relevance = score_document_context_relevance(
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


def score_document_relevance(query, document):
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

    response = client.generate_text(
        prompt=user_prompt, system_instruction=system_prompt, temperature=0
    )

    score_text = response.strip()

    match = re.search(r"(\d+(\.\d+)?)", score_text)

    if match:
        score = float(match.group(1))
        return min(10, max(0, score))
    else:
        return 5.0


def score_document_context_relevance(query, context, document):
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

    response = client.generate_text(
        prompt=user_prompt, system_instruction=system_prompt, temperature=0
    )

    score_text = response.strip()

    match = re.search(r"(\d+(\.\d+)?)", score_text)

    if match:
        score = float(match.group(1))
        return min(10, max(0, score))
    else:
        return 5.0


def adaptive_retrieval(query, k=4, user_context=None):
    query_type = classify_query(query)
    print(f"查询类型: {query_type}")

    match query_type:
        case "Factual":
            results = factual_retrieval_strategy(query, k)
        case "Analtytical":
            results = analytical_retrieval_strategy(query, k)
        case "Opinion":
            results = opinion_retrieval_strategy(query, k)
        case "Contextual":
            results = contextual_retrieval_strategy(query, k, user_context)
        case _:
            results = factual_retrieval_strategy(query, k)
    return results


def generate_response(query, results, query_type):
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

    response = client.generate_text(
        prompt=user_prompt, system_instruction=system_prompt, temperature=0.2
    )

    return response


def rag_with_adaptive_retrieval(file_path, query, k=4, user_context=None):
    print("\n=== RAG WITH ADAPTIVE RETRIEVAL ===")

    # chunks = process_document(file_path)

    query_type = classify_query(query)
    retrieved_docs = adaptive_retrieval(query, k, user_context)

    response = generate_response(query, retrieved_docs, query_type)

    result = {
        "query": query,
        "query_type": query_type,
        "retrieved_docs": retrieved_docs,
        "response": response,
    }

    return result


if __name__ == "__main__":
    result = rag_with_adaptive_retrieval("Agent基础.md", "思维链(CoT)是什么？举个例子")
    print(result)
