"""RAGAS 评测封装：用 LLM 打分评估生成答案的质量。

评测指标：
- answer_relevancy   答案相关性（答案是否紧扣问题）
- answer_similarity  答案相似度（与参考答案的语义相似程度）
- answer_correctness 答案正确性（此处设为 factual correctness，事实性）
"""

from typing import Dict, List

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_correctness, answer_relevancy, answer_similarity

# 只保留事实性维度，去掉语义相似度权重 → answer_correctness 变为 factual_correctness
answer_correctness.weights = [1, 0]


class RAGASEvaluator:
    """RAGAS 评测器的薄封装。"""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
        )
        self.metrics = [answer_relevancy, answer_similarity, answer_correctness]

    def evaluate_single(self, question: str, answer: str, ground_truth: str) -> Dict[str, float]:
        data = {
            "question": [question],
            "answer": [answer],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)

        try:
            result = evaluate(dataset=dataset, metrics=self.metrics, llm=self.llm)

            def extract_scalar(value):
                if isinstance(value, list) and len(value) > 0:
                    value = value[0]
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            return {
                "answer_relevancy": extract_scalar(result["answer_relevancy"]),
                "answer_similarity": extract_scalar(result["answer_similarity"]),
                "answer_correctness": extract_scalar(result["answer_correctness"]),
            }
        except Exception as e:  # noqa: BLE001
            return {
                "answer_relevancy": None,
                "answer_similarity": None,
                "answer_correctness": None,
                "error": str(e),
            }

    def evaluate_batch(
        self, questions: List[str], answers: List[str], ground_truths: List[str]
    ) -> Dict[str, List[float]]:
        data = {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)
        return evaluate(dataset=dataset, metrics=self.metrics, llm=self.llm)
