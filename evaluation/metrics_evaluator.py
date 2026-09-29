"""评测指标计算：对已生成的答案计算 RAGAS 指标并输出汇总。"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from evaluation.metrics import RAGASEvaluator
from kb_agent.core.config import config


class MetricsEvaluator:
    """为已生成的答案计算 RAGAS 指标。"""

    def __init__(self, answers_file: Path, output_file: Path):
        self.answers_file = answers_file
        self.output_file = output_file
        self.ragas_evaluator = RAGASEvaluator(
            model_name=config.evaluation_model,
            api_key=config.model_api_key,
            base_url=config.model_base_url,
        )

    def load_answers(self) -> pd.DataFrame:
        print(f"Loading answers from {self.answers_file}...")
        df = pd.read_csv(self.answers_file)
        print(f"Loaded {len(df)} answers.")
        return df

    def save_incremental(self, df: pd.DataFrame) -> None:
        df.to_csv(self.output_file, index=False)

    def compute_metrics_for_row(
        self, row: pd.Series, idx: int, total: int
    ) -> Dict[str, Any]:
        question = row["question"]
        reference = row["reference"]
        generated_answer = row["generated_answer"]
        agent_error = row.get("agent_error")

        print(f"\n[{idx}/{total}] Computing metrics for: {question[:80]}...")

        has_agent_error = pd.notna(agent_error) and str(agent_error).strip() != ""
        has_no_answer = pd.isna(generated_answer) or str(generated_answer).strip() == ""

        if has_agent_error or has_no_answer:
            reason = "Skipped due to agent error" if has_agent_error else "No answer generated"
            print(f"  Skipping metrics: {reason}")
            return {
                "#": row["#"],
                "question": question,
                "reference": reference,
                "source": row["source"],
                "generated_answer": generated_answer,
                "answer_relevancy": None,
                "answer_similarity": None,
                "answer_correctness": None,
                "agent_error": agent_error if has_agent_error else None,
                "metrics_error": reason,
                "timestamp": row["timestamp"],
            }

        print("  Computing RAGAS metrics...")
        metrics = self.ragas_evaluator.evaluate_single(
            question=question,
            answer=generated_answer,
            ground_truth=reference,
        )
        metrics_error = metrics.pop("error", None)
        print(f"  Metrics: {metrics}")

        return {
            "#": row["#"],
            "question": question,
            "reference": reference,
            "source": row["source"],
            "generated_answer": generated_answer,
            "answer_relevancy": metrics.get("answer_relevancy"),
            "answer_similarity": metrics.get("answer_similarity"),
            "answer_correctness": metrics.get("answer_correctness"),
            "agent_error": agent_error,
            "metrics_error": metrics_error,
            "timestamp": row["timestamp"],
        }

    def save_summary_statistics(self, final_df: pd.DataFrame) -> Path:
        summary_file = self.output_file.parent / self.output_file.name.replace(
            "evaluation_results_", "summary_statistics_"
        )
        successful = final_df[
            (final_df["agent_error"].isna()) & (final_df["metrics_error"].isna())
        ]
        total_cases = len(final_df)

        summary = {
            "Total Test Cases": [total_cases],
            "Successful Evaluations": [len(successful)],
            "Agent Errors": [int(final_df["agent_error"].notna().sum())],
            "Metrics Errors": [int(final_df["metrics_error"].notna().sum())],
            "Success Rate (%)": [
                round((len(successful) / total_cases) * 100, 2) if total_cases else 0
            ],
        }
        for metric in ["answer_relevancy", "answer_similarity", "answer_correctness"]:
            key = f"Mean {metric.replace('_', ' ').title()}"
            summary[key] = [round(successful[metric].mean(), 4)] if len(successful) else [None]

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(summary_file, index=False)
        return summary_file

    def compute_all_metrics(self) -> Path:
        print("=" * 80)
        print("COMPUTING METRICS")
        print("=" * 80)

        answers_df = self.load_answers()
        final_results = []
        total = len(answers_df)

        for idx, row in answers_df.iterrows():
            result = self.compute_metrics_for_row(row, idx + 1, total)
            final_results.append(result)
            self.save_incremental(pd.DataFrame(final_results))

        final_df = pd.DataFrame(final_results)
        summary_file = self.save_summary_statistics(final_df)

        successful = final_df[
            (final_df["agent_error"].isna()) & (final_df["metrics_error"].isna())
        ]
        if len(successful):
            print("\nMetric Statistics (mean):")
            print("-" * 40)
            for metric in ["answer_relevancy", "answer_similarity", "answer_correctness"]:
                print(f"  {metric}: {successful[metric].mean():.4f}")

        print(f"\nSummary statistics saved to: {summary_file}")
        return self.output_file
