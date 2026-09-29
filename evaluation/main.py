"""评测主入口：两阶段（生成答案 → 计算指标）。

用法（在项目根目录执行）：
    python -m evaluation.main
"""

import asyncio
from datetime import datetime
from pathlib import Path

from evaluation.answer_generator import AnswerGenerator
from evaluation.metrics_evaluator import MetricsEvaluator
from kb_agent.agent.create_agent import build_agent


async def main() -> None:
    input_file = Path("data/evaluation_documents/expected_behaviour.xlsx")
    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    answers_file = results_dir / f"answers_{timestamp}.csv"
    final_file = results_dir / f"evaluation_results_{timestamp}.csv"

    # PHASE 1: 生成答案
    print("\n" + "=" * 80)
    print("STARTING EVALUATION PIPELINE")
    print("=" * 80)

    agent = await build_agent()
    generator = AnswerGenerator(input_file, answers_file, agent)
    await generator.generate_all_answers()

    # PHASE 2: 计算指标
    print("\n" + "=" * 80)
    print("PHASE 2: COMPUTING METRICS")
    print("=" * 80)

    evaluator = MetricsEvaluator(answers_file, final_file)
    evaluator.compute_all_metrics()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Answers file: {answers_file}")
    print(f"Final results: {final_file}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
