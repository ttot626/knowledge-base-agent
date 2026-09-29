"""生成答案：把测试问题交给 Agent，逐条生成回答。"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


class AnswerGenerator:
    """用 Agent 为评测问题批量生成答案。"""

    def __init__(self, input_file: Path, output_file: Path, agent):
        self.input_file = input_file
        self.output_file = output_file
        self.agent = agent

    def load_test_cases(self) -> pd.DataFrame:
        print(f"Loading test cases from {self.input_file}...")
        df = pd.read_excel(self.input_file, sheet_name="Sheet1")
        print(f"Loaded {len(df)} test cases.")
        return df

    def save_incremental(self, df: pd.DataFrame) -> None:
        df.to_csv(self.output_file, index=False)

    async def generate_answer_for_question(
        self, question: str, timeout: int = 120
    ) -> Dict[str, Any]:
        try:
            result = await asyncio.wait_for(
                self.agent.ainvoke(
                    {"messages": [{"role": "user", "content": question}]}
                ),
                timeout=timeout,
            )
            return {"answer": result["messages"][-1].content, "error": None}
        except asyncio.TimeoutError:
            return {"answer": None, "error": f"Agent invocation timed out after {timeout}s"}
        except Exception as e:  # noqa: BLE001
            return {"answer": None, "error": f"Agent invocation failed: {str(e)[:200]}"}

    async def generate_all_answers(self) -> Path:
        print("=" * 80)
        print("PHASE 1: GENERATING ANSWERS")
        print("=" * 80)

        df = self.load_test_cases()
        answers = []
        total = len(df)

        for idx, row in df.iterrows():
            question = row["question"]
            reference = row["reference"]
            source = row["source"]

            print(f"\n[{idx + 1}/{total}] Generating answer for: {question[:80]}...")
            result = await self.generate_answer_for_question(question)
            generated = result["answer"]
            agent_error = result["error"]

            if generated:
                print(f"  Generated answer: {generated[:80]}...")
            else:
                print(f"  ERROR: {agent_error}")

            answers.append({
                "#": row["#"],
                "question": question,
                "reference": reference,
                "source": source,
                "generated_answer": generated,
                "agent_error": agent_error,
                "timestamp": datetime.now().isoformat(),
            })
            self.save_incremental(pd.DataFrame(answers))
            print(f"  Saved to {self.output_file}")

        answers_df = pd.DataFrame(answers)
        agent_errors = answers_df["agent_error"].notna().sum()
        print(f"\nAnswer generation complete: {total - int(agent_errors)}/{total} successful.")
        return self.output_file
