import unittest

from mulagt.context import (
    ContextCompressionError,
    ContextCompressor,
    estimate_json_tokens,
)


def make_compressor(**overrides):
    values = {
        "max_input_tokens": 2_048,
        "planner_tokens": 2_048,
        "coder_tokens": 2_048,
        "reviewer_tokens": 2_048,
        "reflection_tokens": 2_048,
        "max_evidence_item_tokens": 512,
        "max_file_item_tokens": 512,
        "max_diff_tokens": 512,
        "max_test_output_tokens": 512,
    }
    values.update(overrides)
    return ContextCompressor(**values)


class ContextCompressorTests(unittest.TestCase):
    def test_planner_prioritizes_grounded_paths_within_budget(self):
        compressor = make_compressor()
        mul_map = [
            {
                "path": f"src/module_{index:03d}.py",
                "language": "python",
                "symbols": [f"symbol_{item}" for item in range(40)],
            }
            for index in range(100)
        ]
        mul_map.append(
            {
                "path": "src/calculator.py",
                "language": "python",
                "symbols": ["safe_divide"],
            }
        )
        payload = {
            "issue": "Fix safe_divide in src/calculator.py. " + "detail " * 800,
            "constraints": ["Keep the public API stable. " * 80],
            "mul_map": mul_map,
            "evidence": [
                {
                    "evidence_id": "E-critical",
                    "chunk_id": "chunk-critical",
                    "document_id": "doc-critical",
                    "path": "src/calculator.py",
                    "source_type": "code",
                    "language": "python",
                    "start_line": 1,
                    "end_line": 20,
                    "rerank_score": 0.99,
                    "text": "def safe_divide(a, b):\n" + "    return a / b\n" * 600,
                }
            ],
        }

        compressed, stats = compressor.compress("change_plan", payload)

        self.assertLessEqual(estimate_json_tokens(compressed), 2_048)
        self.assertEqual(compressed["evidence"][0]["evidence_id"], "E-critical")
        self.assertEqual(compressed["mul_map"][0]["path"], "src/calculator.py")
        self.assertIn("context_notice", compressed)
        self.assertGreater(stats["estimated_tokens_before"], 2_048)
        self.assertGreater(stats["saved_tokens"], 0)

    def test_reviewer_preserves_every_changed_line(self):
        compressor = make_compressor()
        unchanged = "\n".join(
            f" unchanged context line {index}" for index in range(500)
        )
        diff = (
            "diff --git a/src/app.py b/src/app.py\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1,502 +1,502 @@\n"
            f"{unchanged}\n"
            "-old_behavior = unsafe_call()\n"
            "+new_behavior = validated_call()\n"
            " final unchanged line\n"
        )
        payload = {
            "issue": "Replace the unsafe call.",
            "constraints": [],
            "plan": {
                "summary": "Make a targeted replacement.",
                "steps": [
                    {
                        "step_id": "S1",
                        "description": "Replace the call.",
                        "target_files": ["src/app.py"],
                        "evidence_ids": ["E1"],
                        "risk": "low",
                        "verification": "Run tests.",
                    }
                ],
            },
            "evidence": [
                {
                    "evidence_id": "E1",
                    "chunk_id": "C1",
                    "document_id": "D1",
                    "path": "src/app.py",
                    "source_type": "code",
                    "language": "python",
                    "text": "old_behavior = unsafe_call()",
                    "rerank_score": 1.0,
                }
            ],
            "diffs": diff,
        }

        compressed, stats = compressor.compress("review_decision", payload)

        self.assertLessEqual(estimate_json_tokens(compressed), 2_048)
        self.assertIn("-old_behavior = unsafe_call()", compressed["diffs"])
        self.assertIn("+new_behavior = validated_call()", compressed["diffs"])
        self.assertIn("--- a/src/app.py", compressed["diffs"])
        self.assertIn("+++ b/src/app.py", compressed["diffs"])
        self.assertGreater(stats["truncated_fields"], 0)

    def test_reviewer_refuses_when_changed_lines_cannot_fit(self):
        compressor = make_compressor()
        additions = "\n".join(
            f"+changed_{index} = {'x' * 40}" for index in range(600)
        )
        payload = {
            "issue": "Apply a very large generated patch.",
            "constraints": [],
            "plan": {},
            "evidence": [],
            "diffs": (
                "diff --git a/src/generated.py b/src/generated.py\n"
                "--- a/src/generated.py\n"
                "+++ b/src/generated.py\n"
                "@@ -0,0 +1,600 @@\n"
                f"{additions}\n"
            ),
        }

        with self.assertRaises(ContextCompressionError):
            compressor.compress("review_decision", payload)

    def test_reflection_keeps_failure_output_tail(self):
        compressor = make_compressor()
        payload = {
            "issue": "Explain the failed verification.",
            "plan": {"summary": "Run the focused test.", "steps": []},
            "diffs": [
                "--- a/src/app.py\n"
                "+++ b/src/app.py\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new\n"
            ],
            "test_result": {
                "success": False,
                "returncode": 1,
                "stdout": ("setup output\n" * 500) + "FINAL FAILURE STDOUT\n",
                "stderr": ("trace frame\n" * 500) + "FINAL FAILURE STDERR\n",
            },
            "verification": {
                "success": False,
                "verdict": "failed_and_rolled_back",
            },
        }

        compressed, _ = compressor.compress("reflection", payload)

        self.assertLessEqual(estimate_json_tokens(compressed), 2_048)
        self.assertIn("FINAL FAILURE STDOUT", compressed["test_result"]["stdout"])
        self.assertIn("FINAL FAILURE STDERR", compressed["test_result"]["stderr"])


if __name__ == "__main__":
    unittest.main()
