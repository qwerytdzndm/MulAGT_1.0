from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mulagt.agent_interface import run_three_agent_code_generation
from mulagt.config import Settings
from mulagt.models import IssueRequest
from mulagt.mul_tools import PatchApplier
from mulagt.rag.parsers import ALLOWED_SUFFIXES


GENERATION_SEED_MARKER = "MULAGT_GENERATION_SEED"
PYTHON_DEMO_CONSTRAINTS = [
    "Generate Python files only for a new empty project.",
    "Use main.py as the runnable entry point.",
    "Prefer Python standard library modules; use tkinter for simple GUI games.",
    "Do not create JavaScript, HTML, CSS, or browser-only output unless the existing project already uses them.",
]
IGNORED_INDEX_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    ".mul",
    ".mulagt",
}


def _prompt_if_missing(value: str | None, label: str) -> str:
    if value and value.strip():
        return value.strip()
    entered = input(f"{label}: ").strip()
    if not entered:
        raise SystemExit(f"{label} must not be blank")
    return entered


def _write_transcript_markdown(
    path: Path,
    transcript: list[dict[str, Any]],
    translations: list[dict[str, Any]] | None = None,
) -> None:
    lines = ["# Three-Agent Transcript", ""]
    for index, message in enumerate(transcript, start=1):
        agent = message.get("agent", "unknown")
        role = message.get("role", "unknown")
        metadata = message.get("metadata") or {}
        lines.extend(
            [
                f"## {index}. {agent} / {role}",
                "",
                "```json",
                json.dumps(metadata, ensure_ascii=False, indent=2),
                "```",
                "",
                "```text",
                str(message.get("content", "")),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "---",
            "",
            "# Content 中文译文",
            "",
            "说明：这里翻译的是 transcript 中可见的 content 字段；模型隐藏的私有思维链不会被记录或展示。",
            "",
        ]
    )
    if translations:
        for item in translations:
            lines.extend(
                [
                    f"## ZH {item['index']}. {item['agent']} / {item['role']}",
                    "",
                    "```text",
                    str(item.get("content_zh", "")),
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(["未生成中文译文。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_patch(path: Path, proposals: list[dict[str, Any]]) -> None:
    patch = "\n".join(str(item.get("diff", "")) for item in proposals)
    path.write_text(patch.strip() + ("\n" if patch.strip() else ""), encoding="utf-8")


def _print_conversation(transcript: list[dict[str, Any]]) -> None:
    print("")
    print("===== Three-Agent Visible Conversation =====")
    for index, message in enumerate(transcript, start=1):
        agent = message.get("agent", "unknown")
        role = message.get("role", "unknown")
        metadata = message.get("metadata") or {}
        print("")
        print(f"[{index}] {agent} / {role}")
        if metadata:
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
        print(str(message.get("content", "")))
    print("")
    print("===== End Conversation =====")


def _translate_content_to_chinese(
    *,
    content: str,
    settings: Settings,
) -> str:
    if not content.strip():
        return ""
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=90.0,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the supplied transcript content into Simplified "
                        "Chinese. Preserve source code, JSON keys, file paths, "
                        "numbers, hashes, URLs, and command lines exactly. Translate "
                        "only natural-language prose. Return one JSON object with "
                        "this shape: {\"content_zh\":\"string\"}. Do not add analysis."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"content": content},
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=settings.llm_max_output_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return str(parsed.get("content_zh", "")).strip()
    except Exception as exc:
        return f"[中文翻译失败：{type(exc).__name__}]"


def _translate_transcript_contents(
    transcript: list[dict[str, Any]],
    settings: Settings,
) -> list[dict[str, Any]]:
    translations = []
    for index, message in enumerate(transcript, start=1):
        translations.append(
            {
                "index": index,
                "agent": message.get("agent", "unknown"),
                "role": message.get("role", "unknown"),
                "content_zh": _translate_content_to_chinese(
                    content=str(message.get("content", "")),
                    settings=settings,
                ),
            }
        )
    return translations


def _has_indexable_content(root: Path) -> bool:
    if not root.is_dir():
        return False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part.casefold() in IGNORED_INDEX_DIRS for part in relative_parts):
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        try:
            if path.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _seed_generation_project(root: Path, instruction: str) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    if _has_indexable_content(root):
        return []

    files = {
        "README.md": (
            f"# Mulagt Generated Project\n\n"
            f"{GENERATION_SEED_MARKER}\n\n"
            f"User instruction:\n{instruction}\n"
            "\nGeneration target:\n"
            "- Create a Python project.\n"
            "- Use main.py as the runnable entry point.\n"
            "- Prefer tkinter and the Python standard library.\n"
        ),
        "main.py": (
            f"# {GENERATION_SEED_MARKER}: replace this scaffold with a complete Python app.\n"
            "\"\"\"Generated by Mulagt.\n\n"
            f"User instruction: {instruction}\n"
            "\"\"\"\n\n"
            "\n"
            "def main() -> None:\n"
            "    print(\"Mulagt is generating this Python app.\")\n\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "    main()\n"
        ),
    }
    created = []
    for relative_path, content in files.items():
        path = root / relative_path
        if not path.exists():
            path.write_text(content, encoding="utf-8", newline="")
            created.append(relative_path)
    return created


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Start Mulagt's three-agent DeepSeek code-generation workflow "
            "and save the full conversation."
        )
    )
    parser.add_argument(
        "--mul-path",
        help="Target code directory. When omitted, the script asks interactively.",
    )
    parser.add_argument(
        "--instruction",
        help="Code-generation instruction. When omitted, the script asks interactively.",
    )
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Optional extra constraint. Can be provided multiple times.",
    )
    parser.add_argument(
        "--test-command",
        default="python -m unittest discover -s tests",
        help="Recorded verification command for the planning context.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for result.json, transcript.md, and proposal.patch.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a local static server from the target directory after generation.",
    )
    parser.add_argument(
        "--show-conversation",
        action="store_true",
        help="Print the full visible agent conversation after generation.",
    )
    parser.add_argument(
        "--no-translate-transcript",
        action="store_true",
        help="Do not append Chinese translations of transcript content.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port used with --serve.",
    )
    args = parser.parse_args()

    mul_path = _prompt_if_missing(args.mul_path, "目标代码目录 / Target code directory")
    instruction = _prompt_if_missing(args.instruction, "启动指令 / Startup instruction")
    mul_root = Path(mul_path).resolve()
    seeded_files = _seed_generation_project(mul_root, instruction)

    settings = replace(Settings.from_env(), llm_mode="deepseek")
    if not settings.deepseek_api_key:
        print("DEEPSEEK_API_KEY is required for this DeepSeek startup file.")
        print("Set it in the environment or in .env, then run again.")
        return 2

    run_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else settings.runtime_dir
        / "three_agent_runs"
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    result = run_three_agent_code_generation(
        IssueRequest(
            mul_path=str(mul_root),
            issue=instruction,
            constraints=[
                *args.constraint,
                *(PYTHON_DEMO_CONSTRAINTS if seeded_files else []),
            ],
            test_command=args.test_command,
            llm_mode="deepseek",
        ),
        settings=settings,
    )

    apply_result = None
    review = result.get("review") or {}
    if review.get("approved") and result.get("proposals"):
        apply_result = PatchApplier(settings.runtime_dir).apply(
            str(mul_root),
            result["run_id"],
            result["proposals"],
        )
        result["apply_result"] = apply_result
        result["status"] = "applied"
    result["seeded_files"] = seeded_files
    translations = (
        []
        if args.no_translate_transcript
        else _translate_transcript_contents(result["transcript"], settings)
    )
    result["transcript_content_zh"] = translations

    result_path = run_dir / "result.json"
    transcript_path = run_dir / "transcript.md"
    patch_path = run_dir / "proposal.patch"

    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_transcript_markdown(
        transcript_path,
        result["transcript"],
        translations,
    )
    _write_patch(patch_path, result["proposals"])

    print(f"run_id: {result['run_id']}")
    print(f"status: {result['status']}")
    print(f"review approved: {review.get('approved')}")
    print(f"proposals: {len(result.get('proposals', []))}")
    if seeded_files:
        print(f"seeded files: {', '.join(seeded_files)}")
    if apply_result:
        print(f"applied files: {', '.join(apply_result['changed_files'])}")
    elif result.get("proposals"):
        print("applied files: none; reviewer did not approve the proposal")
    print(f"result: {result_path}")
    print(f"transcript: {transcript_path}")
    print(f"patch: {patch_path}")
    if args.show_conversation:
        _print_conversation(result["transcript"])
    index_path = mul_root / "index.html"
    if index_path.is_file():
        print(f"open html: {index_path}")
        if args.serve:
            print(f"serving: http://127.0.0.1:{args.port}/index.html")
            print("Press Ctrl+C to stop the server.")
            return subprocess.call(
                [sys.executable, "-m", "http.server", str(args.port)],
                cwd=str(mul_root),
            )
    main_path = mul_root / "main.py"
    if main_path.is_file():
        print(f"run python: {sys.executable} {main_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
