from pathlib import Path

import pytest


def test_prompt_manager_renders_template_variables(tmp_path: Path):
    """PromptManager 应该能按分组 key 读取模板并替换变量。"""

    from core.utils.prompt_manager import PromptManager

    prompt_file = tmp_path / "prompts.yml"
    prompt_file.write_text(
        """
knowledge:
  answer:
    system: "你是{role}"
    user: |
      问题：{query}
      资料：{context}
""",
        encoding="utf-8",
    )

    manager = PromptManager(config_path=prompt_file)

    assert manager.render("knowledge.answer.system", role="客服") == "你是客服"
    assert manager.render("knowledge.answer.user", query="怎么选", context="参考内容") == "问题：怎么选\n资料：参考内容\n"


def test_prompt_manager_raises_clear_error_for_missing_key(tmp_path: Path):
    """缺少提示词 key 时应该报出明确错误，避免静默退回硬编码。"""

    from core.utils.prompt_manager import PromptManager

    prompt_file = tmp_path / "prompts.yml"
    prompt_file.write_text("knowledge: {}\n", encoding="utf-8")

    manager = PromptManager(config_path=prompt_file)

    with pytest.raises(KeyError, match="knowledge.answer.system"):
        manager.get("knowledge.answer.system")


def test_prompt_manager_raises_clear_error_for_missing_variable(tmp_path: Path):
    """模板变量缺失时应该指出具体 prompt key 和变量名，方便排查配置问题。"""

    from core.utils.prompt_manager import PromptManager

    prompt_file = tmp_path / "prompts.yml"
    prompt_file.write_text(
        """
rag:
  planner:
    user: "问题：{query}，历史：{history}"
""",
        encoding="utf-8",
    )

    manager = PromptManager(config_path=prompt_file)

    with pytest.raises(KeyError, match=r"rag\.planner\.user\.history"):
        manager.render("rag.planner.user", query="怎么清洁")
