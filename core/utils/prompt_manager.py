"""LLM 提示词配置管理器。

提示词内容统一放在 config/prompts.yml，业务代码只通过 PromptManager 按 key 读取。
这样后续调 prompt 不需要进入业务代码里翻长字符串。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.utils.config_handler import CONFIG_DIR, load_yaml_config


class PromptManager:
    """提示词读取与模板渲染门面。

    使用外观模式收敛 YAML 读取、嵌套 key 查找、字符串模板渲染这三件事。
    调用方只需要知道 prompt key 和变量，不需要关心配置文件结构。
    """

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else CONFIG_DIR / "prompts.yml"
        self.config = load_yaml_config(self.config_path)

    def get(self, key: str) -> str:
        """按点分隔 key 获取提示词模板。"""

        value: Any = self.config
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"提示词配置不存在：{key}")
            value = value[part]
        if not isinstance(value, str):
            raise TypeError(f"提示词配置必须是字符串：{key}")
        return value

    def render(self, key: str, **variables: Any) -> str:
        """读取提示词模板并用 format 变量渲染。"""

        template = self.get(key)
        try:
            return template.format(**variables)
        except KeyError as exc:
            missing_name = str(exc).strip("'")
            raise KeyError(f"提示词变量缺失：{key}.{missing_name}") from exc


prompt_manager = PromptManager()
