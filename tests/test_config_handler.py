def test_config_handler_imports_without_optional_prompts_config():
    """提示词配置缺失时，基础配置模块仍应可导入。"""

    import core.utils.config_handler as config_handler

    assert isinstance(config_handler.prompts_conf, dict)
