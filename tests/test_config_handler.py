def test_config_handler_does_not_keep_legacy_prompts_config():
    """旧 prompts.yml 配置机制删除后，基础配置模块仍应可导入。"""

    import core.utils.config_handler as config_handler

    assert isinstance(config_handler.rag_conf, dict)
    assert not hasattr(config_handler, "prompts_conf")
