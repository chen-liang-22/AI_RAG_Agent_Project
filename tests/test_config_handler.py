def test_config_handler_does_not_expose_prompt_config_directly():
    """基础配置模块仍只暴露业务配置，提示词由 PromptManager 单独管理。"""

    import core.utils.config_handler as config_handler

    assert isinstance(config_handler.rag_conf, dict)
    assert not hasattr(config_handler, "prompts_conf")


def test_config_handler_exposes_consolidated_config_sections():
    """配置入口应该从 app/storage/training 三个文件聚合出业务常用配置。"""

    import core.utils.config_handler as config_handler

    assert config_handler.app_conf["rag"]["chat_model_name"] == config_handler.rag_conf["chat_model_name"]
    assert config_handler.storage_conf["qdrant"]["collection_name"] == config_handler.qdrant_conf["collection_name"]
    assert "database" in config_handler.storage_conf
    assert "minio" in config_handler.storage_conf
    assert "redis" in config_handler.storage_conf
    assert "collections" in config_handler.training_conf
