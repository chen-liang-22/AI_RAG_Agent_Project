"""V2 销售训练画像服务测试。"""

from app_v2.application.training import profile_service as profile_service_module
from app_v2.application.training.profile_service import (
    PROFILE_DICTIONARY_CODES,
    TrainingProfileApplicationService,
)


class FakeProfileDictionaryRepository:
    """测试用 V2 字典仓储，记录画像服务到底查询了哪些字典。"""

    def __init__(self):
        self.calls: list[str | None] = []

    def list_items(self, dictionary_code: str | None = None) -> list[dict]:
        """返回最小可组装成树形字典的行数据。"""

        self.calls.append(dictionary_code)
        if dictionary_code == "student_portrait":
            return [
                {
                    "dictionary_item_id": "student_root",
                    "dictionary_code": "student_portrait",
                    "dictionary_name": "学员画像",
                    "item_code": "position_role",
                    "item_name": "岗位角色",
                    "parent_item_id": None,
                    "item_level": 1,
                    "sort_order": 1,
                    "enabled": 1,
                    "description": None,
                    "metadata_json": None,
                },
                {
                    "dictionary_item_id": "student_child",
                    "dictionary_code": "student_portrait",
                    "dictionary_name": "学员画像",
                    "item_code": "overseas_bd",
                    "item_name": "海外BD",
                    "parent_item_id": "student_root",
                    "item_level": 2,
                    "sort_order": 1,
                    "enabled": 1,
                    "description": None,
                    "metadata_json": '{"input_type": "select"}',
                },
            ]
        if dictionary_code == "training_batch_status":
            return [
                {
                    "dictionary_item_id": "batch_status_published",
                    "dictionary_code": "training_batch_status",
                    "dictionary_name": "训练资料状态",
                    "item_code": "published",
                    "item_name": "已发布",
                    "parent_item_id": None,
                    "item_level": 1,
                    "sort_order": 1,
                    "enabled": 1,
                    "description": None,
                    "metadata_json": None,
                }
            ]
        return []


def test_training_profile_dictionaries_use_v2_dictionary_repository(monkeypatch):
    """画像字典查询应该走 V2 字典仓储，不再直接读取旧 KnowledgeStore。"""

    def fail_if_old_store_is_used():
        raise AssertionError("销售训练画像字典不应该继续创建旧 KnowledgeStore")

    monkeypatch.setattr(profile_service_module, "get_knowledge_store", fail_if_old_store_is_used, raising=False)
    repository = FakeProfileDictionaryRepository()

    service = TrainingProfileApplicationService(
        service=object(),
        dictionary_repository=repository,
    )
    groups = service.list_profile_dictionaries()

    assert repository.calls == list(PROFILE_DICTIONARY_CODES)
    assert [group.dictionary_code for group in groups] == ["student_portrait", "training_batch_status"]
    assert groups[0].items[0].item_code == "position_role"
    assert groups[0].items[0].children[0].item_code == "overseas_bd"
    assert groups[0].items[0].children[0].metadata == {"input_type": "select"}
