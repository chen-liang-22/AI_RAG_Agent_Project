import pytest

from rag.query_planner import QueryPlannerModelError, QueryPlannerService


def test_explicit_questions_are_split_without_llm(monkeypatch):
    planner = QueryPlannerService(max_queries=6)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM planner should not be called for explicit multi-question input")

    monkeypatch.setattr(planner, "_plan_with_llm", fail_if_called)

    queries = planner.plan("怎么选扫地机器人？滤网多久换？")

    assert queries == ["怎么选扫地机器人", "滤网多久换"]


def test_planner_falls_back_when_model_fails(monkeypatch):
    planner = QueryPlannerService(max_queries=4)

    def raise_model_error(*args, **kwargs):
        raise QueryPlannerModelError("timeout")

    monkeypatch.setattr(planner, "_plan_with_llm", raise_model_error)

    queries = planner.plan("扫地机器人充电失败怎么办")

    assert queries == ["扫地机器人充电失败怎么办"]


def test_invalid_planner_json_raises_parse_error():
    planner = QueryPlannerService()

    with pytest.raises(ValueError):
        planner._parse_json_object("not json")


def test_initial_plan_uses_only_original_query():
    planner = QueryPlannerService()

    assert planner.plan_initial(" 大户型适合哪些扫地机器人？ ") == ["大户型适合哪些扫地机器人"]


def test_merge_queries_deduplicates_in_order():
    planner = QueryPlannerService(max_queries=4)

    queries = planner.merge_queries(["大户型怎么选"], ["大户型怎么选", "宠物家庭怎么选"])

    assert queries == ["大户型怎么选", "宠物家庭怎么选"]
