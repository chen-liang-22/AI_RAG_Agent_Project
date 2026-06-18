"""考试模块请求和响应模型。

这个文件只定义当前对话式考试流程使用的数据结构：
- 前端先选择向量库、文件和一级目录；
- 后端随机抽题并生成一场考试会话；
- 用户逐轮作答，系统逐轮保存题目、答案、评分和分析。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


# 考试题型枚举：
# single_choice：单选题；multiple_choice：多选题；true_false：判断题；
# short_answer：简答题；fill_blank：填空题。
ExamQuestionType = Literal["single_choice", "multiple_choice", "true_false", "short_answer", "fill_blank"]


class ExamSectionResponse(BaseModel):
    """考试题源目录响应。"""

    section_path: str  # 目录路径
    question_count: int  # 该目录下可抽取的题目数量


class ExamSectionsResponse(BaseModel):
    """考试题源目录列表响应。"""

    collection_name: str  # 向量库名称
    document_id: str | None = None  # 文件编号
    sections: list[ExamSectionResponse] = Field(default_factory=list)  # 目录列表


class ExamStartRequest(BaseModel):
    """开始对话式考试请求体。"""

    collection_name: str | None = None  # 题源所在 Qdrant collection；为空时使用默认 collection
    document_id: str | None = None  # 指定题源文件；为空时从整个 collection 抽题
    section_path: str | None = None  # 指定目录路径；为空时不按目录过滤
    user_id: str | None = None  # 考试用户编号
    round_count: int = Field(default=5, ge=1, le=50)  # 考试轮数
    question_types: list[ExamQuestionType] = Field(
        default_factory=lambda: ["single_choice", "multiple_choice", "true_false", "short_answer", "fill_blank"],
    )  # 允许随机生成的题型
    model_mode: str | None = None  # 主观题分析模型档位
    seed: int | None = None  # 随机种子；传入后同范围抽题更容易复现


class ExamQuestionSource(BaseModel):
    """考试题目来源信息。"""

    document_id: str | None = None  # 来源文件编号
    filename: str | None = None  # 来源文件名
    section_path: str | None = None  # 来源目录路径
    source_page: int | None = None  # 来源页码


class ExamConversationQuestion(BaseModel):
    """前端对话式考试展示题目。"""

    exam_question_id: str  # 本场考试内的题目编号
    round_no: int  # 当前轮次，从 1 开始
    question_type: ExamQuestionType  # 题型
    prompt: str  # 题目文本
    options: list[str] = Field(default_factory=list)  # 选项；非选择题为空
    max_score: float  # 本题满分，所有轮次加起来总分为 100
    source: ExamQuestionSource  # 来源信息


class ExamSessionSummary(BaseModel):
    """考试会话摘要。"""

    session_id: str  # 考试会话编号
    user_id: str | None = None  # 用户编号
    title: str  # 会话标题
    collection_name: str  # 向量库名称
    document_id: str | None = None  # 文件编号
    filename: str | None = None  # 文件名
    section_path: str | None = None  # 目录路径
    round_count: int  # 总轮数
    answered_count: int  # 已答题数
    total_score: float  # 当前总分
    max_score: float  # 满分
    status: str  # active/completed
    current_round: int  # 当前轮次
    created_at: str  # 创建时间
    updated_at: str  # 更新时间
    completed_at: str | None = None  # 完成时间


class ExamStartResponse(BaseModel):
    """开始对话式考试响应。"""

    session: ExamSessionSummary  # 考试会话摘要
    current_question: ExamConversationQuestion  # 第一轮题目


class ExamAnswerRequest(BaseModel):
    """提交单轮考试答案请求体。"""

    answer: str | list[str] = Field(default="")  # 用户答案；多选题可传字符串数组


class ExamAnswerAnalysis(BaseModel):
    """单题作答分析。"""

    is_correct: bool  # 是否判定正确
    score: float  # 本题得分
    max_score: float  # 本题满分
    correct_answer: Any = None  # 正确答案
    reference_answer: str  # 原始参考答案
    hit_points: list[str] = Field(default_factory=list)  # 命中要点
    missing_points: list[str] = Field(default_factory=list)  # 遗漏点
    wrong_points: list[str] = Field(default_factory=list)  # 错误点
    comment: str = ""  # 点评


class ExamAnswerResponse(BaseModel):
    """提交答案响应。"""

    session: ExamSessionSummary  # 更新后的考试会话
    answered_question: ExamConversationQuestion  # 已作答题目
    analysis: ExamAnswerAnalysis  # 本题分析
    next_question: ExamConversationQuestion | None = None  # 下一轮题目；考试完成时为空


class ExamHistoryListResponse(BaseModel):
    """考试历史分页响应。"""

    items: list[ExamSessionSummary]  # 当前页考试记录
    total: int  # 总数
    page: int  # 当前页
    page_size: int  # 每页数量


class ExamQuestionRecord(BaseModel):
    """考试详情中的单题记录。"""

    question: ExamConversationQuestion  # 题目信息
    user_answer: str | None = None  # 用户答案
    analysis: ExamAnswerAnalysis | None = None  # 分析结果；未作答时为空
    answered_at: str | None = None  # 作答时间


class ExamSessionDetailResponse(BaseModel):
    """考试会话详情响应。"""

    session: ExamSessionSummary  # 考试会话摘要
    questions: list[ExamQuestionRecord]  # 全部题目和用户答案
