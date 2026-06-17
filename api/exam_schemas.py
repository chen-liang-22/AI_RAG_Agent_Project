from pydantic import BaseModel, Field


class ExamGenerateRequest(BaseModel):
    """考试试卷生成请求体。"""

    collection_name: str | None = None  # 题源所在 Qdrant collection；为空时使用默认 collection
    document_id: str | None = None  # 指定题源文件；为空时在 collection 内抽题
    section_path: str | None = None  # 指定章节路径；为空时不按章节过滤
    mode: str = "random_practice"  # 组卷模式：random_practice/chapter_test
    question_count: int = Field(default=5, ge=1, le=50)  # 需要生成的题目数量
    score_per_question: int = Field(default=20, ge=1, le=100)  # 单题分值
    difficulty: str = "medium"  # 难度，第一阶段仅透传给前端展示
    seed: int | None = None  # 随机种子；传入后同一范围内可复现抽题结果
    question_types: list[str] = Field(default_factory=lambda: ["short_answer"])  # 题型列表，第一阶段支持简答题


class ExamQuestionSource(BaseModel):
    """考试题目来源信息。"""

    document_id: str | None = None  # 来源文件编号
    filename: str | None = None  # 来源文件名
    section_path: str | None = None  # 来源章节路径
    source_page: int | None = None  # 来源页码


class ExamQuestionResponse(BaseModel):
    """生成试卷中的单道题目。"""

    question_id: str  # 本次试卷中的题目编号
    source_question_id: str | None = None  # 原始知识库题目编号
    question_type: str = "short_answer"  # 题型
    difficulty: str = "medium"  # 难度
    question: str  # 题干
    options: list[str] = Field(default_factory=list)  # 选项，简答题为空
    reference_answer: str  # 参考答案
    score: int  # 本题分值
    source: ExamQuestionSource  # 来源信息


class ExamGenerateResponse(BaseModel):
    """考试试卷生成响应体。"""

    paper_id: str  # 本次生成的试卷编号
    title: str  # 试卷标题
    total_score: int  # 试卷总分
    questions: list[ExamQuestionResponse]  # 题目列表


class ExamAnswerRequest(BaseModel):
    """单道题用户答案。"""

    question_id: str = Field(..., min_length=1)  # 题目编号
    answer: str = ""  # 用户答案
    question: str = ""  # 题干，方便后端无状态评分
    reference_answer: str = ""  # 参考答案，方便后端无状态评分
    max_score: int = Field(default=0, ge=0)  # 本题满分


class ExamGradeRequest(BaseModel):
    """考试批量评分请求体。"""

    paper_id: str = Field(..., min_length=1)  # 试卷编号
    user_id: str | None = None  # 答题用户编号
    model_mode: str | None = None  # 评分模型档位
    answers: list[ExamAnswerRequest] = Field(default_factory=list)  # 用户答案列表


class ExamGradeResult(BaseModel):
    """单题结构化评分结果。"""

    question_id: str  # 题目编号
    score: float  # 实际得分
    max_score: int  # 本题满分
    hit_points: list[str] = Field(default_factory=list)  # 命中要点
    missing_points: list[str] = Field(default_factory=list)  # 缺失要点
    wrong_points: list[str] = Field(default_factory=list)  # 错误点
    comment: str = ""  # 评分点评
    review_suggestion: str = ""  # 复习建议


class ExamGradeResponse(BaseModel):
    """考试批量评分响应体。"""

    paper_id: str  # 试卷编号
    total_score: float  # 总得分
    max_score: int  # 满分
    results: list[ExamGradeResult]  # 单题评分结果
