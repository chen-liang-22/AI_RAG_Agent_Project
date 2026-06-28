"""V2 考试仓储。"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, or_, select

from app.domain.entities import DocumentEntity, ExamQuestionEntity, ExamSessionEntity
from app.infrastructure.id_generator import new_id
from app.infrastructure.orm_session import orm_session_context
from app.application.training_support.repository import utc_now


class ExamRepository:
    """管理考试会话、题目、作答和历史记录的仓储。"""

    def get_document(self, document_id: str) -> DocumentEntity | None:
        """按文档编号查询考试题源文件。"""

        with orm_session_context() as session:
            return session.get(DocumentEntity, document_id)

    def create_exam_session(
            self,
            *,
            session_id: str | None = None,
            user_id: str | None = None,
            title: str | None = None,
            collection_name: str,
            document_id: str | None = None,
            filename: str | None = None,
            section_path: str | None = None,
            round_count: int,
            question_types: list[str],
            model_mode: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ExamSessionEntity:
        """创建考试会话。"""

        now = utc_now()
        clean_session_id = (session_id or "").strip() or new_id()
        exam_session = ExamSessionEntity(
            session_id=clean_session_id,
            user_id=user_id,
            title=(title or "").strip()[:120] or "知识掌握度测评",
            collection_name=collection_name,
            document_id=document_id,
            filename=filename,
            section_path=section_path,
            round_count=int(round_count),
            question_types_json=json.dumps(question_types, ensure_ascii=False),
            status="active",
            current_round=1,
            answered_count=0,
            total_score=0,
            max_score=100,
            model_mode=model_mode,
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False) if metadata else None,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )
        with orm_session_context() as session:
            session.add(exam_session)
        created = self.get_exam_session(clean_session_id)
        if created is None:
            raise RuntimeError(f"考试会话创建失败：{clean_session_id}")
        return created

    def add_exam_question(
            self,
            *,
            session_id: str,
            round_no: int,
            source_question_id: str | None,
            source_document_id: str | None,
            source_filename: str | None,
            source_page: int | None,
            section_path: str | None,
            question_type: str,
            prompt: str,
            options: list[str] | None,
            correct_answer: Any,
            reference_answer: str,
            max_score: float,
    ) -> ExamQuestionEntity:
        """新增考试题目。"""

        now = utc_now()
        exam_question_id = new_id()
        question = ExamQuestionEntity(
            exam_question_id=exam_question_id,
            session_id=session_id,
            round_no=int(round_no),
            source_question_id=source_question_id,
            source_document_id=source_document_id,
            source_filename=source_filename,
            source_page=source_page,
            section_path=section_path,
            question_type=question_type,
            prompt=prompt,
            options_json=json.dumps(options or [], ensure_ascii=False),
            correct_answer_json=json.dumps(correct_answer, ensure_ascii=False),
            reference_answer=reference_answer,
            user_answer=None,
            is_correct=None,
            score=None,
            max_score=float(max_score),
            analysis_json=None,
            status="pending",
            created_at=now,
            answered_at=None,
        )
        with orm_session_context() as session:
            session.add(question)
        created = self.get_exam_question(exam_question_id=exam_question_id)
        if created is None:
            raise RuntimeError(f"考试题目保存失败：{exam_question_id}")
        return created

    def get_exam_session(self, session_id: str) -> ExamSessionEntity | None:
        """查询考试会话详情。"""

        with orm_session_context() as session:
            return session.get(ExamSessionEntity, session_id)

    def get_exam_question(
            self,
            *,
            exam_question_id: str | None = None,
            session_id: str | None = None,
            round_no: int | None = None,
    ) -> ExamQuestionEntity | None:
        """按题目编号或会话轮次查询考试题目。"""

        with orm_session_context() as session:
            if exam_question_id:
                return session.get(ExamQuestionEntity, exam_question_id)
            if session_id and round_no is not None:
                statement = select(ExamQuestionEntity).where(
                    ExamQuestionEntity.session_id == session_id,
                    ExamQuestionEntity.round_no == int(round_no),
                )
                return session.scalars(statement).first()
            return None

    def list_exam_questions(self, session_id: str) -> list[ExamQuestionEntity]:
        """查询会话下的考试题目列表。"""

        statement = (
            select(ExamQuestionEntity)
            .where(ExamQuestionEntity.session_id == session_id)
            .order_by(ExamQuestionEntity.round_no.asc())
        )
        with orm_session_context() as session:
            return list(session.scalars(statement).all())

    def delete_exam_session(self, session_id: str) -> bool:
        """删除考试会话及题目明细。"""

        with orm_session_context() as session:
            exam_session = session.get(ExamSessionEntity, session_id)
            if exam_session is None:
                return False

            questions = session.scalars(
                select(ExamQuestionEntity).where(ExamQuestionEntity.session_id == session_id)
            ).all()
            for question in questions:
                session.delete(question)
            session.delete(exam_session)
        return True

    def answer_exam_question(
            self,
            *,
            session_id: str,
            exam_question_id: str,
            user_answer: str,
            is_correct: bool,
            score: float,
            analysis: dict[str, Any],
    ) -> ExamQuestionEntity:
        """保存单题作答和阅卷结果。"""

        now = utc_now()
        with orm_session_context() as session:
            question = session.get(ExamQuestionEntity, exam_question_id)
            if question is None or question.session_id != session_id:
                raise RuntimeError(f"考试题目不存在：{exam_question_id}")
            question.user_answer = user_answer
            question.is_correct = 1 if is_correct else 0
            question.score = float(score)
            question.analysis_json = json.dumps(analysis, ensure_ascii=False)
            question.status = "answered"
            question.answered_at = now
            session.flush()

            answered_count = int(
                session.scalar(
                    select(func.count()).select_from(ExamQuestionEntity).where(
                        ExamQuestionEntity.session_id == session_id,
                        ExamQuestionEntity.status == "answered",
                    )
                )
                or 0
            )
            total_score = float(
                session.scalar(
                    select(func.coalesce(func.sum(ExamQuestionEntity.score), 0)).where(
                        ExamQuestionEntity.session_id == session_id,
                        ExamQuestionEntity.status == "answered",
                    )
                )
                or 0
            )
            exam_session = session.get(ExamSessionEntity, session_id)
            if exam_session is not None:
                round_count = int(exam_session.round_count or 0)
                completed = answered_count >= round_count > 0
                exam_session.answered_count = answered_count
                exam_session.total_score = total_score
                exam_session.current_round = min(answered_count + 1, round_count)
                exam_session.status = "completed" if completed else "active"
                exam_session.updated_at = now
                exam_session.completed_at = now if completed else None

        answered = self.get_exam_question(exam_question_id=exam_question_id)
        if answered is None:
            raise RuntimeError(f"考试题目不存在：{exam_question_id}")
        return answered

    def list_exam_sessions(
            self,
            *,
            page: int,
            page_size: int,
            user_id: str | None = None,
            keyword: str | None = None,
    ) -> tuple[list[ExamSessionEntity], int]:
        """分页查询考试历史会话。"""

        final_page = max(1, int(page))
        final_page_size = max(1, min(int(page_size), 50))
        offset = (final_page - 1) * final_page_size
        conditions = []
        if user_id:
            conditions.append(ExamSessionEntity.user_id == user_id)
        clean_keyword = (keyword or "").strip()
        if clean_keyword:
            like_keyword = f"%{self._escape_like_keyword(clean_keyword)}%"
            conditions.append(
                or_(
                    ExamSessionEntity.title.like(like_keyword, escape="\\"),
                    ExamSessionEntity.filename.like(like_keyword, escape="\\"),
                    ExamSessionEntity.section_path.like(like_keyword, escape="\\"),
                )
            )

        count_statement = select(func.count()).select_from(ExamSessionEntity)
        list_statement = (
            select(ExamSessionEntity)
            .order_by(ExamSessionEntity.updated_at.desc(), ExamSessionEntity.created_at.desc())
            .limit(final_page_size)
            .offset(offset)
        )
        if conditions:
            count_statement = count_statement.where(*conditions)
            list_statement = list_statement.where(*conditions)
        with orm_session_context() as session:
            total = int(session.scalar(count_statement) or 0)
            rows = list(session.scalars(list_statement).all())
        return rows, total

    @staticmethod
    def _escape_like_keyword(keyword: str) -> str:
        """转义 LIKE 通配符，避免用户输入被当成模式语法。"""

        return keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def get_exam_repository() -> ExamRepository:
    """创建考试仓储。"""

    return ExamRepository()
