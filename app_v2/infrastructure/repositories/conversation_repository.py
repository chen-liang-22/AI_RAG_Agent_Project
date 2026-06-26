"""会话仓储。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, or_, select

from app_v2.shared.pagination import escape_like_keyword, normalize_page
from domain.entities import ConversationEntity, ConversationMessageEntity
from infrastructure.orm_session import orm_session_context
from training.repository import utc_now


class ConversationRepository:
    """封装聊天会话数据访问。

    这里使用仓储模式，把 conversations / conversation_messages 表的查询
    从旧 KnowledgeStore 中拆出来。应用服务只关心“查列表、查详情、删除”，
    不需要知道底层 SQL 怎么写。
    """

    def __init__(self, store: Any | None = None):
        # store 参数仅为旧测试和过渡调用保留，不再作为真实数据源。
        self.store = store

    def list_conversations(
        self,
        *,
        page: int,
        page_size: int,
        user_id: str | None = None,
        keyword: str | None = None,
    ) -> tuple[list[ConversationEntity], int]:
        """分页查询未删除的聊天会话。"""

        page_request = normalize_page(page, page_size, max_page_size=50)
        conditions = [ConversationEntity.status != "deleted"]
        if user_id:
            conditions.append(ConversationEntity.user_id == user_id)
        clean_keyword = (keyword or "").strip()
        if clean_keyword:
            like_keyword = f"%{escape_like_keyword(clean_keyword)}%"
            conditions.append(
                or_(
                    ConversationEntity.title.like(like_keyword, escape="\\"),
                    ConversationEntity.user_id.like(like_keyword, escape="\\"),
                    ConversationEntity.conversation_id.like(like_keyword, escape="\\"),
                )
            )

        count_statement = select(func.count()).select_from(ConversationEntity).where(*conditions)
        list_statement = (
            select(ConversationEntity)
            .where(*conditions)
            .order_by(
                func.coalesce(
                    ConversationEntity.last_message_at,
                    ConversationEntity.updated_at,
                    ConversationEntity.created_at,
                ).desc()
            )
            .limit(page_request.page_size)
            .offset(page_request.offset)
        )
        with orm_session_context() as session:
            total = int(session.scalar(count_statement) or 0)
            rows = session.scalars(list_statement).all()
        return rows, total

    def get_conversation(self, conversation_id: str) -> ConversationEntity | None:
        """按会话编号查询单个会话。"""

        with orm_session_context() as session:
            return session.get(ConversationEntity, conversation_id)

    def list_conversation_messages(self, conversation_id: str) -> list[ConversationMessageEntity]:
        """按顺序查询会话中的全部消息。"""

        statement = (
            select(ConversationMessageEntity)
            .where(ConversationMessageEntity.conversation_id == conversation_id)
            .order_by(ConversationMessageEntity.sequence_no.asc())
        )
        with orm_session_context() as session:
            return session.scalars(statement).all()

    def delete_conversation(self, conversation_id: str) -> bool:
        """软删除会话，并物理删除消息正文。"""

        now = utc_now()
        with orm_session_context() as session:
            conversation = session.get(ConversationEntity, conversation_id)
            if conversation is None or conversation.status == "deleted":
                return False
            messages = session.scalars(
                select(ConversationMessageEntity).where(
                    ConversationMessageEntity.conversation_id == conversation_id
                )
            ).all()
            for message in messages:
                session.delete(message)
            conversation.status = "deleted"
            conversation.message_count = 0
            conversation.updated_at = now
            conversation.last_message_at = now
            return True
