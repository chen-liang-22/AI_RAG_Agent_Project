"""会话仓储。"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, or_, select

from app_v2.shared.pagination import escape_like_keyword, normalize_page
from app_v2.domain.entities import ConversationEntity, ConversationMessageEntity
from app_v2.infrastructure.id_generator import new_id
from app_v2.infrastructure.orm_session import orm_session_context
from app_v2.application.training_support.repository import utc_now


class ConversationRepository:
    """封装聊天会话数据访问。

    这里使用仓储模式，把 conversations / conversation_messages 表的查询
    从旧 KnowledgeStore 中拆出来。应用服务只关心“查列表、查详情、删除”，
    不需要知道底层 SQL 怎么写。
    """

    def __init__(self, store: Any | None = None):
        """初始化聊天仓储。

        store 只为旧测试保留；真实查询统一通过 ORM Session 访问 conversations 表。
        """

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

    def ensure_conversation(
            self,
            *,
            conversation_id: str | None = None,
            user_id: str | None = None,
            title: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ConversationEntity:
        """确保会话存在，已删除会话不会复用原编号。"""

        clean_conversation_id = (conversation_id or "").strip()
        if clean_conversation_id:
            existing = self.get_conversation(clean_conversation_id)
            if existing is not None and existing.status != "deleted":
                return existing
            if existing is not None and existing.status == "deleted":
                clean_conversation_id = new_id()
        else:
            clean_conversation_id = new_id()

        now = utc_now()
        clean_title = (title or "").strip()[:80] or None
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata else None
        conversation = ConversationEntity(
            conversation_id=clean_conversation_id,
            user_id=user_id,
            title=clean_title,
            status="active",
            message_count=0,
            summary=None,
            metadata_json=metadata_json,
            created_at=now,
            updated_at=now,
            last_message_at=None,
        )
        with orm_session_context() as session:
            session.add(conversation)
        created = self.get_conversation(clean_conversation_id)
        if created is None:
            raise RuntimeError(f"Conversation {clean_conversation_id} was not created")
        return created

    def list_conversation_messages(self, conversation_id: str) -> list[ConversationMessageEntity]:
        """按顺序查询会话中的全部消息。"""

        statement = (
            select(ConversationMessageEntity)
            .where(ConversationMessageEntity.conversation_id == conversation_id)
            .order_by(ConversationMessageEntity.sequence_no.asc())
        )
        with orm_session_context() as session:
            return session.scalars(statement).all()

    def list_recent_messages(self, conversation_id: str, limit: int = 20) -> list[ConversationMessageEntity]:
        """查询最近若干条会话消息，并按正序返回。"""

        final_limit = max(1, min(int(limit), 100))
        statement = (
            select(ConversationMessageEntity)
            .where(ConversationMessageEntity.conversation_id == conversation_id)
            .order_by(ConversationMessageEntity.sequence_no.desc())
            .limit(final_limit)
        )
        with orm_session_context() as session:
            rows = session.scalars(statement).all()
        return list(reversed(rows))

    def add_message(
            self,
            *,
            conversation_id: str,
            role: str,
            content: str,
            content_type: str = "text",
            model_name: str | None = None,
            token_count: int | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> ConversationMessageEntity:
        """新增一条会话消息并刷新会话统计。"""

        if self.get_conversation(conversation_id) is None:
            self.ensure_conversation(conversation_id=conversation_id)

        now = utc_now()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata else None
        message_id = new_id()
        with orm_session_context() as session:
            next_sequence_no = int(
                session.scalar(
                    select(func.coalesce(func.max(ConversationMessageEntity.sequence_no), 0) + 1).where(
                        ConversationMessageEntity.conversation_id == conversation_id
                    )
                )
                or 1
            )
            message = ConversationMessageEntity(
                message_id=message_id,
                conversation_id=conversation_id,
                sequence_no=next_sequence_no,
                role=role,
                content=content,
                content_type=content_type,
                model_name=model_name,
                token_count=token_count,
                metadata_json=metadata_json,
                created_at=now,
            )
            session.add(message)
            conversation = session.get(ConversationEntity, conversation_id)
            if conversation is not None:
                conversation.message_count = int(conversation.message_count or 0) + 1
                conversation.updated_at = now
                conversation.last_message_at = now

        created = self.get_message(message_id)
        if created is None:
            raise RuntimeError(f"消息保存失败：{message_id}")
        return created

    def get_message(self, message_id: str) -> ConversationMessageEntity | None:
        """按消息编号查询单条消息。"""

        with orm_session_context() as session:
            return session.get(ConversationMessageEntity, message_id)

    def save_chat_exchange(
            self,
            *,
            conversation_id: str,
            user_message: str,
            assistant_message: str,
            model_name: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> None:
        """保存一问一答两条聊天消息。"""

        self.add_message(conversation_id=conversation_id, role="user", content=user_message)
        self.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_message,
            model_name=model_name,
            metadata=metadata,
        )

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
