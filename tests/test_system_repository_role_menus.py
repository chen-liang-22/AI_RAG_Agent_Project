from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

import pytest
from sqlalchemy import Index, create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.domain.entities import BaseOrmModel, SystemRoleMenuEntity
from app.infrastructure.repositories import system_repository as repository_module
from app.infrastructure.repositories.system_repository import SystemRepository


@contextmanager
def _session_context(factory: sessionmaker[Session]) -> Iterator[Session]:
    """提供测试用 ORM 会话，并模拟项目真实提交/回滚行为。"""

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _build_session_factory() -> sessionmaker[Session]:
    """创建只包含角色菜单关系表的内存数据库。"""

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    BaseOrmModel.metadata.create_all(engine, tables=[SystemRoleMenuEntity.__table__])
    Index(
        "uk_system_role_menus_role_menu",
        SystemRoleMenuEntity.role_id,
        SystemRoleMenuEntity.menu_id,
        unique=True,
    ).create(bind=engine, checkfirst=True)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def test_replace_role_menus_allows_resaving_existing_menu_ids(monkeypatch: pytest.MonkeyPatch):
    """覆盖保存菜单权限时，同一角色原有菜单不应触发唯一键冲突。"""

    factory = _build_session_factory()

    @contextmanager
    def fake_orm_session_context() -> Iterator[Session]:
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(repository_module, "orm_session_context", fake_orm_session_context)

    with _session_context(factory) as session:
        session.add(SystemRoleMenuEntity(
            role_menu_id=1,
            role_id=1001,
            menu_id=2001,
            created_at=datetime(2026, 7, 1),
        ))

    repository = SystemRepository()
    repository.replace_role_menus(role_id=1001, menu_ids=[2001, 2002], relation_ids=[2, 3])

    with _session_context(factory) as session:
        rows = session.scalars(
            select(SystemRoleMenuEntity).where(SystemRoleMenuEntity.role_id == 1001)
        ).all()

    assert [(row.role_menu_id, row.menu_id) for row in rows] == [(2, 2001), (3, 2002)]


def test_replace_role_menus_deduplicates_incoming_menu_ids(monkeypatch: pytest.MonkeyPatch):
    """前端重复提交同一个菜单 ID 时，仓储层应只保留一条关系。"""

    factory = _build_session_factory()

    @contextmanager
    def fake_orm_session_context() -> Iterator[Session]:
        with _session_context(factory) as session:
            yield session

    monkeypatch.setattr(repository_module, "orm_session_context", fake_orm_session_context)

    repository = SystemRepository()
    repository.replace_role_menus(role_id=1001, menu_ids=[2001, 2001, 2002], relation_ids=[2, 3, 4])

    with _session_context(factory) as session:
        rows = session.scalars(
            select(SystemRoleMenuEntity).where(SystemRoleMenuEntity.role_id == 1001)
        ).all()

    assert [(row.role_menu_id, row.menu_id) for row in rows] == [(2, 2001), (4, 2002)]
