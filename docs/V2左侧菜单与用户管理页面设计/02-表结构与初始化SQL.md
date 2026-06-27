# 表结构与初始化 SQL 设计

## 1. 主键和 ID 规则

从本需求开始，新增表主键统一使用雪花算法 ID：

- 数据库字段类型统一使用 `BIGINT`。
- 角色、菜单、角色菜单关系三张新增表都使用雪花 ID 作为主键。
- 新增关系字段也使用雪花 ID，例如 `system_role_menus.role_id` 关联 `system_roles.role_id`。
- 旧表和历史数据本期不强制迁移，例如 `system_users.user_id`、`system_users.role` 保持现状。
- API 返回给前端时，雪花 ID 统一转成字符串，避免 JavaScript `number` 精度丢失。
- 如果已经执行过早期 `VARCHAR` 主键草稿，`CREATE TABLE IF NOT EXISTS` 不会自动修正表结构；需要先确认三张新表没有正式业务数据，再重建 `system_roles`、`system_menus`、`system_role_menus`。

完整可执行脚本见：

```text
docs/V2左侧菜单与用户管理页面设计/06-mysql迁移_V2左侧菜单与用户管理.sql
```

## 2. 角色表：system_roles

```sql
CREATE TABLE IF NOT EXISTS system_roles (
  role_id BIGINT NOT NULL COMMENT '角色唯一编号，雪花算法生成',
  role_code VARCHAR(64) NOT NULL COMMENT '角色编码，例如 admin、user',
  role_name VARCHAR(128) NOT NULL COMMENT '角色名称',
  status VARCHAR(32) NOT NULL DEFAULT 'active' COMMENT '角色状态：active启用、disabled停用',
  sort_order INT NOT NULL DEFAULT 0 COMMENT '排序号，越小越靠前',
  built_in TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否内置角色：1内置、0自定义',
  description VARCHAR(255) NULL COMMENT '角色说明',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  PRIMARY KEY (role_id),
  UNIQUE KEY uk_system_roles_code (role_code),
  KEY idx_system_roles_status_sort (status, sort_order)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统角色表';
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `role_id` | 角色主键，雪花算法生成，数据库类型为 `BIGINT` |
| `role_code` | 角色编码，和 `system_users.role` 对应 |
| `role_name` | 页面展示名称 |
| `status` | `active` 可用，`disabled` 停用 |
| `built_in` | 内置角色不允许删除，避免系统权限被破坏 |
| `sort_order` | 角色下拉和角色列表排序 |

## 3. 菜单表：system_menus

```sql
CREATE TABLE IF NOT EXISTS system_menus (
  menu_id BIGINT NOT NULL COMMENT '菜单唯一编号，雪花算法生成',
  parent_menu_id BIGINT NULL COMMENT '父级菜单编号，一级菜单为空',
  menu_code VARCHAR(128) NOT NULL COMMENT '菜单编码，全局唯一',
  menu_name VARCHAR(128) NOT NULL COMMENT '菜单展示名称',
  menu_type VARCHAR(32) NOT NULL COMMENT '菜单类型：directory目录、page页面',
  page_key VARCHAR(128) NULL COMMENT '前端页面键，当前用于 activePage 切换',
  route_path VARCHAR(255) NULL COMMENT '路由路径，预留给后续 Vue Router 使用',
  component_key VARCHAR(128) NULL COMMENT '前端组件键，预留给动态组件映射使用',
  icon VARCHAR(64) NULL COMMENT '菜单图标名称，对应前端 lucide 图标映射',
  permission_code VARCHAR(128) NULL COMMENT '权限编码，预留给按钮或接口权限使用',
  sort_order INT NOT NULL DEFAULT 0 COMMENT '同级排序号，越小越靠前',
  visible TINYINT(1) NOT NULL DEFAULT 1 COMMENT '是否在左侧菜单展示：1展示、0隐藏',
  status VARCHAR(32) NOT NULL DEFAULT 'active' COMMENT '菜单状态：active启用、disabled停用',
  metadata_json JSON NULL COMMENT '扩展配置，例如徽标、分组说明、外链配置',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间',
  PRIMARY KEY (menu_id),
  UNIQUE KEY uk_system_menus_code (menu_code),
  KEY idx_system_menus_parent_sort (parent_menu_id, sort_order),
  KEY idx_system_menus_status_visible (status, visible)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统菜单表';
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `menu_id` | 菜单主键，雪花算法生成，数据库类型为 `BIGINT` |
| `parent_menu_id` | 父级菜单 ID，关联本表 `menu_id`，一级菜单为空 |
| `menu_code` | 菜单编码，全局唯一，用于后台识别 |
| `page_key` | 前端页面键，当前用于 `activePage` 切换 |
| `route_path` | 路由路径，预留给后续 Vue Router 使用 |
| `visible` | 是否展示到左侧菜单 |
| `status` | 是否启用 |

## 4. 角色菜单关系表：system_role_menus

```sql
CREATE TABLE IF NOT EXISTS system_role_menus (
  role_menu_id BIGINT NOT NULL COMMENT '角色菜单关系编号，雪花算法生成',
  role_id BIGINT NOT NULL COMMENT '角色编号，关联 system_roles.role_id',
  menu_id BIGINT NOT NULL COMMENT '菜单编号，关联 system_menus.menu_id',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  PRIMARY KEY (role_menu_id),
  UNIQUE KEY uk_system_role_menus_role_menu (role_id, menu_id),
  KEY idx_system_role_menus_menu (menu_id),
  CONSTRAINT fk_system_role_menus_role
    FOREIGN KEY (role_id) REFERENCES system_roles (role_id)
    ON DELETE CASCADE,
  CONSTRAINT fk_system_role_menus_menu
    FOREIGN KEY (menu_id) REFERENCES system_menus (menu_id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统角色菜单关系表';
```

说明：

- `role_menu_id` 是关系表自己的雪花 ID 主键。
- `role_id` 关联 `system_roles.role_id`，不再用 `role_code` 做关系字段。
- `menu_id` 关联 `system_menus.menu_id`。
- 一个角色拥有多个菜单。
- 一个菜单可以分配给多个角色。

## 5. system_users 调整建议

当前 `system_users` 作为用户主表继续复用，不新增用户主表。

建议只补充约束认知，不强制改表：

| 字段 | 当前用途 | 一期处理 |
| --- | --- | --- |
| `role` | 用户角色 | 取值来自 `system_roles.role_code` |
| `status` | 用户状态 | 只允许 `active`、`disabled` |
| `password_hash` | 密码哈希 | 接口永不返回 |

本次不建议给 `system_users.role` 直接加外键，原因是旧数据可能存在非标准角色值。业务代码在新增/修改用户时校验角色必须来自启用的 `system_roles`。

## 6. 初始化角色数据

固定初始化数据也使用雪花 ID，便于重复执行脚本和后续文档引用。

```sql
INSERT INTO system_roles (
  role_id, role_code, role_name, status, sort_order, built_in, description, created_at, updated_at
) VALUES
(2033000000000000001, 'admin', '管理员', 'active', 10, 1, '系统内置管理员角色，默认拥有全部菜单和系统管理权限', NOW(), NOW()),
(2033000000000000002, 'user', '普通用户', 'active', 20, 1, '系统内置普通用户角色，默认只拥有业务页面访问权限', NOW(), NOW())
ON DUPLICATE KEY UPDATE
  role_name = VALUES(role_name),
  status = VALUES(status),
  sort_order = VALUES(sort_order),
  built_in = VALUES(built_in),
  description = VALUES(description),
  updated_at = NOW();
```

## 7. 初始化菜单数据

```sql
INSERT INTO system_menus (
  menu_id, parent_menu_id, menu_code, menu_name, menu_type, page_key, route_path,
  component_key, icon, permission_code, sort_order, visible, status, metadata_json, created_at, updated_at
) VALUES
(2033000000000010001, NULL, 'home', '首页', 'page', 'home', '/home', 'HomePage', 'LayoutDashboard', 'dashboard:view', 10, 1, 'active', JSON_OBJECT('sub_label', '系统驾驶舱'), NOW(), NOW()),
(2033000000000010002, NULL, 'chat', '智能客服', 'page', 'chat', '/chat', 'ChatPage', 'Bot', 'chat:view', 20, 1, 'active', JSON_OBJECT('sub_label', 'RAG问答'), NOW(), NOW()),
(2033000000000010003, NULL, 'salesTraining', '销售陪练', 'page', 'salesTraining', '/sales-training', 'SalesTrainingPage', 'BrainCircuit', 'training:view', 30, 1, 'active', JSON_OBJECT('sub_label', 'AI客户训练'), NOW(), NOW()),
(2033000000000010004, NULL, 'exam', '问答考试', 'page', 'exam', '/exam', 'ExamPage', 'ClipboardCheck', 'exam:view', 40, 1, 'active', JSON_OBJECT('sub_label', '知识测评'), NOW(), NOW()),
(2033000000000010005, NULL, 'system', '系统管理', 'directory', NULL, NULL, NULL, 'Settings', 'system:view', 90, 1, 'active', JSON_OBJECT('sub_label', '后台配置'), NOW(), NOW()),
(2033000000000010006, 2033000000000010005, 'userManagement', '用户管理', 'page', 'userManagement', '/system/users', 'UserManagementPage', 'Users', 'system:user:manage', 10, 1, 'active', JSON_OBJECT('sub_label', '账号权限'), NOW(), NOW()),
(2033000000000010007, 2033000000000010005, 'roleManagement', '角色管理', 'page', 'roleManagement', '/system/roles', 'RoleManagementPage', 'ShieldCheck', 'system:role:manage', 20, 1, 'active', JSON_OBJECT('sub_label', '角色菜单'), NOW(), NOW())
ON DUPLICATE KEY UPDATE
  parent_menu_id = VALUES(parent_menu_id),
  menu_name = VALUES(menu_name),
  menu_type = VALUES(menu_type),
  page_key = VALUES(page_key),
  route_path = VALUES(route_path),
  component_key = VALUES(component_key),
  icon = VALUES(icon),
  permission_code = VALUES(permission_code),
  sort_order = VALUES(sort_order),
  visible = VALUES(visible),
  status = VALUES(status),
  metadata_json = VALUES(metadata_json),
  updated_at = NOW();
```

## 8. 初始化角色菜单关系

管理员拥有全部菜单，普通用户只拥有业务菜单。关系表也使用固定雪花 ID 初始化。

```sql
INSERT INTO system_role_menus (role_menu_id, role_id, menu_id, created_at) VALUES
(2033000000000020001, 2033000000000000001, 2033000000000010001, NOW()),
(2033000000000020002, 2033000000000000001, 2033000000000010002, NOW()),
(2033000000000020003, 2033000000000000001, 2033000000000010003, NOW()),
(2033000000000020004, 2033000000000000001, 2033000000000010004, NOW()),
(2033000000000020005, 2033000000000000001, 2033000000000010005, NOW()),
(2033000000000020006, 2033000000000000001, 2033000000000010006, NOW()),
(2033000000000020007, 2033000000000000001, 2033000000000010007, NOW()),
(2033000000000021001, 2033000000000000002, 2033000000000010001, NOW()),
(2033000000000021002, 2033000000000000002, 2033000000000010002, NOW()),
(2033000000000021003, 2033000000000000002, 2033000000000010003, NOW()),
(2033000000000021004, 2033000000000000002, 2033000000000010004, NOW())
ON DUPLICATE KEY UPDATE
  role_id = VALUES(role_id),
  menu_id = VALUES(menu_id);
```

## 9. ORM 实体设计

`domain/entities.py` 新增。数据库实体使用 `int` 承接 `BIGINT`，响应模型再转字符串输出。

```python
class SystemRoleEntity(BaseOrmModel, DictMixin):
    """system_roles 表实体，记录系统角色。"""

    __tablename__ = "system_roles"

    role_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, comment="角色唯一编号，雪花算法生成")
    role_code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, comment="角色编码")
    role_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="角色名称")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", comment="角色状态")
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="排序号")
    built_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="是否内置角色")
    description: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="角色说明")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")
```

```python
class SystemMenuEntity(BaseOrmModel, DictMixin):
    """system_menus 表实体，记录左侧菜单和页面入口。"""

    __tablename__ = "system_menus"

    menu_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, comment="菜单唯一编号，雪花算法生成")
    parent_menu_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, comment="父级菜单编号")
    menu_code: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, comment="菜单编码")
    menu_name: Mapped[str] = mapped_column(String(128), nullable=False, comment="菜单展示名称")
    menu_type: Mapped[str] = mapped_column(String(32), nullable=False, comment="菜单类型")
    page_key: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="前端页面键")
    route_path: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="路由路径")
    component_key: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="组件键")
    icon: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="菜单图标")
    permission_code: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="权限编码")
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="排序号")
    visible: Mapped[int] = mapped_column(Integer, nullable=False, default=1, comment="是否展示")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", comment="菜单状态")
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True, comment="扩展配置 JSON")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
    updated_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="更新时间")
```

```python
class SystemRoleMenuEntity(BaseOrmModel, DictMixin):
    """system_role_menus 表实体，记录角色可见菜单。"""

    __tablename__ = "system_role_menus"

    role_menu_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, comment="角色菜单关系编号，雪花算法生成")
    role_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="角色编号")
    menu_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="菜单编号")
    created_at: Mapped[datetime | str] = mapped_column(DateTime, nullable=False, comment="创建时间")
```

## 10. 数据迁移注意事项

1. 菜单初始化 SQL 使用 `ON DUPLICATE KEY UPDATE`，可重复执行。
2. 角色初始化 SQL 使用 `ON DUPLICATE KEY UPDATE`，可重复执行。
3. 不直接删除已有数据，避免影响后续手动配置。
4. 如果要隐藏某个菜单，改 `visible = 0`。
5. 如果要停用某个菜单，改 `status = disabled`。
6. 如果要停用某个角色，改 `system_roles.status = disabled`。
7. `system_users` 不做破坏性迁移。
8. 后端创建新角色、新菜单、新关系时必须使用雪花 ID；当前代码已复用 `infrastructure.id_generator.new_id()`，不能再使用字符串拼接 ID。
