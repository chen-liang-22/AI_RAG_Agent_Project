-- V2 左侧菜单与用户管理迁移脚本。
-- 适用范围：
-- 1. 新增系统角色表 system_roles。
-- 2. 新增系统菜单表 system_menus。
-- 3. 新增角色菜单关系表 system_role_menus。
-- 4. 初始化当前 V2 左侧菜单、用户管理、角色管理、菜单管理菜单和 admin/user 角色菜单权限。
-- 说明：
-- - 本脚本可重复执行。
-- - 如果已执行过早期 VARCHAR 主键草稿，本脚本不会自动 ALTER 已存在表；需先确认 system_roles、system_menus、
--   system_role_menus 三张新表无正式业务数据后再重建。
-- - 用户管理继续复用已有 system_users 表。
-- - 本次新增表主键统一使用雪花算法 ID，数据库类型使用 BIGINT。
-- - 接口向前端返回雪花 ID 时统一转字符串，避免 JavaScript number 精度丢失。
-- - system_users.role 通过业务代码引用 system_roles.role_code，本脚本不对旧用户表强加外键，避免历史数据阻塞迁移。

CREATE DATABASE IF NOT EXISTS ai_rag_agent
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE ai_rag_agent;

SET NAMES utf8mb4;

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

INSERT INTO system_roles (
  role_id,
  role_code,
  role_name,
  status,
  sort_order,
  built_in,
  description,
  created_at,
  updated_at
) VALUES
(
  2033000000000000001,
  'admin',
  '管理员',
  'active',
  10,
  1,
  '系统内置管理员角色，默认拥有全部菜单和系统管理权限',
  NOW(),
  NOW()
),
(
  2033000000000000002,
  'user',
  '普通用户',
  'active',
  20,
  1,
  '系统内置普通用户角色，默认只拥有业务页面访问权限',
  NOW(),
  NOW()
)
ON DUPLICATE KEY UPDATE
  role_name = VALUES(role_name),
  status = VALUES(status),
  sort_order = VALUES(sort_order),
  built_in = VALUES(built_in),
  description = VALUES(description),
  updated_at = NOW();

INSERT INTO system_menus (
  menu_id,
  parent_menu_id,
  menu_code,
  menu_name,
  menu_type,
  page_key,
  route_path,
  component_key,
  icon,
  permission_code,
  sort_order,
  visible,
  status,
  metadata_json,
  created_at,
  updated_at
) VALUES
(
  2033000000000010001,
  NULL,
  'home',
  '首页',
  'page',
  'home',
  '/home',
  'HomePage',
  'LayoutDashboard',
  'dashboard:view',
  10,
  1,
  'active',
  JSON_OBJECT('sub_label', '系统驾驶舱'),
  NOW(),
  NOW()
),
(
  2033000000000010002,
  NULL,
  'chat',
  '智能客服',
  'page',
  'chat',
  '/chat',
  'ChatPage',
  'Bot',
  'chat:view',
  20,
  1,
  'active',
  JSON_OBJECT('sub_label', 'RAG问答'),
  NOW(),
  NOW()
),
(
  2033000000000010003,
  NULL,
  'salesTraining',
  '销售陪练',
  'page',
  'salesTraining',
  '/sales-training',
  'SalesTrainingPage',
  'BrainCircuit',
  'training:view',
  30,
  1,
  'active',
  JSON_OBJECT('sub_label', 'AI客户训练'),
  NOW(),
  NOW()
),
(
  2033000000000010004,
  NULL,
  'exam',
  '问答考试',
  'page',
  'exam',
  '/exam',
  'ExamPage',
  'ClipboardCheck',
  'exam:view',
  40,
  1,
  'active',
  JSON_OBJECT('sub_label', '知识测评'),
  NOW(),
  NOW()
),
(
  2033000000000010005,
  NULL,
  'system',
  '系统管理',
  'directory',
  NULL,
  NULL,
  NULL,
  'Settings',
  'system:view',
  90,
  1,
  'active',
  JSON_OBJECT('sub_label', '后台配置'),
  NOW(),
  NOW()
),
(
  2033000000000010006,
  2033000000000010005,
  'userManagement',
  '用户管理',
  'page',
  'userManagement',
  '/system/users',
  'UserManagementPage',
  'Users',
  'system:user:manage',
  10,
  1,
  'active',
  JSON_OBJECT('sub_label', '账号权限'),
  NOW(),
  NOW()
),
(
  2033000000000010007,
  2033000000000010005,
  'roleManagement',
  '角色管理',
  'page',
  'roleManagement',
  '/system/roles',
  'RoleManagementPage',
  'ShieldCheck',
  'system:role:manage',
  20,
  1,
  'active',
  JSON_OBJECT('sub_label', '角色菜单'),
  NOW(),
  NOW()
),
(
  2033000000000010008,
  2033000000000010005,
  'menuManagement',
  '菜单管理',
  'page',
  'menuManagement',
  '/system/menus',
  'MenuManagementPage',
  'Menu',
  'system:menu:manage',
  30,
  1,
  'active',
  JSON_OBJECT('sub_label', '菜单配置'),
  NOW(),
  NOW()
)
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

INSERT INTO system_role_menus (role_menu_id, role_id, menu_id, created_at) VALUES
(2033000000000020001, 2033000000000000001, 2033000000000010001, NOW()),
(2033000000000020002, 2033000000000000001, 2033000000000010002, NOW()),
(2033000000000020003, 2033000000000000001, 2033000000000010003, NOW()),
(2033000000000020004, 2033000000000000001, 2033000000000010004, NOW()),
(2033000000000020005, 2033000000000000001, 2033000000000010005, NOW()),
(2033000000000020006, 2033000000000000001, 2033000000000010006, NOW()),
(2033000000000020007, 2033000000000000001, 2033000000000010007, NOW()),
(2033000000000020008, 2033000000000000001, 2033000000000010008, NOW()),
(2033000000000021001, 2033000000000000002, 2033000000000010001, NOW()),
(2033000000000021002, 2033000000000000002, 2033000000000010002, NOW()),
(2033000000000021003, 2033000000000000002, 2033000000000010003, NOW()),
(2033000000000021004, 2033000000000000002, 2033000000000010004, NOW())
ON DUPLICATE KEY UPDATE
  role_id = VALUES(role_id),
  menu_id = VALUES(menu_id);
