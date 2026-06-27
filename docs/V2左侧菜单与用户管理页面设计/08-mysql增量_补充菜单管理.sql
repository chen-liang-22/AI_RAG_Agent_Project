-- V2 系统管理增量脚本：补充“菜单管理”菜单。
-- 适用场景：已经执行过 06-mysql迁移_V2左侧菜单与用户管理.sql，但左侧系统管理下缺少“菜单管理”。
-- 说明：
-- - 本脚本可重复执行。
-- - 菜单主键和角色菜单关系主键继续使用固定雪花 ID，避免重复执行产生多条数据。
-- - 只给 admin 角色授权菜单管理；普通 user 角色不默认开放系统配置入口。

USE ai_rag_agent;

SET NAMES utf8mb4;

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
) VALUES (
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
(2033000000000020008, 2033000000000000001, 2033000000000010008, NOW())
ON DUPLICATE KEY UPDATE
  role_id = VALUES(role_id),
  menu_id = VALUES(menu_id);
