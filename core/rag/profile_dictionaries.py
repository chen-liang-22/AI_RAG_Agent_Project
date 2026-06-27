"""学员和客户画像默认字典。

这些字典项的业务口径来自 docs/当前学员客户画像字典.md。
结构约定：
- 一级字典项表示字段，例如 position_role、customer_category；
- 二级字典项表示字段可选值，例如 overseas_bd、junior；
- 年龄、其他等输入型字段只有一级项，通过 metadata.input_type 标记为 text。
"""

from typing import Any


SOURCE_DOCUMENT = "docs/当前学员客户画像字典.md"
DictionaryItem = tuple[str, str, str | None, int, str, dict[str, Any]]


def field_item(
        code: str,
        name: str,
        sort_order: int,
        *,
        input_type: str = "select",
        multiple: bool = False,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
) -> DictionaryItem:
    """创建画像字段字典项。"""

    has_fixed_options = input_type in {"select", "multi_select"}
    item_metadata = {
        "node_type": "field",
        "field_code": code,
        "input_type": input_type,
        "multiple": multiple,
        "has_fixed_options": has_fixed_options,
        "source_document": SOURCE_DOCUMENT,
    }
    if metadata:
        item_metadata.update(metadata)
    return (
        code,
        name,
        None,
        sort_order,
        description or f"{name}字段",
        item_metadata,
    )


def option_item(
        parent_code: str,
        code: str,
        name: str,
        sort_order: int,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
) -> DictionaryItem:
    """创建画像字段的候选选项字典项。"""

    item_metadata = {
        "node_type": "option",
        "value": code,
        "source_document": SOURCE_DOCUMENT,
    }
    if metadata:
        item_metadata.update(metadata)
    return (
        code,
        name,
        parent_code,
        sort_order,
        description or f"{name}选项",
        item_metadata,
    )


def option_items(parent_code: str, options: list[tuple[str, str]]) -> list[DictionaryItem]:
    """按传入顺序批量创建候选选项。"""

    return [option_item(parent_code, code, name, index) for index, (name, code) in enumerate(options, start=1)]


PROFILE_DICTIONARY_ITEMS = [
    {
        "dictionary_code": "student_portrait",
        "dictionary_name": "学员画像字典",
        "items": [
            field_item("position_role", "职位角色", 1),
            *option_items(
                "position_role",
                [
                    ("外综服客户经理", "wzf_customer_manager"),
                    ("超级客服", "wm_ai_service"),
                    ("海外BD", "overseas_bd"),
                ],
            ),
            field_item("experience_level", "经验等级", 2),
            *option_items(
                "experience_level",
                [
                    ("新手", "beginner"),
                    ("初级", "junior"),
                    ("中级", "intermediate"),
                    ("高级", "senior"),
                ],
            ),
            field_item("task_goal", "任务目标", 3),
            *option_items(
                "task_goal",
                [
                    ("初级", "goal_junior"),
                    ("中级", "goal_intermediate"),
                    ("高级", "goal_senior"),
                ],
            ),
            field_item("weakness_tag", "短板标签", 4, input_type="multi_select", multiple=True),
            *option_items(
                "weakness_tag",
                [
                    ("产品介绍", "product_intro"),
                    ("需求挖掘", "demand_mining"),
                    ("价格谈判", "price_negotiation"),
                    ("异议处理", "objection_handling"),
                    ("逼单技巧", "closing_skills"),
                    ("售后处理", "after_sales"),
                    ("陌拜", "sample_expansion"),
                    ("行业趋势分析", "industry_trend"),
                ],
            ),
            field_item("student_portrait_other", "其他", 5, input_type="text"),
        ],
    },
    {
        "dictionary_code": "wzf_customer_manager",
        "dictionary_name": "外综服客户经理画像字典",
        "items": [
            field_item("customer_category", "客户分类", 1),
            *option_items("customer_category", [("B端", "b_end")]),
            field_item("trade_experience", "外贸经验", 2),
            *option_items(
                "trade_experience",
                [
                    ("没有经验", "no_experience"),
                    ("有经验", "experienced"),
                    ("有沉淀", "well_established"),
                ],
            ),
            field_item("cooperation_stage", "合作阶段", 3),
            *option_items(
                "cooperation_stage",
                [
                    ("新用户", "new_user"),
                    ("老客户复购", "repeat_customer"),
                    ("流失召回", "win_back"),
                    ("战略合作", "strategic_partner"),
                ],
            ),
            field_item("industry", "行业", 4),
            *option_items(
                "industry",
                [
                    ("工程机械", "construction_machinery"),
                    ("农业机械", "agricultural_machinery"),
                    ("电动摩托", "electric_motorcycle"),
                    ("五金", "hardware"),
                    ("激光设备", "laser_equipment"),
                ],
            ),
            field_item("purchase_stage", "当前阶段/采购阶段", 5),
            *option_items(
                "purchase_stage",
                [
                    ("初步了解", "initial_inquiry"),
                    ("方案对比", "solution_comparison"),
                    ("洽判中", "negotiating"),
                    ("即将下单", "about_to_order"),
                    ("售后复购", "after_sales_repurchase"),
                ],
            ),
            field_item("core_focus", "核心关注点", 6),
            *option_items(
                "core_focus",
                [
                    ("质量", "quality"),
                    ("交期", "delivery_time"),
                    ("售后服务", "after_sales_service"),
                    ("品牌", "brand"),
                    ("定制化", "customization"),
                    ("合规认证", "compliance_certification"),
                    ("行业趋势", "industry_trends"),
                    ("竞品对比", "competitor_comparison"),
                    ("同行案例", "peer_cases"),
                    ("软件安全性", "software_security"),
                    ("产品卖点", "product_selling_points"),
                    ("优势", "discount"),
                ],
            ),
            field_item("secondary_focus", "次要关注点", 7),
            *option_items(
                "secondary_focus",
                [
                    ("质量", "s_quality"),
                    ("交期", "s_delivery_time"),
                    ("售后服务", "s_after_sales_service"),
                    ("品牌", "s_brand"),
                    ("定制化", "s_customization"),
                    ("合规认证", "s_compliance_certification"),
                    ("行业趋势", "s_industry_trends"),
                    ("竞品对比", "s_competitor_comparison"),
                    ("同行案例", "s_peer_cases"),
                    ("软件安全性", "s_software_security"),
                    ("产品卖点", "s_product_selling_points"),
                    ("优势", "s_discount"),
                ],
            ),
            field_item("price_sensitivity", "价格敏感度", 8),
            *option_items(
                "price_sensitivity",
                [
                    ("极低", "very_low"),
                    ("低", "low"),
                    ("中", "medium"),
                    ("高", "high"),
                    ("极高", "very_high"),
                ],
            ),
            field_item("applicable_product", "适用产品", 9),
            *option_items(
                "applicable_product",
                [
                    ("AI盈出海", "ai_overseas"),
                    ("外综服产品包", "wzf_customer_manager_product_service_package"),
                    ("共盈出海", "shared_overseas"),
                    ("商学院", "business_school"),
                    ("外综服", "foreign_service"),
                    ("易运盈明道堂", "alternative_mingdao"),
                ],
            ),
            field_item("training_scene", "训练场景", 10),
            *option_items(
                "training_scene",
                [
                    ("电话销售", "phone_sales"),
                    ("线下销售", "offline_sales"),
                    ("会销", "conference_sales"),
                ],
            ),
            field_item("personality", "性格特征", 11),
            *option_items(
                "personality",
                [
                    ("直接果断", "direct_decisive"),
                    ("谨慎保守", "cautious_conservative"),
                    ("挑剔挑刺", "picky_critical"),
                    ("强势主导", "dominant_controlling"),
                ],
            ),
            field_item("wzf_customer_manager_gender", "性别", 12),
            *option_items(
                "wzf_customer_manager_gender",
                [
                    ("男", "wzf_customer_manager_gender_male"),
                    ("女", "wzf_customer_manager_gender_female"),
                ],
            ),
            field_item("wzf_customer_manager_age", "年龄", 13, input_type="text"),
            field_item("wzf_customer_manager_other", "其他", 14, input_type="text"),
        ],
    },
    {
        "dictionary_code": "wm_ai_service",
        "dictionary_name": "超级客服画像字典",
        "items": [
            field_item("super_customer_type", "客户分类", 1),
            *option_items(
                "super_customer_type",
                [
                    ("C端", "super_customer_type_c_end"),
                    ("B端", "super_customer_type_b_end"),
                    ("大业务", "super_customer_type_large_business"),
                ],
            ),
            field_item("super_customer_stage", "客户阶段", 2),
            *option_items(
                "super_customer_stage",
                [
                    ("A：成交客户", "super_customer_stage_a_closed_customer"),
                    ("B1：报价后客户冷淡", "super_customer_stage_b1_cold_after_quote"),
                    ("B2：报价后深互动", "super_customer_stage_b2_deep_interaction_after_quote"),
                    ("B3：PI阶段(高意向)", "super_customer_stage_b3_pi_high_intention"),
                    ("C：产品相关且有回复客户", "super_customer_stage_c_product_replied"),
                    ("D：产品相关不回复客户", "super_customer_stage_d_product_unreplied"),
                    ("E：非产品咨询客户", "super_customer_stage_e_non_product_consulting"),
                    ("F：暂空客户", "super_customer_stage_f_empty"),
                ],
            ),
            field_item("super_customer_location", "客户所在地", 3),
            *option_items(
                "super_customer_location",
                [
                    ("乌兹别克斯坦", "super_location_uzbekistan"),
                    ("阿联酋", "super_location_uae"),
                    ("法国", "super_location_france"),
                    ("西班牙", "super_location_spain"),
                    ("意大利", "super_location_italy"),
                    ("印尼", "super_location_indonesia"),
                    ("迪拜", "super_location_dubai"),
                    ("哈萨克斯坦", "super_location_kazakhstan"),
                    ("土耳其", "super_location_turkey"),
                    ("伊拉克", "super_location_iraq"),
                    ("肯尼亚", "super_location_kenya"),
                    ("智利", "super_location_chile"),
                    ("匈牙利", "super_location_hungary"),
                    ("希腊", "super_location_greece"),
                    ("罗马尼亚", "super_location_romania"),
                    ("越南", "super_location_vietnam"),
                ],
            ),
            field_item("super_customer_source", "客户来源", 4),
            *option_items(
                "super_customer_source",
                [
                    ("Facebook", "super_source_facebook"),
                    ("Google-seo", "super_source_google_seo"),
                    ("阿里国际站", "super_source_alibaba_international"),
                ],
            ),
            field_item("super_customer_intention", "客户意向", 5),
            *option_items(
                "super_customer_intention",
                [
                    ("高意向", "super_intention_high"),
                    ("有兴趣", "super_intention_interested"),
                    ("一般了解", "super_intention_general_understanding"),
                    ("暂不考虑", "super_intention_not_considering"),
                ],
            ),
            field_item("super_cooperation_phase", "合作阶段", 6),
            *option_items(
                "super_cooperation_phase",
                [
                    ("新用户", "super_phase_new_user"),
                    ("老客户", "super_phase_old_customer"),
                    ("公海客户", "super_phase_public_pool_customer"),
                ],
            ),
            field_item("super_industry", "行业", 7),
            *option_items(
                "super_industry",
                [
                    ("工程机械", "s_construction_machinery"),
                    ("农业机械", "s_agricultural_machinery"),
                    ("电动搬运", "super_industry_electric_handling"),
                    ("木工", "woodworking"),
                    ("五金", "s_hardware"),
                    ("激光设备", "s_laser_equipment"),
                    ("智能一体机", "super_industry_intelligent_all_in_one"),
                ],
            ),
            field_item("super_current_stage", "当前阶段", 8),
            *option_items(
                "super_current_stage",
                [
                    ("首次跟进", "super_current_first_follow_up"),
                    ("再次跟进", "super_current_second_follow_up"),
                    ("跟进不回", "super_current_follow_up_no_response"),
                    ("跟进回复", "super_current_follow_up_replied"),
                ],
            ),
            field_item("super_core_concern", "核心关注点", 9),
            *option_items(
                "super_core_concern",
                [
                    ("质量", "super_core_quality"),
                    ("交期", "super_core_delivery_time"),
                    ("售后质保", "super_core_after_sales_warranty"),
                    ("品牌", "super_core_brand"),
                    ("定制化", "super_core_customization"),
                    ("资质认证", "super_core_qualification_certification"),
                    ("信任获取", "super_core_trust_acquisition"),
                    ("付款方式", "super_core_payment_method"),
                    ("预付款占比", "super_core_advance_payment_ratio"),
                    ("产品适配度", "super_core_product_fit"),
                    ("报价", "super_core_quotation"),
                    ("逼单促销", "super_core_closing_promotion"),
                    ("发货", "super_core_delivery"),
                    ("物流：关税/增值税", "super_core_logistics_tariff_vat"),
                    ("经销商政策", "super_core_dealer_policy"),
                ],
            ),
            field_item("super_secondary_concern", "次要关注点", 10),
            *option_items(
                "super_secondary_concern",
                [
                    ("质量", "super_secondary_quality"),
                    ("交期", "super_secondary_delivery_time"),
                    ("售后质保", "super_secondary_after_sales_warranty"),
                    ("品牌", "super_secondary_brand"),
                    ("定制化", "super_secondary_customization"),
                    ("资质认证", "super_secondary_qualification_certification"),
                    ("信任获取", "super_secondary_trust_acquisition"),
                    ("付款方式", "super_secondary_payment_method"),
                    ("预付款占比", "super_secondary_advance_payment_ratio"),
                    ("产品适配度", "super_secondary_product_fit"),
                    ("报价", "super_secondary_quotation"),
                    ("逼单促销", "super_secondary_closing_promotion"),
                    ("发货", "super_secondary_delivery"),
                    ("物流：关税/增值税", "super_secondary_logistics_tariff_vat"),
                    ("经销商政策", "super_secondary_dealer_policy"),
                ],
            ),
            field_item("super_price_sensitivity", "价格敏感度", 11),
            *option_items(
                "super_price_sensitivity",
                [
                    ("极低", "s_very_low"),
                    ("低", "s_low"),
                    ("中", "s_medium"),
                    ("高", "s_high"),
                    ("极高", "s_very_high"),
                ],
            ),
            field_item("super_decision_maker", "是否决策人", 12),
            *option_items(
                "super_decision_maker",
                [
                    ("是", "super_decision_maker_yes"),
                    ("否", "super_decision_maker_no"),
                ],
            ),
            field_item("super_applicable_product", "适用产品", 13),
            *option_items(
                "super_applicable_product",
                [
                    ("拖拉机", "super_product_tractor"),
                    ("高尔夫球车", "super_product_golf_cart"),
                    ("堆高车", "super_product_stacker"),
                    ("挖掘机", "super_product_excavator"),
                    ("滑移", "super_product_skid_steer"),
                    ("液压水井钻机", "super_product_hydraulic_water_well_drilling_rig"),
                    ("激光打标机", "super_product_laser_marking_machine"),
                    ("焊接清洗机", "super_product_welding_cleaning_machine"),
                    ("割草机", "super_product_lawn_mower"),
                    ("身体分析仪", "super_product_body_analyzer"),
                    ("自卸车", "super_product_dump_truck"),
                    ("随心屏", "super_product_portable_screen"),
                ],
            ),
            field_item("super_personality", "性格特征", 14),
            *option_items(
                "super_personality",
                [
                    ("直接果断", "s_direct_decisive"),
                    ("谨慎犹豫", "super_personality_cautious_hesitant"),
                    ("理性数据控", "data_driven"),
                    ("挑剔细节", "super_personality_detail_oriented"),
                    ("强势主导", "super_personality_dominant"),
                ],
            ),
            field_item("super_gender", "性别", 15),
            *option_items(
                "super_gender",
                [
                    ("男", "super_gender_male"),
                    ("女", "super_gender_female"),
                ],
            ),
            field_item("super_age", "年龄", 16, input_type="text"),
            field_item("super_other", "其他", 17, input_type="text"),
        ],
    },
    {
        "dictionary_code": "overseas_bd",
        "dictionary_name": "海外BD画像字典",
        "items": [
            field_item("overseas_bd_customer_type", "客户类型", 1),
            *option_items(
                "overseas_bd_customer_type",
                [
                    ("C端", "overseas_bd_customer_type_c_end"),
                    ("B端", "overseas_bd_customer_type_b_end"),
                    ("G端", "overseas_bd_customer_type_g_end"),
                ],
            ),
            field_item("overseas_bd_customer_category", "客户分类", 2),
            *option_items(
                "overseas_bd_customer_category",
                [
                    ("一般意向客户", "overseas_bd_category_normal_intention_customer"),
                    ("中等意向客户", "overseas_bd_category_medium_intention_customer"),
                    ("高意向客户", "overseas_bd_category_high_intention_customer"),
                ],
            ),
            field_item(
                "overseas_bd_normal_intention_cooperation_stage",
                "一般意向客户合作阶段",
                3,
                metadata={"visible_when": {"overseas_bd_customer_category": "overseas_bd_category_normal_intention_customer"}},
            ),
            *option_items(
                "overseas_bd_normal_intention_cooperation_stage",
                [
                    ("沉默客户", "overseas_bd_normal_stage_silent_customer"),
                    ("初步接触", "overseas_bd_normal_stage_initial_contact"),
                ],
            ),
            field_item(
                "overseas_bd_medium_intention_cooperation_stage",
                "中等意向客户合作阶段",
                4,
                metadata={"visible_when": {"overseas_bd_customer_category": "overseas_bd_category_medium_intention_customer"}},
            ),
            *option_items(
                "overseas_bd_medium_intention_cooperation_stage",
                [
                    ("需求确认", "overseas_bd_medium_stage_requirement_confirmed"),
                    ("样品测试", "overseas_bd_medium_stage_sample_test"),
                ],
            ),
            field_item(
                "overseas_bd_high_intention_cooperation_stage",
                "高意向客户合作阶段",
                5,
                metadata={"visible_when": {"overseas_bd_customer_category": "overseas_bd_category_high_intention_customer"}},
            ),
            *option_items(
                "overseas_bd_high_intention_cooperation_stage",
                [
                    ("管理层对接", "overseas_bd_high_stage_management_docking"),
                    ("沉默客户", "overseas_bd_high_stage_silent_customer"),
                    ("初步接触", "overseas_bd_high_stage_initial_contact"),
                    ("需求确认", "overseas_bd_high_stage_requirement_confirmed"),
                    ("样品测试", "overseas_bd_high_stage_sample_test"),
                    ("报价阶段", "overseas_bd_high_stage_quotation"),
                ],
            ),
            field_item("overseas_bd_customer_location", "客户所在地", 6),
            *option_items(
                "overseas_bd_customer_location",
                [
                    ("印度", "overseas_bd_location_india"),
                    ("马来西亚", "overseas_bd_location_malaysia"),
                    ("哈萨克斯坦", "overseas_bd_location_kazakhstan"),
                    ("土耳其", "overseas_bd_location_turkey"),
                    ("意大利", "overseas_bd_location_italy"),
                    ("匈牙利", "overseas_bd_location_hungary"),
                    ("智利", "overseas_bd_location_chile"),
                    ("法国", "overseas_bd_location_france"),
                    ("肯尼亚", "overseas_bd_location_kenya"),
                ],
            ),
            field_item("overseas_bd_company_size", "公司规模", 7),
            *option_items(
                "overseas_bd_company_size",
                [
                    ("微型团队/个体经营", "overseas_bd_company_micro_team_individual"),
                    ("小型企业", "overseas_bd_company_small_enterprise"),
                    ("初具规模企业", "overseas_bd_company_early_scale_enterprise"),
                    ("成长型企业", "overseas_bd_company_growth_enterprise"),
                    ("中大型企业", "overseas_bd_company_medium_large_enterprise"),
                    ("大型企业", "overseas_bd_company_large_enterprise"),
                ],
            ),
            field_item("overseas_bd_customer_source", "客户来源", 8),
            *option_items(
                "overseas_bd_customer_source",
                [
                    ("展会", "overseas_bd_source_exhibition"),
                    ("地推", "overseas_bd_source_ground_promotion"),
                    ("转介绍", "overseas_bd_source_referral"),
                    ("谷歌搜索", "overseas_bd_source_google_search"),
                    ("地图获客", "overseas_bd_source_map_customer_acquisition"),
                    ("商会", "overseas_bd_source_chamber_of_commerce"),
                    ("海关获客", "overseas_bd_source_customs_customer_acquisition"),
                    ("whatsapp社群", "overseas_bd_source_whatsapp_group"),
                ],
            ),
            field_item("overseas_bd_industry", "行业", 9),
            *option_items(
                "overseas_bd_industry",
                [
                    ("工程机械", "overseas_bd_industry_construction_machinery"),
                    ("农业机械", "overseas_bd_industry_agricultural_machinery"),
                    ("电动搬运", "overseas_bd_industry_electric_handling"),
                    ("木工", "overseas_bd_industry_woodworking"),
                    ("五金", "overseas_bd_industry_hardware"),
                    ("激光设备", "overseas_bd_industry_laser_equipment"),
                    ("户外用品", "overseas_bd_industry_outdoor_products"),
                    ("轮胎", "overseas_bd_industry_tire"),
                ],
            ),
            field_item("overseas_bd_service_content", "服务内容", 10),
            *option_items(
                "overseas_bd_service_content",
                [
                    ("产品", "overseas_bd_service_content_product"),
                    ("服务", "overseas_bd_service_content_service"),
                ],
            ),
            field_item("overseas_bd_product_core_focus", "产品核心关注点", 11),
            *option_items(
                "overseas_bd_product_core_focus",
                [
                    ("质量", "overseas_bd_core_quality"),
                    ("交期", "overseas_bd_core_delivery_time"),
                    ("售后服务", "overseas_bd_core_after_sales_service"),
                    ("品牌", "overseas_bd_core_brand"),
                    ("定制化", "overseas_bd_core_customization"),
                    ("合规认证/同行案例", "overseas_bd_core_compliance_peer_cases"),
                ],
            ),
            field_item("overseas_bd_service_core_focus", "服务核心关注点", 12),
            *option_items(
                "overseas_bd_service_core_focus",
                [
                    ("获客", "overseas_bd_core_customer_acquisition"),
                    ("转化", "overseas_bd_core_conversion"),
                    ("成本", "overseas_bd_core_cost"),
                    ("效果", "overseas_bd_core_effect"),
                    ("市场选择", "overseas_bd_core_market_selection"),
                ],
            ),
            field_item("overseas_bd_product_secondary_focus", "产品次要关注点", 13),
            *option_items(
                "overseas_bd_product_secondary_focus",
                [
                    ("质量", "overseas_bd_secondary_quality"),
                    ("交期", "overseas_bd_secondary_delivery_time"),
                    ("售后服务", "overseas_bd_secondary_after_sales_service"),
                    ("品牌", "overseas_bd_secondary_brand"),
                    ("定制化", "overseas_bd_secondary_customization"),
                    ("合规认证/同行案例", "overseas_bd_secondary_compliance_peer_cases"),
                ],
            ),
            field_item("overseas_bd_service_secondary_focus", "服务次要关注点", 14),
            *option_items(
                "overseas_bd_service_secondary_focus",
                [
                    ("获客", "overseas_bd_secondary_customer_acquisition"),
                    ("转化", "overseas_bd_secondary_conversion"),
                    ("成本", "overseas_bd_secondary_cost"),
                    ("效果", "overseas_bd_secondary_effect"),
                    ("市场选择", "overseas_bd_secondary_market_selection"),
                ],
            ),
            field_item("overseas_bd_price_sensitivity", "价格敏感度", 15),
            *option_items(
                "overseas_bd_price_sensitivity",
                [
                    ("极低", "overseas_bd_price_very_low"),
                    ("低", "overseas_bd_price_low"),
                    ("中", "overseas_bd_price_medium"),
                    ("高", "overseas_bd_price_high"),
                    ("极高", "overseas_bd_price_very_high"),
                ],
            ),
            field_item("overseas_bd_decision_maker", "是否决策人", 16),
            *option_items(
                "overseas_bd_decision_maker",
                [
                    ("是", "overseas_bd_decision_maker_yes"),
                    ("否", "overseas_bd_decision_maker_no"),
                ],
            ),
            field_item("overseas_bd_applicable_product", "适用产品", 17),
            *option_items(
                "overseas_bd_applicable_product",
                [
                    ("代购", "overseas_bd_product_purchasing_agent"),
                    ("代采", "overseas_bd_product_procurement_agent"),
                    ("代卖", "overseas_bd_product_sales_agent"),
                    ("代发", "overseas_bd_product_fulfillment_agent"),
                    ("物流", "overseas_bd_product_logistics"),
                    ("品宣", "overseas_bd_product_brand_promotion"),
                    ("代验", "overseas_bd_product_inspection_agent"),
                ],
            ),
            field_item("overseas_bd_personality", "性格特征", 18),
            *option_items(
                "overseas_bd_personality",
                [
                    ("直接果断", "overseas_bd_personality_direct_decisive"),
                    ("谨慎犹豫", "overseas_bd_personality_cautious_hesitant"),
                    ("挑剔细节", "overseas_bd_personality_detail_oriented"),
                    ("强势主导", "overseas_bd_personality_dominant"),
                ],
            ),
            field_item("overseas_bd_gender", "性别", 19),
            *option_items(
                "overseas_bd_gender",
                [
                    ("男", "overseas_bd_gender_male"),
                    ("女", "overseas_bd_gender_female"),
                ],
            ),
            field_item("overseas_bd_age", "年龄", 20, input_type="text"),
            field_item("overseas_bd_other", "其他", 21, input_type="text"),
        ],
    },
]
