"""V2 领域常量。

只放稳定的工程级默认值。真正需要业务人员配置的内容，优先走字典或配置文件。
"""

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 50
HOME_PAGE_SIZE = 6

DEFAULT_KNOWLEDGE_COLLECTION = "agent"
TRAINING_COLLECTION = "sales_training_cases"
TRAINING_STAGING_COLLECTION = "sales_training_cases_staging"
