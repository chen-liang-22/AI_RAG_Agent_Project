import logging
from utils.path_tool import get_abs_path
import os
import sys
from datetime import datetime

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保日志的目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置  error info debug
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
DEFAULT_LOG_FORMAT = logging.Formatter(LOG_FORMAT)


class ColorFormatter(logging.Formatter):
    """给 PyCharm 控制台的不同日志节点加颜色，文件日志仍保持纯文本。"""

    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    TIME_COLOR = "\033[90m"
    LOGGER_COLOR = "\033[35m"
    LOCATION_COLOR = "\033[36m"
    MESSAGE_COLOR = "\033[37m"
    RESET = "\033[0m"

    def __init__(self, fmt: str | None = None):
        super().__init__(fmt or LOG_FORMAT)
        self.plain_formatter = logging.Formatter(LOG_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        if os.getenv("NO_COLOR"):
            return self.plain_formatter.format(record)

        time_text = self.formatTime(record, self.datefmt)
        level_color = self.LEVEL_COLORS.get(record.levelno, self.MESSAGE_COLOR)
        message = record.getMessage()
        location = f"{record.filename}:{record.lineno}"

        colored_message = (
            f"{self.TIME_COLOR}{time_text}{self.RESET} - "
            f"{self.LOGGER_COLOR}{record.name}{self.RESET} - "
            f"{level_color}{record.levelname}{self.RESET} - "
            f"{self.LOCATION_COLOR}{location}{self.RESET} - "
            f"{self.MESSAGE_COLOR}{message}{self.RESET}"
        )

        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            colored_message = f"{colored_message}\n{level_color}{exception_text}{self.RESET}"
        if record.stack_info:
            stack_text = self.formatStack(record.stack_info)
            colored_message = f"{colored_message}\n{self.LOCATION_COLOR}{stack_text}{self.RESET}"

        return colored_message


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加Handler
    if logger.handlers:
        return logger

    # 控制台Handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColorFormatter(LOG_FORMAT))

    logger.addHandler(console_handler)

    # 文件Handler
    if not log_file:        # 日志文件的存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger


# 快捷获取日志器
logger = get_logger()


if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
