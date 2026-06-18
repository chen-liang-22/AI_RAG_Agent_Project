import os
import sys
from datetime import datetime
import logging

from utils.path_tool import get_abs_path

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保日志的目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的格式配置  error info debug
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
DEFAULT_LOG_FORMAT = logging.Formatter(LOG_FORMAT)
CONSOLE_HANDLER_MARK = "_agent_console_handler"
FILE_HANDLER_MARK = "_agent_file_handler"


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
        log_file: str | None = None,
) -> logging.Logger:
    """获取项目日志器，并确保控制台和文件日志都已挂载。

    注意：uvicorn --reload 和 PyCharm 控制台都可能影响 stdout/stderr，
    因此这里不能只要发现 logger.handlers 不为空就直接返回。
    每次调用都要检查控制台 handler 和文件 handler 是否仍然存在。
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    final_console_level = _read_log_level_from_env("AGENT_CONSOLE_LOG_LEVEL", console_level)
    final_file_level = _read_log_level_from_env("AGENT_FILE_LOG_LEVEL", file_level)
    final_log_file = log_file or os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    _ensure_console_handler(logger, final_console_level)
    _ensure_file_handler(logger, final_file_level, final_log_file)

    return logger


def _read_log_level_from_env(env_name: str, default_level: int) -> int:
    """从环境变量读取日志级别，未配置或配置错误时使用默认级别。"""

    raw_level = os.getenv(env_name)
    if not raw_level:
        return default_level

    clean_level = raw_level.strip().upper()
    if clean_level.isdigit():
        return int(clean_level)
    return int(getattr(logging, clean_level, default_level))


def _ensure_console_handler(logger: logging.Logger, console_level: int) -> None:
    """确保控制台日志 handler 存在，并绑定到当前进程的 stdout。"""

    for handler in list(logger.handlers):
        if _is_console_handler(handler):
            stream = getattr(handler, "stream", None)
            if getattr(stream, "closed", False):
                logger.removeHandler(handler)
                handler.close()
                continue
            handler.setLevel(console_level)
            handler.setFormatter(ColorFormatter(LOG_FORMAT))
            setattr(handler, CONSOLE_HANDLER_MARK, True)
            return

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColorFormatter(LOG_FORMAT))
    setattr(console_handler, CONSOLE_HANDLER_MARK, True)
    logger.addHandler(console_handler)


def _ensure_file_handler(logger: logging.Logger, file_level: int, log_file: str) -> None:
    """确保文件日志 handler 存在，并写入当天日志文件。"""

    target_path = os.path.abspath(log_file)
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        current_path = os.path.abspath(getattr(handler, "baseFilename", ""))
        if current_path == target_path:
            handler.setLevel(file_level)
            handler.setFormatter(DEFAULT_LOG_FORMAT)
            setattr(handler, FILE_HANDLER_MARK, True)
            return

    file_handler = logging.FileHandler(target_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    setattr(file_handler, FILE_HANDLER_MARK, True)
    logger.addHandler(file_handler)


def _is_console_handler(handler: logging.Handler) -> bool:
    """判断 handler 是否为控制台 handler。"""

    if getattr(handler, CONSOLE_HANDLER_MARK, False):
        return True
    return isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)


# 快捷获取日志器
logger = get_logger()


if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
