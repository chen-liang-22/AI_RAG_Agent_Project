$ErrorActionPreference = "Stop"

# 切换到脚本所在目录，避免 PyCharm 从其他工作目录启动时找不到配置文件。
Set-Location -Path $PSScriptRoot

# 使用 UTF-8 输出，减少中文日志在 PowerShell / PyCharm 控制台里乱码。
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# 关闭 Python 输出缓冲，确保日志实时显示在 PyCharm 控制台。
$env:PYTHONUNBUFFERED = "1"

# 控制项目日志输出级别；需要更细日志时把 INFO 改成 DEBUG。
$env:AGENT_CONSOLE_LOG_LEVEL = "INFO"
$env:AGENT_FILE_LOG_LEVEL = "DEBUG"

# 激活项目 Python 3.12 虚拟环境。
& "$PSScriptRoot\.venv312\Scripts\Activate.ps1"

# 前台启动 API 服务。必须在 PyCharm Terminal / Run 窗口里运行，控制台才能看到实时日志。
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
