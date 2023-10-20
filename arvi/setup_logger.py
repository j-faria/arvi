import sys
from loguru import logger

logger.remove()
logger.configure(extra={"indent": ""})
logger.add(
    sys.stdout,
    colorize=True,
    # format="<green>{time:YYYY-MM-DDTHH:mm:ss}</green> <level>{message}</level>",
    format="{extra[indent]}<level>{message}</level>",
)
