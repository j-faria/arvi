import sys

def setup_logger():
    from loguru import logger
    try:
        import marimo as mo
        if mo.running_in_notebook():
            raise NotImplementedError
    except (NotImplementedError, AttributeError):
        pass
    except (ImportError, ModuleNotFoundError):
        logger.remove()
    else:
        logger.remove()

    logger.configure(extra={"indent": ""})
    logger.add(
        sys.stdout,
        colorize=True,
        # format="<green>{time:YYYY-MM-DDTHH:mm:ss}</green> <level>{message}</level>",
        format="{extra[indent]}<level>{message}</level>",
    )

    return logger
