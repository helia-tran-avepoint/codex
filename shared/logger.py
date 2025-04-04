import logging.config
import os
import re
import sys
from logging.handlers import RotatingFileHandler

os.makedirs("./logs", exist_ok=True)

LOGGER_LEVEL = {
    logging.ERROR: ["httpx", "matplotlib", "websockets", "urllib3"],
    logging.INFO: [
        "graphviz",
        "trulens",
        "docker.auth",
        "httpcore",
        "openai._base_client",
        "chromadb",
        "core.response_synthesizers.refine",
        "storage.kvstore.simple_kvstore",
        "fsspec.implementations.local",
    ],
}


class PathFormatter(logging.Formatter):
    def format(self, record):
        if "pathname" in record.__dict__.keys():
            words = re.split(r"[\\/]", record.pathname)
            record.pathname = "/".join(words[-3:])
        return super().format(record)


def get_logger(sevice_name: str = "", debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("root")
    for level, packages in LOGGER_LEVEL.items():
        for package in packages:
            logging.getLogger(package).setLevel(level)

    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    formatter = PathFormatter(
        fmt="%(asctime)s %(levelname)-8s: %(threadName)s %(pathname)s::%(funcName)s[line:%(lineno)d] %(message)s"
    )

    file_name = sevice_name if sevice_name else "sevice"
    rotateHandler = RotatingFileHandler(
        f"./logs/{file_name}.log", "a", 10 * 1024 * 1024, 10
    )
    rotateHandler.setFormatter(formatter)
    rotateHandler.setLevel(log_level)
    logger.addHandler(rotateHandler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    return logger


# filter print info in console
original_print = print


def custom_print(*args, **kwargs):
    message = " ".join(map(str, args))
    if "instrumenting <class" not in message:
        original_print(*args, **kwargs)


sys.modules["builtins"].print = custom_print
