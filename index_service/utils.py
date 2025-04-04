import json
import os
import random
import re
import time
from hashlib import md5, sha256
from pathlib import Path
from typing import Literal

from llama_index.core import Settings

from index_service import app_config, logger

# import chardet


class DocCacheManager:
    DOCUMENT_CACHE_FLODER = "./transform_cache/doc_cache"
    DOCUMENT_CACHE_INFO = f"{DOCUMENT_CACHE_FLODER}/doc_info.json"
    DOCUMENT_CACHE_LOCK = f"{DOCUMENT_CACHE_FLODER}/doc_lock.json"
    DOCUMENT_CACHE_LOCK_LIFESPAN = 1200  # second
    DOCUMENT_CACHE_LOCK_RETRY = 5  # retry get lock times.

    def __init__(self):
        if not os.path.exists(self.DOCUMENT_CACHE_FLODER):
            os.makedirs(self.DOCUMENT_CACHE_FLODER, exist_ok=True)

    def __enter__(self):
        for _ in range(self.DOCUMENT_CACHE_LOCK_RETRY):
            if self._get_lock():
                self._load_doc_cache()
                break
            time.sleep(random.randint(0, 10) / 10)
            logger.info(f"get doc cache log error , retry : {_} times")
        else:
            logger.info(f"Cache Invalidation , Can't get lock....")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._patch_doc_cache()
        self._free_lock()

    def __call__(self, func: Literal["save", "load"]):
        if func == "load":
            self._load_doc_cache()
        elif func == "save":
            self._patch_doc_cache()
        else:
            raise ValueError(f"not support function parameters.....")
        return self

    def _get_lock(self):
        if not os.path.exists(self.DOCUMENT_CACHE_LOCK):
            lock_obj = {"CreateTime": time.time()}
            with open(self.DOCUMENT_CACHE_LOCK, "w") as f:
                json.dump(lock_obj, f)
            return lock_obj
        else:
            with open(self.DOCUMENT_CACHE_LOCK, "r") as f:
                lock_obj = json.load(f)
            now_time = time.time()
            if now_time - lock_obj["CreateTime"] > self.DOCUMENT_CACHE_LOCK_LIFESPAN:
                lock_obj["CreateTime"] = now_time
                with open(self.DOCUMENT_CACHE_LOCK, "w") as f:
                    json.dump(lock_obj, f)
                return lock_obj
            else:
                return None

    def _free_lock(self):
        if os.path.exists(self.DOCUMENT_CACHE_FLODER):
            os.remove(self.DOCUMENT_CACHE_LOCK)
            return True
        else:
            return False

    def _load_doc_cache(self):
        if os.path.exists(self.DOCUMENT_CACHE_INFO):
            with open(self.DOCUMENT_CACHE_INFO, "r") as f:
                self.doc_cache = json.load(f)
        else:
            self.doc_cache = {}

    def _patch_doc_cache(self):
        if not os.path.exists(self.DOCUMENT_CACHE_FLODER):
            os.makedirs(self.DOCUMENT_CACHE_FLODER, exist_ok=True)
        with open(self.DOCUMENT_CACHE_INFO, "w") as f:
            json.dump(self.doc_cache, f)


def rename_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            sanitized_name = re.sub(r"[^\w.\-]", "_", file_name)
            sanitized_name = sanitized_name.lstrip("_")
            if file_name != sanitized_name:
                old_path = os.path.join(root, file_name)
                new_path = os.path.join(root, sanitized_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {file_name} -> {sanitized_name}")
                except Exception as e:
                    print(f"Failed to rename {file_name}: {e}")


# def detect_encoding(file_path, num_bytes=-1):
#     with open(file_path, "rb") as f:
#         raw_data = f.read(num_bytes)
#     result = chardet.detect(raw_data)
#     return result["encoding"]


def get_text_hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def get_text_md5(text):
    _md5 = md5()
    _md5.update(text.encode("utf-8"))
    return _md5.hexdigest()


def save_json_file(path, content):
    with open(path, "w") as f:
        json.dump(content, f)


def load_json_file(path, default=[]):
    try:
        with open(path, "r") as f:
            result = json.load(f)
    except Exception as e:
        logger.error(f"Load json file {path} failed, error info: {e}")
        result = default

    return result


def safe_save_file(func, path, *args, **kwargs):
    new_path = (
        Path(path).parent
        / f"{Path(path).name.rsplit('.', 1)[0]}_neweast.{Path(path).name.rsplit('.', 1)[-1]}"
    )
    func(new_path, *args, **kwargs)
    if os.path.exists(path):
        logger.debug(f"update file: {new_path}")
        os.remove(path)
        os.rename(new_path, path)
    else:
        os.rename(new_path, path)


def get_collection_name(
    project_name: str, data_type: str, knowledge_name: str, subfix=""
) -> str:
    if subfix:
        subfix = f"_{subfix.removeprefix("_")}"
    collection_name = f"{project_name}_{data_type}_{knowledge_name}{subfix}"
    return collection_name


def get_embedding_size():
    return len(Settings.embed_model.get_text_embedding("Hello World!"))


def construct_dir_path(
    base_path: str = app_config.persist_dir, parts: list[str] = []
) -> Path:
    full_path = Path(base_path).joinpath(*parts)
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path
