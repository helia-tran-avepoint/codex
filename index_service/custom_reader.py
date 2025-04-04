from __future__ import annotations

import asyncio
import importlib
import json
import mimetypes
import multiprocessing
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from functools import reduce
from itertools import repeat
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Generator, List, Optional, Type, cast

import fsspec
import pandas as pd
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from llama_index.core.async_utils import get_asyncio_module, run_jobs
from llama_index.core.readers.base import BaseReader, ResourcesReaderMixin
from llama_index.core.schema import Document
from tqdm import tqdm

# from index_service.utils import detect_encoding


class PandasCSVReader(BaseReader):
    r"""Pandas-based CSV parser.

    Parses CSVs using the separator detection from Pandas `read_csv`function.
    If special parameters are required, use the `pandas_config` dict.

    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.

        col_joiner (str): Separator to use for joining cols per row.
            Set to ", " by default.

        row_joiner (str): Separator to use for joining each row.
            Only used when `concat_rows=True`.
            Set to "\n" by default.

        pandas_config (dict): Options for the `pandas.read_csv` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
            for more information.
            Set to empty dict by default, this means pandas will try to figure
            out the separators, table head, etc. on its own.

    """

    def __init__(
        self,
        *args: Any,
        concat_rows: bool = True,
        col_joiner: str = ", ",
        row_joiner: str = "\n",
        pandas_config: dict = {},
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._col_joiner = col_joiner
        self._row_joiner = row_joiner
        self._pandas_config = pandas_config

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""

        # encoding = detect_encoding(str(file))
        # self._pandas_config["encoding"] = encoding

        if fs:
            with fs.open(file) as f:
                df = pd.read_csv(f, **self._pandas_config)
        else:
            df = pd.read_csv(file, **self._pandas_config)

        text_list = df.apply(
            lambda row: (self._col_joiner).join(row.astype(str).tolist()), axis=1
        ).tolist()

        if self._concat_rows:
            return [
                Document(
                    text=(self._row_joiner).join(text_list), metadata=extra_info or {}  # type: ignore
                )
            ]
        else:
            return [
                Document(text=text, metadata=extra_info or {}) for text in text_list  # type: ignore
            ]


class PandasExcelReader(BaseReader):
    r"""Pandas-based Excel parser.

    Parses Excel files using the Pandas `read_excel`function.
    If special parameters are required, use the `pandas_config` dict.

    Args:
        concat_rows (bool): whether to concatenate all rows into one document.
            If set to False, a Document will be created for each row.
            True by default.

        sheet_name (str | int | None): Defaults to None, for all sheets, otherwise pass a str or int to specify the sheet to read.

        pandas_config (dict): Options for the `pandas.read_excel` function call.
            Refer to https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
            for more information.
            Set to empty dict by default.

    """

    def __init__(
        self,
        *args: Any,
        concat_rows: bool = True,
        sheet_name=None,
        pandas_config: dict = {},
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)
        self._concat_rows = concat_rows
        self._sheet_name = sheet_name
        self._pandas_config = pandas_config

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        openpyxl_spec = importlib.util.find_spec("openpyxl")  # type: ignore
        if openpyxl_spec is not None:
            pass
        else:
            raise ImportError(
                "Please install openpyxl to read Excel files. You can install it with 'pip install openpyxl'"
            )

        # sheet_name of None is all sheets, otherwise indexing starts at 0
        if fs:
            with fs.open(file) as f:
                dfs = pd.read_excel(f, self._sheet_name, **self._pandas_config)
        else:
            dfs = pd.read_excel(file, self._sheet_name, **self._pandas_config)

        documents = []

        # handle the case where only a single DataFrame is returned
        if isinstance(dfs, pd.DataFrame):
            df = dfs.fillna("")

            # Convert DataFrame to list of rows

            text_obj = df.astype(str).apply(lambda row: " ".join(row.values), axis=1)

            if not text_obj.empty:
                text_list = text_obj.tolist()

                if self._concat_rows:
                    documents.append(
                        Document(text="\n".join(text_list), metadata=extra_info or {})
                    )
                else:
                    documents.extend(
                        [
                            Document(text=text, metadata=extra_info or {})
                            for text in text_list
                        ]
                    )
        else:
            for df in dfs.values():
                df = df.fillna("")

                # Convert DataFrame to list of rows

                text_obj = df.astype(str).apply(lambda row: " ".join(row), axis=1)

                if text_obj.empty:
                    continue
                text_list = text_obj.tolist()

                if self._concat_rows:
                    documents.append(
                        Document(text="\n".join(text_list), metadata=extra_info or {})
                    )
                else:
                    documents.extend(
                        [
                            Document(text=text, metadata=extra_info or {})
                            for text in text_list
                        ]
                    )

        return documents


class SourceCodeJsonReader(BaseReader):
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "sourceCode" in data:
                content = data.get("sourceCode", "")  # should be enhanced
                metadata = data
                metadata.update(extra_info or {})
            else:
                content = data
                metadata = {}
        # return [Document(text=data)]
        return [
            Document(
                text=content,
                metadata=metadata,
                excluded_embed_metadata_keys=["sourceCode"],
                excluded_llm_metadata_keys=["sourceCode"],
            )
        ]
