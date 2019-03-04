
import logging
import pickle
import os
from pathlib import Path, PosixPath
import shutil
import tempfile

import msgpack
import requests
from tqdm import tqdm

from claf import nsml

logger = logging.getLogger(__name__)


class CachePath:
    if nsml.IS_ON_NSML:
        ROOT = Path("./claf_cache")
    else:
        ROOT = Path.home() / ".claf_cache"
    DATASET = ROOT / "dataset"
    MACHINE = ROOT / "machine"
    PRETRAINED_VECTOR = ROOT / "pretrained_vector"
    TOKEN_COUNTER = ROOT / "token_counter"
    VOCAB = ROOT / "vocab"


class DataHandler:
    """
    DataHandler with CachePath

    - read (from_path, from_http)
    - dump (.msgpack or .pkl (pickle))
    - load
    """

    def __init__(self, cache_path=CachePath.ROOT):
        if type(cache_path) != PosixPath:
            raise ValueError(f"cache_path type is PosixPath (use pathlib.Path). not f{type(cache_path)}")

        self.cache_path = cache_path
        cache_path.mkdir(parents=True, exist_ok=True)

    def convert_cache_path(self, path):
        cache_data_path = self.cache_path / Path(path)
        return cache_data_path

    def read_embedding(self, file_path):
        raise NotImplementedError()

    def read(self, file_path, encoding="utf-8", return_path=False):
        if file_path.startswith("http"):
           file_path = self._read_from_http(file_path, encoding)

        path = Path(file_path)
        if path.exists():
            if return_path:
                return path
            return path.read_bytes().decode(encoding)

        if nsml.IS_ON_NSML:
            path = nsml.DATASET_PATH / path

        if path.exists():
            if return_path:
                return path
            return path.read_bytes().decode(encoding)
        else:
            raise FileNotFoundError(f"{file_path} is not found.")

    def _read_from_http(self, file_path, encoding, return_path=False):
        cache_data_path = self.cache_path / Path(file_path).name
        if cache_data_path.exists():
            logger.info(f"'{file_path}' is already downloaded.")
            pass
        else:
            with tempfile.TemporaryFile() as temp_file:
                self._download_from_http(temp_file, file_path)
                temp_file.flush()
                temp_file.seek(0)

                with open(cache_data_path, 'wb') as cache_file:
                    shutil.copyfileobj(temp_file, cache_file)

        return cache_data_path

    def _download_from_http(self, temp_file, url):
        req = requests.get(url, stream=True)
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        with tqdm(total=total, unit="B", unit_scale=True, desc="download...") as pbar:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    temp_file.write(chunk)
                    pbar.update(len(chunk))

    def cache_token_counter(self, data_reader_config, tokenizer_name, obj=None):
        data_paths = os.path.basename(data_reader_config.train_file_path)
        if getattr(data_reader_config, "valid_file_path", None):
            data_paths += "#" + os.path.basename(data_reader_config.valid_file_path)

        path = self.cache_path / data_reader_config.dataset / data_paths
        path.mkdir(parents=True, exist_ok=True)
        path = path / tokenizer_name

        if obj:
            self.dump(path, obj)
        else:
            return self.load(path)

    def load(self, file_path, encoding="utf-8"):
        path = self.cache_path / file_path
        logger.info(f"load path: {path}")

        msgpack_path = path.with_suffix(".msgpack")
        if msgpack_path.exists():
            return self._load_msgpack(msgpack_path, encoding)

        pickle_path = path.with_suffix(".pkl")
        if pickle_path.exists():
            return self._load_pickle(pickle_path, encoding)

        return None

    def _load_msgpack(self, path, encoding):
        with open(path, "rb") as in_file:
            return msgpack.unpack(in_file, encoding=encoding)

    def _load_pickle(self, path, encoding):
        with open(path, "rb") as in_file:
            return pickle.load(in_file, encoding=encoding)

    def dump(self, file_path, obj, encoding="utf-8"):
        path = self.cache_path / file_path
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path.with_suffix(".msgpack"), "wb") as out_file:
                msgpack.pack(obj, out_file, encoding=encoding)
        except TypeError:
            os.remove(path.with_suffix(".msgpack"))
            with open(path.with_suffix(".pkl"), "wb") as out_file:
                pickle.dump(obj, out_file, protocol=pickle.HIGHEST_PROTOCOL)
