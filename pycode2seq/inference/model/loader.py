import os

import requests
import tarfile

from pathlib import Path

class ModelLoader:
    URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def model_path(name: str):
        cache_path = Path.home() / Path(".cache", "pycode2seq")
        return os.path.join(cache_path, name)

    @staticmethod
    def load(name: str, file_id: str) -> str:
        path = ModelLoader.model_path(name)

        session = requests.Session()

        response = session.get(ModelLoader.URL, params={'id': file_id}, stream=True)
        token = ModelLoader._get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(ModelLoader.URL, params=params, stream=True)

        model_tar_path = os.path.join(path, "model.tar.xz")
        ModelLoader._save_response_content(response, model_tar_path)

        ModelLoader._extract_archive(model_tar_path, path)

        return path

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    @staticmethod
    def _save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    @staticmethod
    def _extract_archive(archive_path: str, extract_path: str):
        tar = tarfile.open(archive_path)
        tar.extractall(extract_path)
        tar.close()
