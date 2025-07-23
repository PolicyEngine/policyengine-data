"""
Functionality for uploading and downloading datasets in PolicyEngine.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import pandas as pd
import requests

from .tools.hugging_face import *
from .tools.win_file_manager import WindowsAtomicFileManager


def atomic_write(file: Path, content: bytes) -> None:
    """
    Atomically update the target file with the content. Any existing file will be unlinked rather than overritten.

    Implemented by
    1. Downloading the file to a temporary file with a unique name
    2. renaming (not copying) the file to the target name so that the operation is atomic (either the file is there or it's not, no partial file)

    If a process is reading the original file when a new file is renamed, that should relink the file, not clear and overwrite the old one so
    both processes should continue happily.
    """
    if sys.platform == "win32":
        manager = WindowsAtomicFileManager(file)
        manager.write(content)
    else:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=file.parent.absolute().as_posix(),
            prefix=file.name + ".download.",
            delete=False,
        ) as f:
            try:
                f.write(content)
                f.close()
                os.rename(f.name, file.absolute().as_posix())
            except:
                f.delete = True
                f.close()
                raise


def download(local_dir: Path, url: str, version: Optional[str] = None) -> None:
    """Downloads a file to the dataset's file path.

    Args:
        local_dir (Path): The path to save the downloaded file.
        url (str): The url to download.
        version (Optional[str]): The version of the file to download. Defaults to None.
    """

    if url is None:
        url = self.url

    if "POLICYENGINE_GITHUB_MICRODATA_AUTH_TOKEN" not in os.environ:
        auth_headers = {}
    else:
        auth_headers = {
            "Authorization": f"token {os.environ['POLICYENGINE_GITHUB_MICRODATA_AUTH_TOKEN']}",
        }

    # "release://" is a special protocol for downloading from GitHub releases
    # e.g. release://policyengine/policyengine-us/cps-2023/cps_2023.h5
    # release://org/repo/release_tag/file_path
    # Use the GitHub API to get the download URL for the release asset

    if url.startswith("release://"):
        org, repo, release_tag, file_path = url.split("/")[2:]
        url = f"https://api.github.com/repos/{org}/{repo}/releases/tags/{release_tag}"
        response = requests.get(url, headers=auth_headers)
        if response.status_code != 200:
            raise ValueError(
                f"Invalid response code {response.status_code} for url {url}."
            )
        assets = response.json()["assets"]
        for asset in assets:
            if asset["name"] == file_path:
                url = asset["url"]
                break
        else:
            raise ValueError(
                f"File {file_path} not found in release {release_tag} of {org}/{repo}."
            )
    elif url.startswith("hf://"):
        owner_name, model_name, file_name = url.split("/")[2:]
        _download_from_huggingface(
            local_dir, owner_name, model_name, file_name, version
        )
        return
    else:
        url = url

    response = requests.get(
        url,
        headers={
            "Accept": "application/octet-stream",
            **auth_headers,
        },
    )

    if response.status_code != 200:
        raise ValueError(
            f"Invalid response code {response.status_code} for url {url}."
        )

    atomic_write(file_path, response.content)


def _download_from_huggingface(
    local_dir: Path,
    owner_name: str,
    model_name: str,
    file_name: str,
    version: Optional[str] = None,
) -> None:
    """Downloads the dataset from HuggingFace.

    Args:
        local_dir (Path): The path to save the downloaded file.
        owner_name (str): The owner name.
        model_name (str): The model name.
        file_name (str): The file name.
        version (Optional[str]): The version of the file to download. Defaults to None.
    """

    print(
        f"Downloading from HuggingFace {owner_name}/{model_name}/{file_name}",
        file=sys.stderr,
    )

    download_huggingface_dataset(
        repo=f"{owner_name}/{model_name}",
        repo_filename=file_name,
        version=version,
        local_dir=local_dir,
    )


def upload(file_path: Path, url: str) -> None:
    """Uploads the dataset to a URL.

    Args:
        file_path (Path): The path to the file to upload.
        url (str): The url to upload.
    """
    if url.startswith("hf://"):
        owner_name, model_name, file_name = url.split("/")[2:]
        _upload_to_huggingface(file_path, owner_name, model_name, file_name)


def _upload_to_huggingface(
    file_path: Path, owner_name: str, model_name: str, file_name: str
) -> None:
    """Uploads the dataset to HuggingFace.

    Args:
        file_path (Path): The path to the file to upload.
        owner_name (str): The owner name.
        model_name (str): The model name.
    """

    print(
        f"Uploading to HuggingFace {owner_name}/{model_name}/{file_name}",
        file=sys.stderr,
    )

    token = get_or_prompt_hf_token()
    api = HfApi()

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name,
        repo_id=f"{owner_name}/{model_name}",
        repo_type="model",
        token=token,
    )
