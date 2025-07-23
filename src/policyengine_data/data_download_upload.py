"""
Functionality for uploading and downloading datasets in PolicyEngine.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Enum, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import requests

from .tools.hugging_face import *
from .tools.win_file_manager import WindowsAtomicFileManager

logger = logging.getLogger(__name__)


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


class CloudLocation(Enum):
    HUGGING_FACE = "HUGGING_FACE"
    GOOGLE_CLOUD_STORAGE = "GOOGLE_CLOUD_STORAGE"


def download(
    local_dir: Path,
    url: str,
    cloud_location: Optional[str] = None,
    version: Optional[str] = None,
) -> None:
    """Downloads a file from a cloud location to the local directory.

    Args:
        local_dir (Path): The path to save the downloaded file.
        url (str): The url to download from.
        cloud_location (Optional[str]): The cloud location to download from. Defaults to None.
        version (Optional[str]): The version of the file to download. Defaults to None.
    """
    if cloud_location is None:
        cloud_location = identify_location(url)
    else:
        if cloud_location not in CloudLocation:
            raise ValueError(f"Unsupported cloud location: {cloud_location}")

    if cloud_location == CloudLocation.HUGGING_FACE:
        owner, model, filename = parse_hugging_face_url(url)
        download_from_hugging_face(local_dir, owner, model, filename, version)
    elif cloud_location == CloudLocation.GOOGLE_CLOUD_STORAGE:
        download_from_gcs(local_dir, url)
    else:
        raise ValueError(f"Unsupported cloud location for URL: {url}")


def upload(
    local_dir: Path, url: str, cloud_location: Optional[str] = None
) -> None:
    """Uploads a file from the local directory to a cloud location.

    Args:
        local_dir (Path): The path to the directory containing the file to upload.
        url (str): The url to upload to.
        cloud_location (Optional[str]): The cloud location to upload to. Defaults to None.
    """
    if cloud_location is None:
        cloud_location = identify_location(url)
    else:
        if cloud_location not in CloudLocation:
            raise ValueError(f"Unsupported cloud location: {cloud_location}")

    if cloud_location == CloudLocation.HUGGING_FACE:
        owner, model, filename = parse_hugging_face_url(url)
        upload_to_hugging_face(local_dir, owner, model, filename)
    elif cloud_location == CloudLocation.GOOGLE_CLOUD_STORAGE:
        upload_to_gcs(local_dir, url)
    else:
        raise ValueError(f"Unsupported cloud location for URL: {url}")


def identify_location(url: str) -> CloudLocation:
    """Identifies the cloud storage location from a URL.

    Args:
        url (str): The URL to analyze.

    Returns:
        CloudLocation: The identified cloud storage location.

    Raises:
        ValueError: If the URL format is not recognized.
    """
    if url.startswith("hf://"):
        return CloudLocation.HUGGING_FACE
    elif url.startswith("gs://") or url.startswith(
        "https://storage.googleapis.com/"
    ):
        return CloudLocation.GOOGLE_CLOUD_STORAGE
    else:
        # Default to Hugging Face for backwards compatibility
        return CloudLocation.HUGGING_FACE


def validate_hugging_face_url(url: str) -> bool:
    """Validates a Hugging Face URL format.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not url.startswith("hf://"):
        return False

    parts = url[5:].split("/")  # Remove "hf://" prefix
    return len(parts) == 3 and all(part for part in parts)


def parse_hugging_face_url(url: str) -> Tuple[str, str, str]:
    """Parses a Hugging Face URL into its components.

    Args:
        url (str): The Hugging Face URL to parse.

    Returns:
        Tuple[str, str, str]: Owner name, model name, and filename.

    Raises:
        ValueError: If the URL is not a valid Hugging Face URL.
    """
    if not validate_hugging_face_url(url):
        raise ValueError(f"Invalid Hugging Face URL format: {url}")

    parts = url[5:].split("/")  # Remove "hf://" prefix
    return parts[0], parts[1], parts[2]


def download_from_hugging_face(
    local_dir: Path,
    owner_name: str,
    model_name: str,
    file_name: str,
    version: Optional[str] = None,
) -> None:
    """Downloads a file from Hugging Face.

    Args:
        local_dir (Path): The path to save the downloaded file.
        owner_name (str): The owner name.
        model_name (str): The model name.
        file_name (str): The file name.
        version (Optional[str]): The version of the file to download.
    """
    logger.info(
        f"Downloading from HuggingFace {owner_name}/{model_name}/{file_name}",
    )

    download_huggingface_dataset(
        repo=f"{owner_name}/{model_name}",
        repo_filename=file_name,
        version=version,
        local_dir=local_dir,
    )


def upload_to_hugging_face(
    local_dir: Path, owner_name: str, model_name: str, file_name: str
) -> None:
    """Uploads a file to Hugging Face.

    Args:
        local_dir (Path): The path to the directory containing the file to upload.
        owner_name (str): The owner name.
        model_name (str): The model name.
        file_name (str): The file name.
    """
    logger.info(
        f"Uploading to HuggingFace {owner_name}/{model_name}/{file_name}",
    )

    token = get_or_prompt_hf_token()
    api = HfApi()

    api.upload_file(
        path_or_fileobj=local_dir,
        path_in_repo=file_name,
        repo_id=f"{owner_name}/{model_name}",
        repo_type="model",
        token=token,
    )


def download_from_gcs(local_dir: Path, url: str) -> None:
    """Downloads a file from Google Cloud Storage.

    Args:
        local_dir (Path): The path to save the downloaded file.
        url (str): The GCS URL to download from.
    """
    # Convert gs:// URLs to https:// if needed
    if url.startswith("gs://"):
        bucket_and_path = url[5:]
        url = f"https://storage.googleapis.com/{bucket_and_path}"

    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError(
            f"Failed to download from GCS. Status code: {response.status_code}"
        )

    # Extract filename from URL
    filename = url.split("/")[-1]
    file_path = local_dir / filename

    atomic_write(file_path, response.content)


def upload_to_gcs(local_dir: Path, url: str) -> None:
    """Uploads a file to Google Cloud Storage.

    Args:
        local_dir (Path): The path to the directory containing the file to upload.
        url (str): The GCS URL to upload to.
    """
    raise NotImplementedError(
        "Google Cloud Storage upload is not yet implemented. "
        "Please use Hugging Face for now."
    )
