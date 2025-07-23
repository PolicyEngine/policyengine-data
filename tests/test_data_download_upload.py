"""
Test data download and upload functionality.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
import sys
import threading
from policyengine_core.tools.win_file_manager import WindowsAtomicFileManager
import tempfile
from pathlib import Path
import uuid


def test_atomic_write():
    if sys.platform != "win32":
        from policyengine_core.data.dataset import atomic_write

        with NamedTemporaryFile(mode="w") as file:
            file.write("Hello, world\n")
            file.flush()
            # Open the file before overwriting
            with open(file.name, "r") as file_original:

                atomic_write(Path(file.name), "NOPE\n".encode())

                # Open file descriptor still points to the old node
                assert file_original.readline() == "Hello, world\n"
                # But if I open it again it has the new content
                with open(file.name, "r") as file_updated:
                    assert file_updated.readline() == "NOPE\n"


def test_atomic_write_windows():
    if sys.platform == "win32":
        temp_dir = Path(tempfile.gettempdir())
        temp_files = [
            temp_dir / f"tempfile_{uuid.uuid4().hex}.tmp" for _ in range(5)
        ]

        managers = [WindowsAtomicFileManager(path) for path in temp_files]

        contents_list = [
            [f"Content_{i}_{j}".encode() for j in range(5)] for i in range(5)
        ]

        check_results = [[] for _ in range(5)]

        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=file_task,
                args=(managers[i], contents_list[i], check_results[i]),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for i, results in enumerate(check_results):
            for expected, actual in results:
                assert (
                    expected == actual
                ), f"Mismatch in file {i}: expected {expected}, got {actual}"

        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()


def file_task(manager, contents, check_results):
    for content in contents:
        manager.write(content)
        actual_content = manager.read().decode()
        expected_content = content.decode()
        check_results.append((expected_content, actual_content))
