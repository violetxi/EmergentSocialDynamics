import os


def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        print(f"Creating directory {directory}")
        os.makedirs(directory)