from pathlib import Path
from typing import List


def get_audio_pathlib(folder_path: str, full_path: bool = True) -> List:
    fp = [
        x
        for x in Path(folder_path).glob("**/*")
        if x.is_file() and x.suffix in [".wav",]
    ]

    if not full_path:
        fp = [x.name for x in fp]
    return fp
