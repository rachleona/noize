import os
import sys

from collections.abc import Callable
from pathlib import Path
from typing import Any
from packaging.version import Version

import fsspec
import torch
from torch.types import Storage

_WEIGHTS_ONLY = Version(torch.__version__) >= Version("2.4")


def get_user_data_dir(appname: str) -> Path:
    TTS_HOME = os.environ.get("TTS_HOME")
    XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
    if TTS_HOME is not None:
        ans = Path(TTS_HOME).expanduser().resolve(strict=False)
    elif XDG_DATA_HOME is not None:
        ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
    elif sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)


def load_fsspec(
    path: str | os.PathLike[Any],
    map_location: (
        str | Callable[[Storage, str], Storage] | torch.device | dict[str, str] | None
    ) = None,
    *,
    cache: bool = True,
    **kwargs: Any,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/trainer_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    is_local = Path(path).exists()
    if cache and not is_local:
        with fsspec.open(
            f"filecache::{path}",
            filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
            mode="rb",
        ) as f:
            return torch.load(
                f, map_location=map_location, weights_only=_WEIGHTS_ONLY, **kwargs
            )
    else:
        with fsspec.open(str(path), "rb") as f:
            return torch.load(
                f, map_location=map_location, weights_only=_WEIGHTS_ONLY, **kwargs
            )
