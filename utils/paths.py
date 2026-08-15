import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


ASSET_ROOT_ENV = "T2ICOUNT_ASSET_ROOT"
PathLike = Union[str, os.PathLike]


class AssetPathError(FileNotFoundError):
    """Raised when a required external T2ICount asset is unavailable."""


def _expanded(path: PathLike) -> Path:
    # Python 3.8 on Windows can leave a nonexistent path relative after
    # Path.resolve(); abspath keeps fail-fast diagnostics unambiguous.
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def require_file(path: PathLike, label: str) -> Path:
    resolved = _expanded(path)
    if not resolved.is_file():
        raise AssetPathError("Missing {} file:\n{}".format(label, resolved))
    return resolved


def require_directory(path: PathLike, label: str) -> Path:
    resolved = _expanded(path)
    if not resolved.is_dir():
        raise AssetPathError("Missing {} directory:\n{}".format(label, resolved))
    return resolved


def resolve_asset_root(asset_root: Optional[PathLike] = None, required: bool = True) -> Optional[Path]:
    configured = asset_root or os.environ.get(ASSET_ROOT_ENV)
    if configured:
        root = _expanded(configured)
        if required and not root.is_dir():
            raise AssetPathError("Missing asset root directory:\n{}".format(root))
        return root
    if required:
        raise AssetPathError(
            "T2ICount asset root is not configured. Set {} or pass --asset-root."
            .format(ASSET_ROOT_ENV)
        )
    return None


@dataclass(frozen=True)
class AssetPaths:
    root: Path

    @classmethod
    def from_sources(cls, asset_root: Optional[PathLike] = None, required: bool = True):
        root = resolve_asset_root(asset_root, required=required)
        return cls(root) if root is not None else None

    @property
    def clip_dir(self) -> Path:
        return self.root / "pretrained" / "clip-vit-large-patch14"

    @property
    def sd_checkpoint(self) -> Path:
        return self.root / "pretrained" / "sd-v1-5" / "v1-5-pruned-emaonly.ckpt"

    @property
    def official_checkpoint(self) -> Path:
        return self.root / "checkpoints" / "official" / "best_model_paper.pth"

    def dataset_dir(self, dataset: str) -> Path:
        names = {
            "fsc147": "FSC147",
            "carpk": "CARPK",
            "idcia": "IDCIA",
        }
        try:
            name = names[dataset.casefold()]
        except KeyError:
            raise ValueError("Unknown dataset: {}".format(dataset))
        return self.root / "datasets" / name


def resolve_required_file(
    explicit: Optional[PathLike], default: Optional[Path], label: str
) -> Path:
    if explicit is None and default is None:
        raise AssetPathError(
            "{} is not configured. Set {}, or pass an explicit path."
            .format(label, ASSET_ROOT_ENV)
        )
    return require_file(explicit if explicit is not None else default, label)


def resolve_required_directory(
    explicit: Optional[PathLike], default: Optional[Path], label: str
) -> Path:
    if explicit is None and default is None:
        raise AssetPathError(
            "{} is not configured. Set {}, or pass an explicit path."
            .format(label, ASSET_ROOT_ENV)
        )
    return require_directory(explicit if explicit is not None else default, label)
