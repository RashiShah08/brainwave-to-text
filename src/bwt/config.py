"""Project configuration.

Precedence, lowest to highest: dataclass defaults -> ``configs/default.yaml``
-> environment variables (``BWT_*``) -> explicit CLI flags. Nothing needs a
config file to run; the file exists so that a deployment can pin choices
without editing code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from bwt.data.epochs import DEFAULT_TMAX, DEFAULT_TMIN
from bwt.data.physionet import DEFAULT_TASK
from bwt.paths import configs_dir
from bwt.pipelines import DEFAULT_PIPELINE


@dataclass
class DataConfig:
    task: str = DEFAULT_TASK
    tmin: float = DEFAULT_TMIN
    tmax: float = DEFAULT_TMAX
    n_jobs: int = 4
    use_cache: bool = True


@dataclass
class TrainConfig:
    pipeline: str = DEFAULT_PIPELINE
    random_state: int = 42
    cv_splits: int = 5
    #: Protocols to run during `bwt train`. Both are reported in the model card.
    protocols: tuple[str, ...] = ("within_subject", "cross_subject")
    permutations: int = 0  # 0 disables the permutation test (it is slow)


@dataclass
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    #: Hard cap on uploads. A 14-run EEGMMIDB subject is ~30 MB; 64 MB is
    #: generous for a single file and small enough to bound memory.
    max_upload_mb: int = 64
    model: str | None = None  # artifact directory name; None = most recent
    #: Extra epochs beyond the model's window are ignored past this many, to
    #: bound work per request.
    max_epochs_per_request: int = 256
    strict_versions: bool = False
    #: Server worker threads. Paced live replays may occupy all but one of
    #: them, so a slow replay can never lock everyone else out.
    threads: int = 4
    #: Sites that may show these pages in a frame (CSP ``frame-ancestors``),
    #: such as a portfolio's live preview. Every page is public and there are no
    #: accounts, so framing exposes nothing; ``'none'`` blocks it everywhere.
    frame_ancestors: str = "https: http://localhost:5173"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)

    # -- construction ----------------------------------------------------- #

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        config = cls()
        path = Path(path) if path else configs_dir() / "default.yaml"
        if path.is_file():
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            config = config.merged(payload)
        return config.with_env()

    def merged(self, payload: dict[str, Any]) -> Config:
        current = asdict(self)
        for section, values in (payload or {}).items():
            if section in current and isinstance(values, dict):
                current[section].update(values)
        return self._from_dict(current)

    @classmethod
    def _from_dict(cls, payload: dict[str, Any]) -> Config:
        return cls(
            data=DataConfig(**payload.get("data", {})),
            train=TrainConfig(**payload.get("train", {})),
            serve=ServeConfig(**payload.get("serve", {})),
        )

    def with_env(self) -> Config:
        """Apply ``BWT_<SECTION>_<FIELD>`` environment overrides."""
        current = asdict(self)
        for section_name, section in (
            ("data", DataConfig), ("train", TrainConfig), ("serve", ServeConfig)
        ):
            for spec in fields(section):
                env_key = f"BWT_{section_name.upper()}_{spec.name.upper()}"
                if env_key not in os.environ:
                    continue
                raw = os.environ[env_key]
                current[section_name][spec.name] = _coerce(raw, spec.type)
        return self._from_dict(current)

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(raw: str, annotation: Any) -> Any:
    text = str(annotation)
    if text.startswith(("tuple", "list")):
        # A comma-separated value. Stored as the raw string, anything iterating
        # the setting would walk its characters.
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    if "bool" in text:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if "int" in text:
        return int(raw)
    if "float" in text:
        return float(raw)
    return raw


__all__ = ["Config", "DataConfig", "ServeConfig", "TrainConfig"]
