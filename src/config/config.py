"""Configuration loader and ProjectSettings dataclass.

Usage:
    from src.config import load_config, ProjectSettings
    cfg = load_config('config.yaml')
    settings = ProjectSettings.from_dict(cfg)
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


@dataclass
class ProjectSettings:
    project_name: str = "NLP Project"
    authors: list = None
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    processing_scripts: str = "data/processing_scripts"
    experiments: str = "experiments"
    models: str = "models"
    artifacts: str = "artifacts"
    text_col: str = "text"
    label_col: str = "label"
    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    seed: int = 42
    train_val_test: tuple = (0.8, 0.1, 0.1)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectSettings":
        project = d.get("project", {}) or {}
        paths = d.get("paths", {}) or {}
        preprocessing = d.get("preprocessing", {}) or {}
        training = d.get("training", {}) or {}

        return cls(
            project_name=project.get("name", cls.project_name),
            authors=project.get("authors", project.get("author", [])),
            data_raw=paths.get("data_raw", cls.data_raw),
            data_processed=paths.get("data_processed", cls.data_processed),
            processing_scripts=paths.get("processing_scripts", cls.processing_scripts),
            experiments=paths.get("experiments", cls.experiments),
            models=paths.get("models", cls.models),
            artifacts=paths.get("artifacts", cls.artifacts),
            text_col=preprocessing.get("text_col", cls.text_col),
            label_col=preprocessing.get("label_col", cls.label_col),
            lowercase=preprocessing.get("lowercase", cls.lowercase),
            remove_urls=preprocessing.get("remove_urls", cls.remove_urls),
            remove_mentions=preprocessing.get("remove_mentions", cls.remove_mentions),
            seed=training.get("seed", cls.seed),
            train_val_test=tuple(training.get("train_val_test", cls.train_val_test)),
        )

# run python -c "from src.config import load_config, ProjectSettings; cfg=load_config('config.yaml'); s=ProjectSettings.from_dict(cfg); print('Project:', s.project_name); print('Data raw:', s.data_raw)"