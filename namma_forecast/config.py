"""Configuration loading and project paths."""
from __future__ import annotations
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = path.parent
    return cfg


def resolve(cfg: dict, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def load_sources(cfg: dict) -> dict:
    with open(resolve(cfg, cfg["paths"]["sources"]), encoding="utf-8") as f:
        return json.load(f)


def save_sources(cfg: dict, sources: dict) -> None:
    with open(resolve(cfg, cfg["paths"]["sources"]), "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2)
