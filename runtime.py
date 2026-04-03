"""
runtime.py - Shared startup/runtime helpers for CLI entrypoints.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RECOMMENDED_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
INSTALL_HINT = f"{RECOMMENDED_PYTHON} -m pip install -r requirements.txt"
LIGHTGBM_MACOS_HINT = (
    "Optional LightGBM setup on macOS: brew install libomp "
    f"and then run {RECOMMENDED_PYTHON} -m pip install --force-reinstall lightgbm"
)


def recommended_command(script_name: str, *args: str) -> str:
    parts = [str(RECOMMENDED_PYTHON), script_name, *args]
    return " ".join(parts)


def missing_modules(required_modules: list[str] | tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def log_runtime_environment(
    logger: logging.Logger,
    app_name: str,
    required_modules: list[str] | tuple[str, ...] = (),
) -> list[str]:
    logger.info("%s interpreter: %s", app_name, sys.executable)
    if Path(sys.executable).resolve() != RECOMMENDED_PYTHON.resolve():
        logger.warning(
            "%s is not running from the project virtualenv. Recommended command: %s",
            app_name,
            recommended_command(app_name),
        )

    missing = missing_modules(required_modules)
    if missing:
        logger.warning(
            "Missing Python modules in this interpreter: %s. Install them with: %s",
            ", ".join(missing),
            INSTALL_HINT,
        )
    return missing


def describe_optional_lightgbm_failure(exc: Exception) -> str:
    return (
        "LightGBM is unavailable, so the pipeline will continue without it. "
        f"Reason: {exc}. {LIGHTGBM_MACOS_HINT}"
    )
