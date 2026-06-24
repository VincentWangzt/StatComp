from __future__ import annotations

import csv
import functools
import inspect
import json
import logging
import math
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

from dotenv import load_dotenv


LOGGER = logging.getLogger("sivi")
F = TypeVar("F", bound=Callable[..., Any])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_wandb_environment() -> None:
    """Load repository-local W&B settings without replacing shell values."""
    load_dotenv(_repo_root() / ".env", override=False)


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        value = value.item()
    converted = float(value)
    if not math.isfinite(converted):
        return converted
    return converted


def _config_dict(config: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        container = OmegaConf.to_container(config, resolve=True)
        converted = dict(container) if isinstance(container, Mapping) else {}
    except Exception:
        if isinstance(config, Mapping):
            converted = dict(config)
        else:
            converted = {}
    return _remove_secrets(converted)


def _remove_secrets(value: Any) -> Any:
    """Remove credential-shaped keys before configuration leaves the process."""
    if isinstance(value, Mapping):
        cleaned = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {"api_key", "wandb_api_key"}:
                continue
            cleaned[str(key)] = _remove_secrets(item)
        return cleaned
    if isinstance(value, list):
        return [_remove_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [_remove_secrets(item) for item in value]
    return value


class ExperimentLogger:
    """Central scalar, timing, file, image, CSV, and W&B logger for one run."""

    CSV_PREFIXES = ("metric/", "time_avg/")

    def __init__(
        self,
        *,
        save_path: str | Path,
        config: Any,
        runner_name: str,
        target_type: str,
        vi_model_type: str,
        seed: int,
        time_avg_window: int = 100,
    ) -> None:
        load_wandb_environment()
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.current_step = 0
        self.time_avg_window = max(1, int(time_avg_window))
        self._time_history: dict[str, deque[tuple[int, float]]] = defaultdict(deque)
        self._lock = threading.RLock()
        self._finished = False

        self.metrics_path = self.save_path / "metrics.csv"
        new_file = not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
        self._metrics_fh = self.metrics_path.open(
            "a", encoding="utf-8", newline="", buffering=1
        )
        self._metrics_writer = csv.DictWriter(
            self._metrics_fh,
            fieldnames=["tag", "step", "wall_time", "value"],
        )
        if new_file:
            self._metrics_writer.writeheader()
            self._metrics_fh.flush()

        config_dict = _config_dict(config)
        tracking = config_dict.get("tracking", {})
        if not isinstance(tracking, Mapping):
            tracking = {}
        self.enabled = bool(tracking.get("enabled", True))
        campaign = str(tracking.get("campaign") or "").strip()
        timestamp = self.save_path.name
        base_name = f"{runner_name}-{target_type}-seed{seed}-{timestamp}"
        self.run_name = f"{campaign}-{base_name}" if campaign else base_name
        self.tags = [
            f"method:{runner_name}",
            f"target:{target_type}",
            f"vi:{vi_model_type}",
        ]
        extra_tags = tracking.get("tags", [])
        if isinstance(extra_tags, Sequence) and not isinstance(extra_tags, str):
            self.tags.extend(str(tag) for tag in extra_tags)
        if campaign:
            self.tags.append(f"campaign:{campaign}")

        self.run: Any | None = None
        self.wandb_mode = "disabled"
        self.wandb_run_id = ""
        self.wandb_url = ""
        self.wandb_path = ""
        if self.enabled:
            self._init_wandb(
                project=str(
                    tracking.get("project")
                    or os.getenv("WANDB_PROJECT")
                    or "StatComp"
                ),
                entity=str(
                    tracking.get("entity") or os.getenv("WANDB_ENTITY") or ""
                )
                or None,
                group=campaign or None,
                config=config_dict,
            )
        self._write_metadata()

    def _init_wandb(
        self,
        *,
        project: str,
        entity: str | None,
        group: str | None,
        config: dict[str, Any],
    ) -> None:
        import wandb

        mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
        init_kwargs = {
            "project": project,
            "entity": entity,
            "group": group,
            "name": self.run_name,
            "tags": self.tags,
            "config": config,
            "dir": str(self.save_path),
            "job_type": "train",
            "mode": mode,
            "reinit": "finish_previous",
            "settings": wandb.Settings(init_timeout=30),
        }
        try:
            self.run = wandb.init(**init_kwargs)
            self.wandb_mode = mode
        except Exception as exc:
            if mode == "offline":
                LOGGER.warning("Failed to initialize offline W&B logging: %s", exc)
                self.run = None
                self.wandb_mode = "disabled"
                return
            LOGGER.warning(
                "Online W&B initialization failed; falling back to offline mode: %s",
                exc,
            )
            try:
                wandb.finish(exit_code=1, quiet=True)
            except Exception:
                pass
            init_kwargs["mode"] = "offline"
            try:
                self.run = wandb.init(**init_kwargs)
                self.wandb_mode = "offline"
            except Exception as offline_exc:
                LOGGER.warning(
                    "Offline W&B initialization also failed; continuing with local CSV/logs: %s",
                    offline_exc,
                )
                self.run = None
                self.wandb_mode = "disabled"

        if self.run is not None:
            self.wandb_run_id = str(self.run.id or "")
            self.wandb_url = str(self.run.url or "")
            run_path = self.run.path
            self.wandb_path = (
                run_path
                if isinstance(run_path, str)
                else "/".join(str(part) for part in run_path)
            )
            self.run.define_metric("epoch")
            self.run.define_metric("pretrain_step")
            self.run.define_metric("warmup_epoch")
            for pattern in (
                "train/*",
                "metric/*",
                "diagnostic/*",
                "time/*",
                "time_avg/*",
            ):
                self.run.define_metric(pattern, step_metric="epoch")
            self.run.define_metric("pretrain/*", step_metric="pretrain_step")
            self.run.define_metric("warmup/*", step_metric="warmup_epoch")

    def _write_metadata(self) -> None:
        metadata = {
            "run_name": self.run_name,
            "run_id": self.wandb_run_id,
            "run_url": self.wandb_url,
            "run_path": self.wandb_path,
            "mode": self.wandb_mode,
            "metrics_path": str(self.metrics_path),
            "tags": self.tags,
        }
        (self.save_path / "wandb_run.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
        )

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def _step_key(self, tags: Sequence[str]) -> str:
        if tags and all(tag.startswith("pretrain/") for tag in tags):
            return "pretrain_step"
        if tags and all(tag.startswith("warmup/") for tag in tags):
            return "warmup_epoch"
        return "epoch"

    def log_scalars(
        self,
        values: Mapping[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        if not values:
            return
        numeric = {str(tag): _as_float(value) for tag, value in values.items()}
        resolved_step = self.current_step if step is None else int(step)
        wall_time = time.time()
        with self._lock:
            for tag, value in numeric.items():
                if tag.startswith(self.CSV_PREFIXES):
                    self._metrics_writer.writerow(
                        {
                            "tag": tag,
                            "step": resolved_step,
                            "wall_time": wall_time,
                            "value": value,
                        }
                    )
            self._metrics_fh.flush()
            if self.run is not None:
                step_key = self._step_key(list(numeric))
                self.run.log({step_key: resolved_step, **numeric})

    def record_timing(
        self,
        name: str,
        elapsed: float,
        *,
        step: int | None = None,
    ) -> None:
        resolved_step = self.current_step if step is None else int(step)
        history = self._time_history[name]
        history.append((resolved_step, float(elapsed)))
        cutoff = resolved_step - self.time_avg_window
        while history and history[0][0] <= cutoff:
            history.popleft()
        average = sum(value for _, value in history) / len(history)
        self.log_scalars(
            {f"time/{name}": elapsed, f"time_avg/{name}": average},
            step=resolved_step,
        )

    @contextmanager
    def timer(self, name: str, *, step: int | None = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_timing(name, time.perf_counter() - start, step=step)

    def log_image(self, tag: str, path: str | Path, *, step: int) -> None:
        if self.run is None:
            return
        import wandb

        self.run.log({"epoch": int(step), tag: wandb.Image(str(path))})

    def log_text(self, tag: str, text: str) -> None:
        if self.run is not None:
            self.run.log({tag: text})

    def update_config(self, config: Any) -> None:
        """Refresh W&B config after runner-specific config composition."""
        if self.run is not None:
            self.run.config.update(_config_dict(config), allow_val_change=True)

    def finish(self, *, exit_code: int = 0) -> None:
        if self._finished:
            return
        self._finished = True
        with self._lock:
            self._metrics_fh.flush()
            self._metrics_fh.close()
        if self.run is not None:
            for filename in ("full_config.yaml", "run.log", "wandb_run.json"):
                path = self.save_path / filename
                if path.exists():
                    self.run.save(str(path), base_path=str(self.save_path), policy="end")
            self.run.finish(exit_code=exit_code)
        self._write_metadata()


def _logger_from_call(args: tuple[Any, ...]) -> ExperimentLogger | None:
    if not args:
        return None
    owner = args[0]
    if isinstance(owner, ExperimentLogger):
        return owner
    logger = getattr(owner, "experiment_logger", None)
    return logger if isinstance(logger, ExperimentLogger) else None


def _step_from_call(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    logger: ExperimentLogger,
) -> int:
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        for name in ("epoch", "step", "epoch_outer"):
            if name in bound.arguments:
                return int(bound.arguments[name])
    except (TypeError, ValueError):
        pass
    return logger.current_step


def metric(
    *names: str,
    prefix: str | None = None,
) -> Callable[[F], F]:
    """Log a decorated function's scalar, tuple, or mapping return value."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            logger = _logger_from_call(args)
            if logger is None:
                return result
            if isinstance(result, Mapping):
                values = dict(result)
            elif isinstance(result, tuple):
                if len(names) != len(result):
                    raise ValueError(
                        f"@metric on {func.__qualname__} requires one name per tuple value"
                    )
                values = dict(zip(names, result))
            else:
                if len(names) != 1:
                    raise ValueError(
                        f"@metric on {func.__qualname__} requires exactly one name"
                    )
                values = {names[0]: result}
            if prefix:
                values = {
                    key if str(key).startswith(f"{prefix}/") else f"{prefix}/{key}": value
                    for key, value in values.items()
                }
            step = _step_from_call(func, args, kwargs, logger)
            logger.log_scalars(values, step=step)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def timer(name: str | None = None) -> Callable[[F], F]:
    """Time a decorated method and route raw/rolling values through its logger."""

    def decorator(func: F) -> F:
        timer_name = name or func.__name__.removeprefix("eval_")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = _logger_from_call(args)
            if logger is None:
                return func(*args, **kwargs)
            step = _step_from_call(func, args, kwargs, logger)
            with logger.timer(timer_name, step=step):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
