from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_grid_benchmark import (  # noqa: E402
    _annotate_config,
    _apply_variant_overrides,
    _standardize_common,
)
from grid_benchmark_common import (  # noqa: E402
    CAMPAIGN_SLUG,
    GENERATED_CONFIG_DIR,
    REPO_ROOT as COMMON_REPO_ROOT,
    VARIANT_SPECS,
    run_id_for,
)


METHODS = ["sivi", "uivi", "rsivi", "aisivi", "dsivi", "ksivi"]
METHOD_LABELS = {
    "sivi": "SIVI",
    "uivi": "UIVI",
    "rsivi": "RSIVI",
    "aisivi": "AISIVI",
    "dsivi": "DSIVI",
    "ksivi": "KSIVI",
}
METHOD_TO_VARIANT = {
    "sivi": "sivi",
    "uivi": "uivi",
    "rsivi": "rsivi",
    "aisivi": "aisivi",
    "dsivi": "dsivi_default",
    "ksivi": "ksivi_custom",
}
METHOD_TO_BASE_PREFIX = {
    "sivi": "sivi",
    "uivi": "uivi",
    "rsivi": "rsivi",
    "aisivi": "aisivi",
    "dsivi": "dsivi",
    "ksivi": "ksivi",
}

TOY_TARGETS = [
    "banana",
    "multimodal",
    "x_shaped",
    "student_uc",
    "8_gaussians",
    "Langevin_post",
]
LR_TARGETS = ["LRwaveform"]
BNN_TARGETS = [
    "Bnn_boston",
    "Bnn_concrete",
    "Bnn_power",
    "Bnn_protein",
    "Bnn_winered",
    "Bnn_yacht",
]
TARGETS = TOY_TARGETS + LR_TARGETS + BNN_TARGETS
TARGET_GROUPS = {
    "all": TARGETS,
    "toy": TOY_TARGETS,
    "LRwaveform": LR_TARGETS,
    "BNN": BNN_TARGETS,
}
TARGET_TO_GROUP = {
    **{target: "toy" for target in TOY_TARGETS},
    **{target: "LRwaveform" for target in LR_TARGETS},
    **{target: "BNN" for target in BNN_TARGETS},
}

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ConfigError(Exception):
    pass


def _relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _query_value(query: dict[str, list[str]], name: str, default: str | None = None) -> str | None:
    values = query.get(name)
    if not values:
        return default
    return values[0]


def _query_list(query: dict[str, list[str]], name: str, default: list[str]) -> list[str]:
    raw = _query_value(query, name)
    if raw is None or not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _validate_method(method: str) -> str:
    if method not in METHODS:
        raise ConfigError(f"Unknown method {method!r}.")
    return method


def _validate_target(target: str) -> str:
    if target not in TARGETS:
        raise ConfigError(f"Unknown target {target!r}.")
    return target


def _base_path(method: str, target: str) -> Path:
    return REPO_ROOT / "configs" / f"{METHOD_TO_BASE_PREFIX[method]}_{target}.yaml"


def _source_base_path_for_computed(method: str, target: str) -> Path:
    variant = METHOD_TO_VARIANT[method]
    source_method = VARIANT_SPECS[variant]["source_method"]
    return REPO_ROOT / "configs" / f"{source_method}_{target}.yaml"


def _checked_in_generated_path(method: str, target: str) -> Path:
    variant = METHOD_TO_VARIANT[method]
    run_id = run_id_for(target, variant, "on")
    return GENERATED_CONFIG_DIR / f"{run_id}.yaml"


def _read_yaml_config(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        return data, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _merge_nested_config(
    config: dict[str, Any],
    path_key: str,
    type_key: str,
    section_key: str,
    default_dir: str,
) -> tuple[dict[str, Any], str | None]:
    if path_key not in config:
        model_type = config.get(type_key)
        if not model_type:
            return config, None
        config[path_key] = f"configs/{default_dir}/{model_type}.yaml"

    nested_path = REPO_ROOT / str(config[path_key])
    nested_data, error = _read_yaml_config(nested_path)
    if error is not None:
        return config, f"{path_key}: {error}"
    if nested_data is None:
        return config, f"{path_key}: missing file {config[path_key]}"

    merged = OmegaConf.merge(
        {section_key: OmegaConf.create(nested_data)},
        OmegaConf.create(config),
    )
    return OmegaConf.to_container(merged, resolve=True), None


def _resolve_nested_configs(config: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    resolved = deepcopy(config)
    if "device" not in resolved:
        resolved["device"] = "cuda" if resolved.get("use_cuda", False) else "cpu"

    target_type = resolved.get("target_type")
    if target_type and "target_config_path" not in resolved:
        resolved["target_config_path"] = f"configs/targets/{target_type}.yaml"

    for args in [
        ("target_config_path", "target_type", "target", "targets"),
        ("vi_model_config_path", "vi_model_type", "vi_model", "vi_models"),
        (
            "reverse_model_config_path",
            "reverse_model_type",
            "reverse_model",
            "reverse_models",
        ),
    ]:
        resolved, error = _merge_nested_config(resolved, *args)
        if error is not None:
            return resolved, error

    reverse_model = resolved.get("reverse_model")
    vi_model = resolved.get("vi_model")
    if isinstance(reverse_model, dict) and isinstance(vi_model, dict):
        if "z_dim" in vi_model:
            reverse_model["z_dim"] = vi_model["z_dim"]
        if "epsilon_dim" in vi_model:
            reverse_model["epsilon_dim"] = vi_model["epsilon_dim"]

    return resolved, None


def _read_effective_config(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    config, error = _read_yaml_config(path)
    if error is not None or config is None:
        return config, error
    return _resolve_nested_configs(config)


def _raw_file(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return path.read_text(encoding="utf-8"), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _computed_config(method: str, target: str) -> tuple[dict[str, Any] | None, str | None]:
    source_path = _source_base_path_for_computed(method, target)
    config, error = _read_yaml_config(source_path)
    if error is not None:
        return None, error
    if config is None:
        return None, None

    try:
        variant = METHOD_TO_VARIANT[method]
        config = deepcopy(config)
        _standardize_common(config, target, variant, True)
        _apply_variant_overrides(config, target, variant)
        _annotate_config(config, run_id_for(target, variant, "on"), target, variant, "on")
        return _resolve_nested_configs(config)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def _yaml_text(data: dict[str, Any] | None) -> str | None:
    if data is None:
        return None
    return OmegaConf.to_yaml(OmegaConf.create(data), resolve=True)


def _config_payload(
    kind: str,
    method: str,
    target: str,
    path: Path,
    data: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    exists = path.exists() if kind != "computed" else data is not None
    return {
        "id": f"{kind}:{method}:{target}",
        "kind": kind,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "target": target,
        "target_group": TARGET_TO_GROUP[target],
        "variant": METHOD_TO_VARIANT[method],
        "path": _relpath(path),
        "display_path": _relpath(path),
        "exists": exists,
        "error": error,
        "data": data,
    }


def _config_for_source(kind: str, method: str, target: str) -> dict[str, Any]:
    if kind == "base":
        path = _base_path(method, target)
        data, error = _read_effective_config(path)
        return _config_payload(kind, method, target, path, data, error)
    if kind == "checked_in":
        path = _checked_in_generated_path(method, target)
        data, error = _read_effective_config(path)
        return _config_payload(kind, method, target, path, data, error)
    if kind == "computed":
        path = _source_base_path_for_computed(method, target)
        data, error = _computed_config(method, target)
        payload = _config_payload(kind, method, target, path, data, error)
        payload["display_path"] = f"computed from {_relpath(path)}"
        return payload
    raise ConfigError(f"Unknown source kind {kind!r}.")


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if isinstance(value, dict):
        if not value:
            rows[prefix or "<root>"] = {}
        for key in sorted(value):
            key_text = str(key)
            path = key_text if not prefix else f"{prefix}.{key_text}"
            rows.update(_flatten(value[key], path))
        return rows
    rows[prefix or "<root>"] = value
    return rows


def _value_signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _display_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(", ", ": "), default=str)


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "data"}


def _comparison_summary(configs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "config_count": len(configs),
        "row_count": len(rows),
        "changed_count": sum(1 for row in rows if row["changed"]),
        "missing_file_count": sum(1 for config in configs if config["data"] is None and config["error"] is None),
        "error_count": sum(1 for config in configs if config["error"] is not None),
    }


def _build_comparison(configs: list[dict[str, Any]]) -> dict[str, Any]:
    flattened_by_config: list[dict[str, Any]] = []
    all_paths: set[str] = set()

    for config in configs:
        flat = {} if config["data"] is None else _flatten(config["data"])
        flattened_by_config.append(flat)
        all_paths.update(flat.keys())

    rows = []
    for path in sorted(all_paths):
        cells = []
        signatures = []
        for config, flat in zip(configs, flattened_by_config):
            if config["data"] is None:
                status = "missing_file" if config["error"] is None else "error"
                value = None
                signature = f"__{status}__"
            elif path not in flat:
                status = "missing_key"
                value = None
                signature = "__missing_key__"
            else:
                status = "present"
                value = flat[path]
                signature = _value_signature(value)
            signatures.append(signature)
            cells.append(
                {
                    "config_id": config["id"],
                    "status": status,
                    "value": value,
                    "display": "" if status != "present" else _display_value(value),
                }
            )
        rows.append(
            {
                "path": path,
                "depth": 0 if path == "<root>" else path.count("."),
                "key": path.rsplit(".", 1)[-1],
                "changed": len(set(signatures)) > 1,
                "cells": cells,
            }
        )

    return {
        "configs": [_public_config(config) for config in configs],
        "rows": rows,
        "summary": _comparison_summary(configs, rows),
    }


def _metadata() -> dict[str, Any]:
    methods = [
        {
            "name": method,
            "label": METHOD_LABELS[method],
            "variant": METHOD_TO_VARIANT[method],
            "base_prefix": METHOD_TO_BASE_PREFIX[method],
        }
        for method in METHODS
    ]
    targets = [
        {
            "name": target,
            "label": target,
            "group": TARGET_TO_GROUP[target],
        }
        for target in TARGETS
    ]

    availability: dict[str, Any] = {"base": {}, "checked_in": {}, "computed": {}}
    for method in METHODS:
        availability["base"][method] = {}
        availability["checked_in"][method] = {}
        availability["computed"][method] = {}
        for target in TARGETS:
            base_path = _base_path(method, target)
            checked_path = _checked_in_generated_path(method, target)
            source_path = _source_base_path_for_computed(method, target)
            availability["base"][method][target] = {
                "exists": base_path.exists(),
                "path": _relpath(base_path),
            }
            availability["checked_in"][method][target] = {
                "exists": checked_path.exists(),
                "path": _relpath(checked_path),
            }
            availability["computed"][method][target] = {
                "exists": source_path.exists(),
                "path": f"computed from {_relpath(source_path)}",
            }

    return {
        "campaign_slug": CAMPAIGN_SLUG,
        "repo_root": str(COMMON_REPO_ROOT),
        "methods": methods,
        "targets": targets,
        "target_groups": TARGET_GROUPS,
        "availability": availability,
        "defaults": {
            "method": "sivi",
            "target": "banana",
            "group": "all",
            "generated_source": "both",
        },
    }


def _compare(query: dict[str, list[str]]) -> dict[str, Any]:
    mode = _query_value(query, "mode", "generated_vs_base")
    if mode == "generated_vs_base":
        method = _validate_method(_query_value(query, "method", "sivi") or "sivi")
        target = _validate_target(_query_value(query, "target", "banana") or "banana")
        source = _query_value(query, "source", "both") or "both"
        configs = [_config_for_source("base", method, target)]
        if source in {"checked_in", "both"}:
            configs.append(_config_for_source("checked_in", method, target))
        if source in {"computed", "both"}:
            configs.append(_config_for_source("computed", method, target))
        if source not in {"checked_in", "computed", "both"}:
            raise ConfigError(f"Unknown generated source {source!r}.")
        payload = _build_comparison(configs)
        payload["mode"] = mode
        payload["title"] = f"{method} on {target}: base vs generated"
        return payload

    if mode == "methods_for_target":
        target = _validate_target(_query_value(query, "target", "banana") or "banana")
        methods = [_validate_method(method) for method in _query_list(query, "methods", METHODS)]
        configs = [_config_for_source("base", method, target) for method in methods]
        payload = _build_comparison(configs)
        payload["mode"] = mode
        payload["title"] = f"Base methods on {target}"
        return payload

    if mode == "targets_for_method":
        method = _validate_method(_query_value(query, "method", "sivi") or "sivi")
        targets = [_validate_target(target) for target in _query_list(query, "targets", TARGETS)]
        configs = [_config_for_source("base", method, target) for target in targets]
        payload = _build_comparison(configs)
        payload["mode"] = mode
        payload["title"] = f"Base targets for {method}"
        return payload

    raise ConfigError(f"Unknown compare mode {mode!r}.")


def _raw(query: dict[str, list[str]]) -> dict[str, Any]:
    method = _validate_method(_query_value(query, "method", "sivi") or "sivi")
    target = _validate_target(_query_value(query, "target", "banana") or "banana")
    kind = _query_value(query, "kind", "base") or "base"

    if kind == "base":
        path = _base_path(method, target)
        data, error = _read_effective_config(path)
        text = _yaml_text(data)
        exists = data is not None
        display_path = f"resolved from {_relpath(path)}"
    elif kind == "checked_in":
        path = _checked_in_generated_path(method, target)
        data, error = _read_effective_config(path)
        text = _yaml_text(data)
        exists = data is not None
        display_path = f"resolved from {_relpath(path)}"
    elif kind == "computed":
        path = _source_base_path_for_computed(method, target)
        data, error = _computed_config(method, target)
        text = _yaml_text(data)
        exists = data is not None
        display_path = f"computed from {_relpath(path)}"
    else:
        raise ConfigError(f"Unknown raw kind {kind!r}.")

    return {
        "kind": kind,
        "method": method,
        "target": target,
        "path": _relpath(path),
        "display_path": display_path,
        "exists": exists,
        "error": error,
        "text": text,
    }


def _snapshot() -> dict[str, Any]:
    roots = [
        REPO_ROOT / "configs",
        GENERATED_CONFIG_DIR,
        STATIC_DIR,
        Path(__file__),
        REPO_ROOT / "scripts" / "config_review_server.py",
    ]
    files: dict[str, Any] = {}
    for root in roots:
        if root.is_file():
            candidates = [root]
        elif root.exists():
            candidates = sorted(path for path in root.rglob("*") if path.is_file())
        else:
            candidates = []
        for path in candidates:
            if path.suffix not in {".yaml", ".yml", ".py", ".html", ".css", ".js"}:
                continue
            stat = path.stat()
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            files[_relpath(path)] = {
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
                "sha1": digest,
            }
    snapshot_hash = hashlib.sha1(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"hash": snapshot_hash, "files": files}


class ConfigReviewHandler(BaseHTTPRequestHandler):
    server_version = "ConfigReviewHTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_file(STATIC_DIR / "index.html")
            elif parsed.path.startswith("/static/"):
                self._send_file(STATIC_DIR / parsed.path.removeprefix("/static/"))
            elif parsed.path == "/api/metadata":
                self._send_json(_metadata())
            elif parsed.path == "/api/compare":
                self._send_json(_compare(query))
            elif parsed.path == "/api/raw":
                self._send_json(_raw(query))
            elif parsed.path == "/api/snapshot":
                self._send_json(_snapshot())
            else:
                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown path {parsed.path}")
        except ConfigError as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        message = "%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args)
        sys.stderr.write(message)

    def _send_json(self, payload: Any) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_error(HTTPStatus.FORBIDDEN, "Static path is outside the reviewer directory.")
            return
        if not resolved.exists() or not resolved.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, f"Static file not found: {path.name}")
            return
        body = resolved.read_bytes()
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        if resolved.suffix in {".html", ".css", ".js"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": message}, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the read-only config reviewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), ConfigReviewHandler)
    except OSError as exc:
        parser.error(f"Could not bind {args.host}:{args.port}: {exc}")
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving config reviewer at {url}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping config reviewer.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
