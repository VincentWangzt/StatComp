# Config Reviewer

Read-only local UI for reviewing base YAML configs against the current grid
benchmark generation policy.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\config_review_server.py --port 8765
```

Then open:

```text
http://127.0.0.1:8765/
```

The server reloads YAML data on every request and the browser polls for file
changes, so edits made in an editor appear without restarting the server.

Comparisons use resolved effective configs. The reviewer expands the same nested
config layers the runners load at runtime:

- `target_config_path` into `target`
- `vi_model_config_path` into `vi_model`
- `reverse_model_config_path` into `reverse_model`

If a nested path is omitted, the reviewer uses the runner-style default path
derived from `target_type`, `vi_model_type`, or `reverse_model_type`.

The reviewer intentionally does not edit configs or apply standardization. It
compares:

- base configs under `configs/<method>_<target>.yaml`
- checked-in generated configs under `configs/generated/grid_benchmark_20260330`
- in-memory computed generated-policy configs using the current generator helper
  functions

Canonical method mapping:

- `sivi -> sivi`
- `uivi -> uivi`
- `rsivi -> rsivi`
- `aisivi -> aisivi`
- `dsivi -> dsivi_default`
- `ksivi -> ksivi_custom`
