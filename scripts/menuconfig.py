#!/usr/bin/env python3
"""
GUI or terminal menu to create or edit the phase0 YAML config.
Run via: make menuconfig   or   python scripts/menuconfig.py [CONFIG_PATH] [--tui]
  Default: GUI (tkinter). Use --tui for terminal menu.
"""

import sys
from pathlib import Path

# Run from repo root so compressor is importable
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import yaml
from compressor.config import REQUIRED_KEYS, load_and_validate, _get

# Key path -> type for prompts (str, int, float, bool)
KEY_TYPE = {
    "seed": "int",
    "device": "str",
    "output_dir": "str",
    "log_level": "str",
    "target.model_id": "str",
    "target.sglang.base_url": "str",
    "target.sglang.timeout_s": "float",
    "target.sglang.max_retries": "int",
    "target.temperature": "float",
    "target.top_p": "float",
    "target.max_new_tokens": "int",
    "target.seed": "int",
    "target.prompt_template": "str",
    "reflector.model_id": "str",
    "reflector.api": "str",
    "reflector.temperature": "float",
    "reflector.top_p": "float",
    "reflector.max_new_tokens": "int",
    "reflector.seed": "int",
    "reflector.compress_prompt_template": "str",
    "reflector.api_base": "str",
    "reflector.api_key_env": "str",
    "data.financebench_path": "str",
    "data.split_ratios.train": "float",
    "data.split_ratios.val": "float",
    "data.split_ratios.test": "float",
    "phase0.verbose_num_samples": "int",
    "phase0.k": "int",
    "phase0.output_path": "str",
    "phase0.filters.drop_special_tokens": "bool",
    "phase0.filters.drop_whitespace_only": "bool",
    "phase0.filters.drop_digit_only": "bool",
    "phase0.correctness.metric": "str",
    "phase0.correctness.threshold": "float",
    "phase0.correctness.tolerance": "float",
    "phase0.correctness.evaluator_model": "str",
    "phase0.correctness.evaluator_api_base": "str",
    "phase0.correctness.evaluator_api_key_env": "str",
    "phase0.correctness.qualitative_forgiving": "bool",
    "phase0.correctness.evaluator_temperature": "float",
    "phase0.correctness.evaluator_top_p": "float",
    "phase0.correctness.evaluator_max_new_tokens": "int",
    "phase0.correctness.evaluator_seed": "int",
    "phase0.target_concurrency": "int",
    "phase0.reflector_concurrency": "int",
    "phase0.judge_concurrency": "int",
}

# Section name -> list of key paths (order preserved)
SECTIONS = [
    ("General", ["seed", "device", "output_dir", "log_level"]),
    ("Target", [
        "target.model_id", "target.sglang.base_url", "target.sglang.timeout_s",
        "target.sglang.max_retries", "target.temperature", "target.top_p",
        "target.max_new_tokens", "target.seed", "target.prompt_template",
    ]),
    ("Reflector", [
        "reflector.model_id", "reflector.api", "reflector.temperature", "reflector.top_p",
        "reflector.max_new_tokens", "reflector.seed", "reflector.compress_prompt_template",
        "reflector.api_base", "reflector.api_key_env",
    ]),
    ("Data", [
        "data.financebench_path", "data.split_ratios.train", "data.split_ratios.val", "data.split_ratios.test",
    ]),
    ("Phase0", [
        "phase0.verbose_num_samples", "phase0.k", "phase0.output_path",
        "phase0.filters.drop_special_tokens", "phase0.filters.drop_whitespace_only", "phase0.filters.drop_digit_only",
        "phase0.target_concurrency", "phase0.reflector_concurrency", "phase0.judge_concurrency",
    ]),
    ("Phase0 Correctness", [
        "phase0.correctness.metric", "phase0.correctness.threshold", "phase0.correctness.tolerance",
        "phase0.correctness.evaluator_model", "phase0.correctness.evaluator_api_base", "phase0.correctness.evaluator_api_key_env",
        "phase0.correctness.qualitative_forgiving",
        "phase0.correctness.evaluator_temperature", "phase0.correctness.evaluator_top_p",
        "phase0.correctness.evaluator_max_new_tokens", "phase0.correctness.evaluator_seed",
    ]),
]


def _set_by_path(cfg: dict, path: str, value) -> None:
    parts = path.split(".")
    cur = cfg
    for i, k in enumerate(parts[:-1]):
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def _default_for_typ(typ: str):
    if typ == "str":
        return ""
    if typ == "int":
        return 0
    if typ == "float":
        return 0.0
    if typ == "bool":
        return False
    return ""


def _parse_value(raw: str, typ: str):
    raw = raw.strip()
    if typ == "str":
        return raw
    if typ == "bool":
        if raw.lower() in ("1", "true", "yes", "on", "y"):
            return True
        if raw.lower() in ("0", "false", "no", "off", "n", ""):
            return False
        raise ValueError("Enter true/false, 1/0, or y/n")
    if typ == "int":
        return int(raw)
    if typ == "float":
        return float(raw)
    return raw


def build_empty_config() -> dict:
    cfg = {}
    for path in REQUIRED_KEYS:
        typ = KEY_TYPE.get(path, "str")
        _set_by_path(cfg, path, _default_for_typ(typ))
    return cfg


def load_config(path: Path) -> dict:
    base = build_empty_config()
    if not path.exists():
        return base
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    for path in REQUIRED_KEYS:
        v = _get(data, path)
        if v is None:
            continue
        if isinstance(v, (str, list, dict)) and len(v) == 0:
            continue
        if isinstance(v, float) and (v != v or v == float("inf")):
            continue
        _set_by_path(base, path, v)
    return base


def save_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def format_display_value(path: str, value) -> str:
    typ = KEY_TYPE.get(path, "str")
    if typ == "bool":
        return "true" if value else "false"
    s = str(value)
    if len(s) > 50:
        return s[:47] + "..."
    return s


def _run_gui(config_path: Path) -> None:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext

    cfg = load_config(config_path)
    config_path = config_path.resolve()

    root = tk.Tk()
    root.title("Phase 0 Config — menuconfig")
    root.minsize(700, 500)
    root.geometry("900x600")

    # Path label
    path_frame = ttk.Frame(root, padding=4)
    path_frame.pack(fill=tk.X)
    ttk.Label(path_frame, text="Config:", font=("", 9)).pack(side=tk.LEFT)
    ttk.Label(path_frame, text=str(config_path), font=("", 9), foreground="gray").pack(side=tk.LEFT, padx=4)

    # Paned: left = sections, right = form
    paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # Left: section list
    left = ttk.Frame(paned)
    paned.add(left, weight=0)
    ttk.Label(left, text="Section").pack(anchor=tk.W)
    listbox = tk.Listbox(left, height=20, width=22, font=("", 11), selectmode=tk.SINGLE, exportselection=False)
    listbox.pack(fill=tk.BOTH, expand=True, pady=4)
    for name, _ in SECTIONS:
        listbox.insert(tk.END, name)
    listbox.selection_set(0)

    # Right: scrollable form
    right = ttk.Frame(paned)
    paned.add(right, weight=1)
    form_container = ttk.Frame(right)
    form_container.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(form_container)
    scrollbar = ttk.Scrollbar(form_container)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=canvas.yview)
    canvas.config(yscrollcommand=scrollbar.set)
    form_inner = ttk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=form_inner, anchor=tk.NW)

    def _on_frame_configure(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)

    form_inner.bind("<Configure>", _on_frame_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    # Widgets per path: path -> (widget or var, typ)
    widgets: dict = {}

    def build_form(section_index: int) -> None:
        for w in form_inner.winfo_children():
            w.destroy()
        widgets.clear()
        _, keys = SECTIONS[section_index]
        for path in keys:
            typ = KEY_TYPE.get(path, "str")
            val = _get(cfg, path)
            row = ttk.Frame(form_inner)
            row.pack(fill=tk.X, pady=2)
            label = path  # use full path as label
            ttk.Label(row, text=label, width=42, anchor=tk.W).pack(side=tk.LEFT, padx=(0, 8))
            if typ == "bool":
                var = tk.BooleanVar(value=bool(val))
                w = ttk.Checkbutton(row, variable=var)
                w.pack(side=tk.LEFT)
                widgets[path] = (var, typ)
            else:
                sval = str(val) if val is not None else ""
                if "template" in path.lower() or "prompt" in path.lower():
                    e = scrolledtext.ScrolledText(row, height=3, width=40, wrap=tk.WORD, font=("", 10))
                    e.insert("1.0", sval)
                    e.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    widgets[path] = (e, typ)
                else:
                    e = ttk.Entry(row, width=40)
                    e.insert(0, sval)
                    e.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    widgets[path] = (e, typ)

    def on_select(_event=None) -> None:
        sel = listbox.curselection()
        if sel:
            flush_to_config()  # save current section edits before switching
            build_form(sel[0])

    def flush_to_config() -> None:
        for path, (w, typ) in widgets.items():
            try:
                if typ == "bool":
                    val = w.get()
                elif isinstance(w, tk.Text):
                    val = w.get("1.0", tk.END).strip()
                else:
                    val = w.get().strip()
                if typ != "bool" and val == "":
                    continue
                if typ != "str":
                    val = _parse_value(str(val), typ)
                _set_by_path(cfg, path, val)
            except Exception:
                pass

    def on_save() -> None:
        flush_to_config()
        try:
            save_config(cfg, config_path)
            load_and_validate(str(config_path))
            messagebox.showinfo("Saved", f"Config saved and validated:\n{config_path}")
            root.destroy()
        except Exception as e:
            if messagebox.askyesno("Validation failed", f"{e}\n\nSave anyway?"):
                save_config(cfg, config_path)
                messagebox.showinfo("Saved", f"Saved (validation failed):\n{config_path}")
                root.destroy()

    def on_cancel() -> None:
        root.destroy()

    listbox.bind("<<ListboxSelect>>", on_select)
    build_form(0)

    btn_frame = ttk.Frame(root, padding=4)
    btn_frame.pack(fill=tk.X)
    ttk.Button(btn_frame, text="Save", command=on_save).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=2)

    root.mainloop()


def run_menu(config_path: Path) -> None:
    cfg = load_config(config_path)

    while True:
        print()
        print("=== Phase 0 Config ===")
        print(f"  Config file: {config_path}")
        print()
        for i, (name, _) in enumerate(SECTIONS, 1):
            print(f"  {i}. {name}")
        print(f"  {len(SECTIONS) + 1}. Save and exit")
        print(f"  {len(SECTIONS) + 2}. Exit without saving")
        try:
            choice = input("\nSelect option (number): ").strip()
            n = int(choice)
        except (ValueError, EOFError):
            continue
        if n == len(SECTIONS) + 1:
            try:
                save_config(cfg, config_path)
                # Validate after save
                load_and_validate(str(config_path))
                print(f"Saved and validated: {config_path}")
            except Exception as e:
                print(f"Error: {e}")
                if input("Save anyway? [y/N]: ").strip().lower() != "y":
                    continue
                save_config(cfg, config_path)
                print("Saved (validation failed).")
            return
        if n == len(SECTIONS) + 2:
            print("Exited without saving.")
            return
        if 1 <= n <= len(SECTIONS):
            section_name, keys = SECTIONS[n - 1]
            while True:
                print()
                print(f"--- {section_name} ---")
                for i, path in enumerate(keys, 1):
                    val = _get(cfg, path)
                    disp = format_display_value(path, val)
                    print(f"  {i}. {path} = {disp}")
                print(f"  {len(keys) + 1}. Back")
                try:
                    sub = input("\nSelect key to edit (number): ").strip()
                    sn = int(sub)
                except (ValueError, EOFError):
                    continue
                if sn == len(keys) + 1:
                    break
                if 1 <= sn <= len(keys):
                    path = keys[sn - 1]
                    typ = KEY_TYPE.get(path, "str")
                    current = _get(cfg, path)
                    prompt = f"  {path} ({typ})"
                    if typ == "bool":
                        prompt += " [true/false]"
                    prompt += f"\n  Current: {format_display_value(path, current)}\n  New value: "
                    try:
                        raw = input(prompt).strip()
                        if raw == "":
                            continue
                        val = _parse_value(raw, typ)
                        _set_by_path(cfg, path, val)
                        print("  Updated.")
                    except Exception as e:
                        print(f"  Invalid: {e}")


def main():
    args = [a for a in sys.argv[1:] if a != "--tui" and not a.startswith("-")]
    use_tui = "--tui" in sys.argv
    default_path = _repo_root / "config.yaml"
    config_path = Path(args[0]) if args else default_path
    config_path = config_path.resolve()

    if use_tui:
        run_menu(config_path)
        return

    import tkinter
    try:
        _run_gui(config_path)
    except tkinter.TclError as e:
        if "display" in str(e).lower() or "cannot open" in str(e).lower():
            print("GUI not available (no display), falling back to TUI.", file=sys.stderr)
            run_menu(config_path)
        else:
            raise
    except Exception as e:
        if "tk" in str(type(e).__name__).lower():
            print("GUI failed, falling back to TUI:", e, file=sys.stderr)
            run_menu(config_path)
        else:
            raise


if __name__ == "__main__":
    main()
