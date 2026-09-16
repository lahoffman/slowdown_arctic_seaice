#!/usr/bin/env python3
"""
make_figures.py — build every figure a manuscript includes (--paper aies|grl), from the manifest in
configs/figures_<paper>.py, into results/figures/<paper>/ under the manuscript's own filenames.

make_figure entries are grouped by identical options and built in one make_figure.py call each
(shared data cache); diagnostic entries are copied from results/figures/diagnostics/. Missing
inputs are reported, not fatal, so the script can be rerun after every new run and the status
table (results/figures/aies/STATUS.md) shows what is current and what is still waiting.

Usage:
  python scripts/make_figures.py                 # everything
  python scripts/make_figures.py --only fig_S5.pdf fig_2.pdf
  python scripts/make_figures.py --diagnostics-only     # just copy the 09_*/10_* PNGs (seconds)
  python scripts/make_figures.py --list
  python scripts/make_figures.py --paper grl           # the GRL paper (configs/figures_grl.py)
"""

import argparse
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
import importlib

DIAG = paths.FIGURES_DIR / "diagnostics"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", default=None, help="manuscript filenames to build")
    p.add_argument("--diagnostics-only", action="store_true")
    p.add_argument("--fmt", default="pdf", choices=["pdf", "png"])
    p.add_argument("--list", action="store_true")
    p.add_argument("--paper", default="aies", choices=["aies", "grl"],
                   help="which manifest (configs/figures_<paper>.py) and output folder (results/figures/<paper>/)")
    return p.parse_args()


def main():
    a = parse_args()
    MANIFEST = importlib.import_module(f"configs.figures_{a.paper}").MANIFEST
    OUT = paths.FIGURES_DIR / a.paper
    items = {k: v for k, v in MANIFEST.items() if not a.only or k in a.only}
    if a.list:
        for k, v in items.items():
            print(f"  {k:36s} {v['kind']:12s} {' '.join(v.get('ids', [])):16s} {' '.join(v.get('args', [])) or v.get('src', '')}")
        return
    OUT.mkdir(parents=True, exist_ok=True); tmp = OUT / "_build"; tmp.mkdir(exist_ok=True)
    status = {}

    # 1. diagnostics: copy
    for name, v in items.items():
        if v["kind"] != "diagnostic":
            continue
        src = DIAG / v["src"]
        if src.exists():
            shutil.copy2(src, OUT / name); status[name] = ("ok", time.strftime("%Y-%m-%d %H:%M", time.localtime(src.stat().st_mtime)))
        else:
            status[name] = ("missing", f"needs {v['needs']}")

    # 2. make_figure: group by identical args
    if not a.diagnostics_only:
        groups = defaultdict(list)
        for name, v in items.items():
            if v["kind"] == "make_figure":
                groups[tuple(v["args"])].append((name, v))
        for args, entries in groups.items():
            ids = sorted({i for _, v in entries for i in v["ids"]})
            cmd = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "make_figure.py"), *ids,
                   "--fmt", a.fmt, "--out-dir", str(tmp), *args]
            print("\n>>", " ".join(cmd[2:]))
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(r.stdout[-2000:]);
            if r.returncode:
                print(r.stderr[-2000:])
            suffix = args[args.index("--suffix") + 1] if "--suffix" in args else ""
            for name, v in entries:
                built = tmp / f"fig_{v['ids'][0]}{suffix}.{a.fmt}"
                if built.exists() and built.stat().st_mtime > time.time() - 3600:
                    shutil.move(str(built), OUT / name); status[name] = ("ok", time.strftime("%Y-%m-%d %H:%M"))
                else:
                    status[name] = ("missing", f"needs {v['needs']}")

    # 3. status table
    lines = [f"# {a.paper.upper()} figure status\n", f"built {time.strftime('%Y-%m-%d %H:%M')} on {Path.home().name}@{__import__('socket').gethostname()}\n",
             "| file | status | built / needs |", "|---|---|---|"]
    for name in items:
        st, note = status.get(name, ("skipped", ""))
        lines.append(f"| `{name}` | {'✅' if st == 'ok' else '⏳'} {st} | {note} |")
    (OUT / "STATUS.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[2:]))
    n_ok = sum(1 for s in status.values() if s[0] == "ok")
    print(f"\n{n_ok}/{len(items)} figures current in {OUT}")


if __name__ == "__main__":
    main()
