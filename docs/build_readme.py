"""
build_readme.py - keeps README.md in sync with the code.

    python docs/build_readme.py                 refresh the generated parts of README.md
    python docs/build_readme.py --check         exit 1 if README.md is out of date (for CI)
    python docs/build_readme.py --validate      also re-run the accuracy tests (needs NASA data/cache)
    python docs/build_readme.py --screenshots   also retake the screenshots (needs Playwright + Edge/Chrome)

HOW IT WORKS
------------
README.md is hand-written, except for blocks between markers like

    <!-- AUTO:MODULES -->  ...generated...  <!-- /AUTO:MODULES -->

This script rebuilds every such block from the code itself:
    ARCHITECTURE  a Mermaid diagram (GitHub draws it) built from each file's
                  `import` lines, so new files appear automatically
    MODULES       a table of every .py file and the first line of its docstring
    SETTINGS      the SETTINGS block of main.py
    VALIDATION    accuracy results, from docs/validation.json (--validate)
    STATS         file and line counts
Everything else in README.md is yours to edit.

Generated text contains nothing that changes on its own (no dates or commit
hashes), so the GitHub Action (.github/workflows/readme.yml) only commits a
new README when the CODE actually changed something.
"""

import argparse
import ast
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
IMAGES = ROOT / "docs" / "images"
VALIDATION = ROOT / "docs" / "validation.json"

# Which layer each file belongs to in the diagram. Files not listed here are
# still drawn, under "Other", so the diagram never silently misses a file.
LAYERS = {
    "Entry points": ["main", "check_accuracy"],
    "NASA data": ["nasa_api", "data_sources", "orbital_elements"],
    "Physics engine": ["body", "physics", "integrator", "simulation"],
    "3D view": ["visualize", "camera", "trails", "starfield", "swarm_view", "rings", "ui_theme"],
}
NASA_MARKERS = {"astroquery", "urllib"}   # imports that mean "talks to NASA"


# ---------------------------------------------------------------- helpers
def modules():
    return sorted(ROOT.glob("*.py"))


def module_info(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc = ast.get_docstring(tree) or ""
    summary = doc.strip().splitlines()[0] if doc.strip() else ""
    summary = re.sub(rf"^{re.escape(path.name)}\s*[-:]\s*", "", summary)   # "x.py - text" -> "text"
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.add(node.module.split(".")[0])
    lines = len(path.read_text(encoding="utf-8").splitlines())
    return {"summary": summary, "imports": imports, "lines": lines}


def replace_block(text, name, content):
    pattern = re.compile(rf"(<!-- AUTO:{name} -->)(.*?)(<!-- /AUTO:{name} -->)", re.S)
    if not pattern.search(text):
        print(f"  (no AUTO:{name} block in README.md, skipped)")
        return text
    return pattern.sub(lambda m: f"{m.group(1)}\n{content.strip()}\n{m.group(3)}", text)


# ------------------------------------------------------------ AUTO blocks
def block_architecture(info):
    names = {p.stem for p in modules()}
    layer_of = {m: layer for layer, ms in LAYERS.items() for m in ms if m in names}
    other = sorted(names - set(layer_of))
    out = ["```mermaid", "flowchart LR"]
    out.append('    NASA[("NASA / JPL<br/>Horizons · SBDB")]')
    for layer, members in list(LAYERS.items()) + [("Other", other)]:
        present = [m for m in (members if layer != "Other" else other) if m in names]
        if not present:
            continue
        # Mermaid ids must not start with a digit ("3D view"), so prefix them.
        out.append(f'    subgraph layer_{re.sub(r"[^A-Za-z0-9]", "_", layer)}["{layer}"]')
        for m in present:
            out.append(f'        {m}["{m}.py"]')
        out.append("    end")
    for m in sorted(names):
        for dep in sorted(info[m]["imports"] & names):
            out.append(f"    {m} --> {dep}")
        if info[m]["imports"] & NASA_MARKERS:
            out.append(f"    {m} -.-> NASA")
    out += ["    classDef nasa fill:#fdf3e2,stroke:#a3640b", "    class NASA nasa", "```"]
    return "\n".join(out)


def block_modules(info):
    rows = ["| File | What it does | Lines |", "|---|---|---:|"]
    for p in modules():
        i = info[p.stem]
        rows.append(f"| [`{p.name}`]({p.name}) | {i['summary']} | {i['lines']} |")
    return "\n".join(rows)


def block_settings():
    text = (ROOT / "main.py").read_text(encoding="utf-8")
    start = text.find("# ====")
    end = text.find("\n", text.find("# ====", start + 1))
    return "```python\n" + text[start:end].strip() + "\n```"


def block_stats(info):
    n = len(info)
    total = sum(i["lines"] for i in info.values())
    return f"**{n} Python files · {total:,} lines**, most of them comments explaining the physics for beginners."


def block_validation():
    if not VALIDATION.exists():
        return "_No results yet: run `python docs/build_readme.py --validate`._"
    data = json.loads(VALIDATION.read_text())
    rows = ["| Setup (start 2026-01-01, simulate 1 year) | Energy error | Earth vs NASA's real position |",
            "|---|---:|---:|"]
    for r in data["runs"]:
        rows.append(f"| {r['label']} | {r['energy']} | **{r['off_by']}** |")
    return "\n".join(rows)


# ------------------------------------------------------------ --validate
VALIDATION_RUNS = [
    ("Sun + Earth, forward Euler", ["--integrator", "euler"]),
    ("Sun + Earth, velocity Verlet", []),
    ("All 8 planets + Moon, velocity Verlet", ["--bodies", "Sun", "Mercury", "Venus", "Earth", "Moon",
                                               "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]),
    ("Same, 15-minute steps", ["--bodies", "Sun", "Mercury", "Venus", "Earth", "Moon", "Mars",
                               "Jupiter", "Saturn", "Uranus", "Neptune", "--dt-hours", "0.25"]),
]


def run_validation():
    runs = []
    for label, args in VALIDATION_RUNS:
        print(f"  validating: {label}")
        out = subprocess.run([sys.executable, str(ROOT / "check_accuracy.py"), *args], cwd=ROOT,
                             capture_output=True, text=True, timeout=1800).stdout
        energy = re.search(r"worst during run\s*:\s*(\S+)", out)
        off = re.search(r"off by\s*:\s*([\d,]+ km)", out)
        if not (energy and off):
            sys.exit(f"Could not read check_accuracy.py output for '{label}':\n{out}")
        runs.append({"label": label, "energy": energy.group(1), "off_by": off.group(1)})
    VALIDATION.write_text(json.dumps({"runs": runs}, indent=2))


# --------------------------------------------------------- --screenshots
SHOTS = [
    # (file name, body to follow or None, seconds to settle, extra keys to hold)
    ("overview.jpg", "Sun", 14, [("s", 2.0), ("e", 1.3), ("ArrowDown", 0.25)]),
    ("saturn.jpg", "Saturn", 14, []),
    ("jupiter.jpg", "Jupiter", 14, []),
    ("earth.jpg", "Earth", 14, [("s", 0.8)]),
]


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def take_screenshots():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("Screenshots need Playwright: pip install playwright")
    IMAGES.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    env = dict(os.environ, VPYTHON_HTTP_PORT=str(port),
               BROWSER=f'"{sys.executable}" -c pass %s')   # stop VPython opening its own tab
    sim = subprocess.Popen([sys.executable, "main.py"], cwd=ROOT, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with sync_playwright() as p:
            browser = None
            for channel in ("msedge", "chrome", None):      # whatever browser is installed
                try:
                    browser = p.chromium.launch(channel=channel, headless=True,
                                                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
                    break
                except Exception:
                    continue
            if browser is None:
                sys.exit("No Chromium-type browser found for screenshots.")
            page = browser.new_page(viewport={"width": 1400, "height": 820})
            for _ in range(180):                          # wait for NASA data + the server
                try:
                    page.goto(f"http://localhost:{port}")
                    break
                except Exception:
                    time.sleep(1)
            page.wait_for_selector(".ss-panel", timeout=300000)
            time.sleep(25)                                 # let trails build up
            page.mouse.move(700, 410)
            page.keyboard.press("h"); time.sleep(1.5)
            page.screenshot(path=str(IMAGES / "controls.jpg"), type="jpeg", quality=85)
            page.locator(".ss-panel button", has_text="Pause").first.click()
            page.keyboard.press("Escape"); time.sleep(1)
            for name, body, settle, keys in SHOTS:
                page.keyboard.press("h"); time.sleep(1)
                page.locator(".ss-panel select").first.select_option(label=body)
                page.keyboard.press("Escape")
                time.sleep(settle)
                page.mouse.move(700, 410)
                for key, seconds in keys:
                    page.keyboard.down(key); time.sleep(seconds); page.keyboard.up(key)
                if keys:
                    time.sleep(6)
                page.screenshot(path=str(IMAGES / name), type="jpeg", quality=85)
                print(f"  screenshot: docs/images/{name}")
            browser.close()
    finally:
        sim.kill()


# ------------------------------------------------------------------ main
def build(check=False):
    if not README.exists():
        sys.exit("README.md not found")
    info = {p.stem: module_info(p) for p in modules()}
    old = README.read_text(encoding="utf-8")
    new = old
    for name, content in [("ARCHITECTURE", block_architecture(info)), ("MODULES", block_modules(info)),
                          ("SETTINGS", block_settings()), ("VALIDATION", block_validation()),
                          ("STATS", block_stats(info))]:
        new = replace_block(new, name, content)
    if check:
        print("README.md is up to date." if new == old else "README.md is OUT OF DATE: run python docs/build_readme.py")
        return 0 if new == old else 1
    if new != old:
        README.write_text(new, encoding="utf-8")
        print("README.md updated.")
    else:
        print("README.md already up to date.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh the generated parts of README.md")
    parser.add_argument("--check", action="store_true", help="only report whether README.md is up to date")
    parser.add_argument("--validate", action="store_true", help="re-run the accuracy tests first")
    parser.add_argument("--screenshots", action="store_true", help="retake the screenshots first")
    args = parser.parse_args()
    if args.validate:
        run_validation()
    if args.screenshots:
        take_screenshots()
    sys.exit(build(check=args.check))
