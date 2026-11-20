"""
build_docs.py - turn the guide's HTML source into the PDF.

    python docs/build_docs.py           build docs/Solar_System_Simulator_Guide.pdf
    python docs/build_docs.py --check   list code files changed since the last build
                                        (exit code 1 if the guide may be out of date)

HOW IT WORKS
------------
1. Reads the hand-written guide:  docs/source/guide.html
2. Fills in the parts that are generated FROM THE CODE, so they can never
   go out of date. The source marks them with placeholders:
       <!--AUTO:BUILD_INFO-->       date, git commit, file sizes
       <!--AUTO:SETTINGS-->         the current SETTINGS block of main.py
       <!--AUTO:CODE_REFERENCE-->   every module, class and function with its
                                    docstring, read with Python's `ast` module
3. Writes docs/build/guide.html, then asks Microsoft Edge (or Chrome) to
   "print" it to PDF in headless mode (no window opens).
4. Saves a fingerprint (hash) of every .py file in docs/build/manifest.json.
   --check compares against it to tell you which code changed since then.

The hand-written chapters still need updating by hand (or by Claude, see
CLAUDE.md in the project root) when the code changes.
"""

import argparse
import ast
import datetime
import hashlib
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
PROJECT = DOCS.parent
SOURCE = DOCS / "source" / "guide.html"
BUILD = DOCS / "build"
OUTPUT_HTML = BUILD / "guide.html"
MANIFEST = BUILD / "manifest.json"
PDF = DOCS / "Solar_System_Simulator_Guide.pdf"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


# ---------------------------------------------------------------- helpers
def code_files():
    """The project's own Python files (not the virtual environment)."""
    return sorted(p for p in PROJECT.glob("*.py"))


def fingerprints():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files()}


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def first_paragraph(doc):
    """The docstring up to its first blank line."""
    if not doc:
        return ""
    paragraph = []
    for line in doc.strip().splitlines():
        if not line.strip():
            break
        paragraph.append(line.strip())
    return " ".join(paragraph)


def esc(text):
    return html.escape(text, quote=False)


# ---------------------------------------------------------- AUTO sections
def build_info():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    commit = git("rev-parse", "--short", "HEAD") or "not a git repository"
    dirty = " (plus uncommitted changes)" if git("status", "--porcelain", "--", "*.py") else ""
    rows = "".join(
        f"<tr><td><code>{p.name}</code></td><td>{len(p.read_text(encoding='utf-8').splitlines())}</td></tr>"
        for p in code_files()
    )
    return (
        f"<p><b>Built:</b> {now} &nbsp;·&nbsp; <b>Code version:</b> git commit "
        f"<code>{esc(commit)}</code>{dirty}</p>"
        f"<table class='small'><tr><th>File</th><th>Lines</th></tr>{rows}</table>"
    )


def settings_block():
    """Copy the SETTINGS block of main.py exactly as it is right now."""
    text = (PROJECT / "main.py").read_text(encoding="utf-8")
    start = text.find("# ====")
    end = text.find("# ====", start + 1)
    if start < 0 or end < 0:
        return "<p><i>(Could not find the SETTINGS block in main.py.)</i></p>"
    end = text.find("\n", end)
    return f"<pre class='code'>{esc(text[start:end])}</pre>"


def describe_function(node, prefix=""):
    args = ast.unparse(node.args)
    doc = first_paragraph(ast.get_docstring(node))
    doc_html = f"<div class='refdoc'>{esc(doc)}</div>" if doc else ""
    return f"<div class='ref'><code class='sig'>{prefix}{node.name}({esc(args)})</code>{doc_html}</div>"


def code_reference():
    parts = []
    for path in code_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parts.append(f"<h3 class='refmod'>{path.name}</h3>")
        mod_doc = ast.get_docstring(tree)
        if mod_doc:
            # module docstrings start with "file.py - summary"
            parts.append(f"<p class='refsummary'>{esc(first_paragraph(mod_doc))}</p>")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    parts.append(describe_function(node))
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                parts.append(f"<div class='ref'><code class='sig'>class {node.name}</code>"
                             f"<div class='refdoc'>{esc(first_paragraph(ast.get_docstring(node)))}</div></div>")
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and (not item.name.startswith("_") or item.name == "__init__"):
                        parts.append(describe_function(item, prefix=f"{node.name}."))
    return "\n".join(parts)


# ------------------------------------------------------------------ build
def find_browser():
    for candidate in BROWSERS:
        if Path(candidate).exists():
            return candidate
    for name in ("msedge", "chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    return None


def build():
    source = SOURCE.read_text(encoding="utf-8")
    page = (source
            .replace("<!--AUTO:BUILD_INFO-->", build_info())
            .replace("<!--AUTO:SETTINGS-->", settings_block())
            .replace("<!--AUTO:CODE_REFERENCE-->", code_reference()))
    BUILD.mkdir(exist_ok=True)
    OUTPUT_HTML.write_text(page, encoding="utf-8")
    print(f"HTML written: {OUTPUT_HTML}")

    browser = find_browser()
    if browser is None:
        sys.exit("No Edge or Chrome found. Open docs/build/guide.html in a browser and print it to PDF.")
    # A PDF that's open in a viewer may be locked, so write to a temp name first.
    tmp_pdf = BUILD / "guide_tmp.pdf"
    subprocess.run([
        browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={tmp_pdf}", OUTPUT_HTML.as_uri(),
    ], check=True, timeout=180, capture_output=True)
    try:
        tmp_pdf.replace(PDF)
    except PermissionError:
        sys.exit(f"Could not overwrite {PDF.name}. Is it open in a PDF viewer? New copy: {tmp_pdf}")
    MANIFEST.write_text(json.dumps(fingerprints(), indent=2))
    print(f"PDF written:  {PDF}")


def check():
    if not MANIFEST.exists():
        print("The guide has never been built. Run: python docs/build_docs.py")
        return 1
    old = json.loads(MANIFEST.read_text())
    new = fingerprints()
    changed = sorted(n for n in new if n in old and new[n] != old[n])
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    if not (changed or added or removed):
        print("Guide is up to date with the code.")
        return 0
    for label, names in (("changed", changed), ("new", added), ("removed", removed)):
        for n in names:
            print(f"  {label:>8}: {n}")
    print("Update docs/source/guide.html for these, then run: python docs/build_docs.py")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check", action="store_true", help="only report code changed since the last build")
    sys.exit(check() if parser.parse_args().check else build())
