# Solar System Simulator

N-body simulator of the Solar System in Python: real starting data from NASA/JPL,
velocity Verlet integration, VPython 3D view with a free-flying camera. A learning
project that follows the book *Moving Planets Around* (PDF in this folder).

## Keep the guide in sync with the code (always)

`docs/Solar_System_Simulator_Guide.pdf` explains the whole project (physics, numerical
methods, NASA data, every file) for a beginner. The owner wants it updated **every time
the code changes**.

After any change to a `.py` file, or to behaviour, settings or results:

1. Update the matching chapter(s) in `docs/source/guide.html`, the hand-written source
   of the PDF. Explain new physics in plain words, as the rest of the guide does.
2. Add a dated entry at the top of the table in **Appendix C: Document history** in
   that file, saying what changed and which chapters were updated.
3. Rebuild: `.venv\Scripts\python.exe docs\build_docs.py`
4. Confirm: `.venv\Scripts\python.exe docs\build_docs.py --check` prints "up to date".

Appendix A (code reference, built from docstrings) and Appendix B (the SETTINGS block of
`main.py`) are generated at build time, so keep docstrings accurate. Any numbers quoted in
the guide (accuracy results, counts from NASA) must come from an actual run, not from memory.

## Commands

```powershell
.venv\Scripts\Activate.ps1
python main.py                     # 3D view (opens a browser tab; VPython exits when the tab closes)
python check_accuracy.py           # physics vs NASA, no graphics (see --help)
python nasa_api.py                 # demo of the NASA data module
python docs\build_docs.py          # rebuild the guide PDF (uses Microsoft Edge headless)
```

## Conventions

- SI units inside the physics; convert only at the edges (NASA data in, drawing out).
- Masses come from NASA GM values (`mass = GM / G`), never typed in kg. See guide Section 21.
- Physics stays in `physics.py`/`integrator.py`/`simulation.py`. Drawing tricks (scales,
  exaggeration) stay in `visualize.py`/`camera.py`.
- Comments explain the physics for a beginner; keep that style.
- Windows: `vpython` 7.6.5 needs `setuptools<81` (pinned in requirements.txt).
