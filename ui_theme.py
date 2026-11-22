"""
ui_theme.py - restyles VPython's web page: full-window 3D view, a slide-out
control sidebar, and a heads-up display (HUD) in the top-left corner.

HOW THIS WORKS
--------------
VPython runs in a browser tab. Its page has three parts:
    title    (above the 3D view)  -> we turn it into the HUD overlay
    canvas   (the 3D view)         -> we stretch it to fill the whole window
    caption  (below the 3D view)   -> we turn it into the sidebar panel
VPython inserts caption text into the page with jQuery's .append(), which
also runs any <style> and <script> it contains. So install(scene) sends one
block of CSS (the looks) and JavaScript (the behaviour) through the caption.
No VPython files are modified.

WHAT THE JAVASCRIPT DOES
------------------------
- Resizes the 3D canvas to the window, and again whenever the window resizes.
- Moves the caption into a panel that slides in from the right.
  Open/close: the button in the top-right corner, or the H key. Esc closes it.
- Hands keyboard focus back to the page after you use a slider or menu,
  so the arrow keys fly the camera instead of nudging the slider.
- Stops arrow keys / Space / PageUp / PageDown from scrolling the page.
- Turns the MOUSE WHEEL into short presses of PageUp/PageDown. The camera
  treats those as a burst of forward/backward thrust, so the wheel glides
  instead of jumping (VPython's own wheel zoom is switched off in camera.py).
"""

CSS = """
html, body { margin:0; padding:0; overflow:hidden; background:#000; height:100%; }
body { font-family: "Segoe UI", system-ui, sans-serif; color:#dfe6ee; }
.ui-resizable-handle { display:none !important; }

/* --- the HUD (VPython's title area) --- */
.ss-hud { position:fixed; top:14px; left:18px; z-index:20; pointer-events:none;
          text-shadow:0 1px 3px #000, 0 0 8px #000; line-height:1.35; }
.ss-hud .ss-date { font-size:22px; font-weight:600; letter-spacing:0.02em; color:#fff; }
.ss-hud .ss-info { font-size:13px; color:#b9c6d6; }
.ss-hud .ss-hint { font-size:12px; color:#7f8ea3; }

/* --- the sidebar (VPython's caption area) --- */
.ss-panel { position:fixed; top:0; right:0; height:100vh; width:340px; box-sizing:border-box;
            padding:58px 22px 24px; overflow-y:auto; z-index:30;
            background:rgba(9,13,22,0.86); backdrop-filter:blur(8px);
            border-left:1px solid rgba(140,170,210,0.18);
            transform:translateX(105%); transition:transform 0.35s cubic-bezier(.2,.8,.2,1);
            font-size:13px; line-height:1.55; white-space:pre-wrap !important; }
.ss-panel.open { transform:none; }
.ss-panel .ss-title { position:absolute; top:16px; left:22px; font-size:15px; font-weight:700; color:#fff; }
.ss-panel .ss-h { margin:18px 0 8px; font-size:10.5px; font-weight:700; letter-spacing:0.14em;
                  text-transform:uppercase; color:#7fa6d6; border-bottom:1px solid rgba(140,170,210,0.18);
                  padding-bottom:4px; }
.ss-panel .ss-note { font-size:11.5px; color:#8f9db0; }
.ss-panel kbd { font-family:Consolas, monospace; font-size:11px; padding:0 4px; border-radius:3px;
                background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); }
.ss-panel button { background:#1c2a40 !important; color:#e6eef8 !important; border:1px solid #2e4566 !important;
                   border-radius:6px !important; padding:5px 12px !important; margin:3px 4px 3px 0;
                   font-size:12.5px !important; cursor:pointer; }
.ss-panel button:hover { background:#25395a !important; }
.ss-panel select { background:#111a29; color:#e6eef8; border:1px solid #2e4566; border-radius:6px;
                   padding:4px 6px; font-size:12.5px; width:165px; }
.ss-panel input[type=checkbox] { accent-color:#4d8fe0; vertical-align:middle; }
.ss-panel .ui-slider { background:#1b2638; border:1px solid #2e4566; height:6px; border-radius:3px;
                       margin:9px 0 4px; }
.ss-panel .ui-slider .ui-slider-handle { background:#4d8fe0; border:none; border-radius:50%;
                                         width:15px; height:15px; top:-5px; outline:none; }

/* --- the open/close button --- */
#ss-toggle { position:fixed; top:12px; right:14px; z-index:40; width:38px; height:38px;
             border-radius:9px; border:1px solid rgba(140,170,210,0.35); background:rgba(9,13,22,0.7);
             color:#e6eef8; font-size:19px; cursor:pointer; }
#ss-toggle:hover { background:rgba(37,57,90,0.9); }
"""

JS = """
(function () {
  function findCanvas() {
    if (typeof glowObjs === 'undefined') return null;
    // VPython also makes a default "scene" canvas that never appears on the
    // page, so pick the canvas that's actually attached to the document.
    for (var i = 0; i < glowObjs.length; i++) {
      var o = glowObjs[i];
      if (o && o.__canvas_element && o.wrapper && document.body.contains(o.wrapper[0])) return o;
    }
    return null;
  }

  function setup() {
    var cvs = findCanvas();
    if (!cvs) return false;
    // The page is:  <div title> ... <div canvas-wrapper> <div caption>.
    // (Found by inspecting the live page: VPython copies the title and
    //  caption into these divs, so we style the divs themselves.)
    var wrapper = cvs.wrapper[0];
    var kids = Array.prototype.slice.call(wrapper.parentElement.children);
    var hud = null, panel = null;
    for (var t = 0; t < kids.indexOf(wrapper); t++) {
      if (kids[t].tagName === 'DIV') { hud = kids[t]; break; }   // (a <script> comes first)
    }
    if (!hud) return false;
    for (var k = kids.indexOf(wrapper) + 1; k < kids.length; k++) {
      if (kids[k].tagName === 'DIV') { panel = kids[k]; break; }
    }
    if (!panel || panel.querySelectorAll('button').length < 2) return false;  // not built yet

    // 1. Full-window 3D view. Resize both stacked canvases: the 3D one and
    //    the transparent one on top that draws the labels.
    wrapper.style.position = 'fixed';
    wrapper.style.left = '0'; wrapper.style.top = '0'; wrapper.style.zIndex = '1';
    function fit() {
      var w = window.innerWidth, h = window.innerHeight;
      [cvs.__canvas_element, cvs.__overlay_element].forEach(function (c) {
        if (!c) return;
        c.width = w; c.height = h; c.style.width = w + 'px'; c.style.height = h + 'px';
      });
      cvs.__width = w; cvs.__height = h;   // GlowScript's own record of its size (mouse maths)
      wrapper.style.width = w + 'px'; wrapper.style.height = h + 'px';
    }
    fit();
    window.addEventListener('resize', fit);

    // 2. HUD and sidebar
    hud.classList.add('ss-hud');
    panel.classList.add('ss-panel');
    var title = document.createElement('div');
    title.className = 'ss-title';
    title.textContent = 'Solar System Simulator';
    panel.insertBefore(title, panel.firstChild);

    var toggleBtn = document.createElement('button');
    toggleBtn.id = 'ss-toggle';
    toggleBtn.title = 'Controls (H)';
    toggleBtn.innerHTML = '&#9776;';
    document.body.appendChild(toggleBtn);
    function setOpen(open) {
      panel.classList.toggle('open', open);
      toggleBtn.innerHTML = open ? '&times;' : '&#9776;';
    }
    toggleBtn.addEventListener('click', function () { setOpen(!panel.classList.contains('open')); toggleBtn.blur(); });

    var fsBtn = document.createElement('button');
    fsBtn.textContent = 'Full screen (F11)';
    fsBtn.addEventListener('click', function () {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen();
      else document.exitFullscreen();
      fsBtn.blur();
    });
    var fsHolder = document.getElementById('ss-fullscreen-slot');
    if (fsHolder) fsHolder.appendChild(fsBtn);

    // 3. Keyboard: H toggles the panel, Esc closes it, keys never scroll the page
    var noScroll = [' ', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown'];
    document.addEventListener('keydown', function (e) {
      var tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' && e.target.type === 'text') return;
      if (e.key === 'h' || e.key === 'H') setOpen(!panel.classList.contains('open'));
      if (e.key === 'Escape') setOpen(false);
      if (noScroll.indexOf(e.key) >= 0) e.preventDefault();
    });

    // 4. After using a control, give the keyboard back to the camera
    panel.addEventListener('mouseup', function () {
      setTimeout(function () {
        var a = document.activeElement;
        if (a && a !== document.body && a.tagName !== 'SELECT') a.blur();
      }, 0);
    });
    panel.addEventListener('change', function (e) {
      if (e.target && e.target.tagName === 'SELECT') e.target.blur();
    });

    // 5. Mouse wheel -> short PageUp/PageDown presses = smooth thrust bursts
    var wheelKey = null, wheelTimer = null;
    function fakeKey(type, code) {
      var ev = $.Event(type); ev.which = code; ev.keyCode = code;
      $(document).trigger(ev);
    }
    wrapper.addEventListener('wheel', function (e) {
      e.preventDefault(); e.stopPropagation();
      var code = e.deltaY < 0 ? 33 : 34;   // 33 = PageUp (forward), 34 = PageDown (back)
      if (wheelKey !== code) {
        if (wheelKey) fakeKey('keyup', wheelKey);
        fakeKey('keydown', code); wheelKey = code;
      }
      clearTimeout(wheelTimer);
      wheelTimer = setTimeout(function () { fakeKey('keyup', wheelKey); wheelKey = null; }, 150);
    }, { passive: false, capture: true });
    return true;
  }

  // VPython builds the page a piece at a time, so retry until it's ready.
  var timer = setInterval(function () { if (setup()) clearInterval(timer); }, 100);
})();
"""


def install(scene):
    """Send the styling and behaviour to the browser. Call AFTER all controls exist."""
    scene.append_to_caption("<style>" + CSS + "</style><script>" + JS + "</script>")
