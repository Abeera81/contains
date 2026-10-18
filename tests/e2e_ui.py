"""Real-browser UI check (headless Chrome over the DevTools protocol; genuine mouse events).

    uv run python -I tests/e2e_ui.py

Starts the real app on a local port with a FAKE sampler (no Tinker calls), loads the page, then:
  1. clicks a card chevron -> evidence panel collapses; clicks again -> re-expands
  2. clicks Copy -> clipboard holds the Contains line (+ Verify line)
  3. scrolls to the bottom -> the last content is not covered by the fixed disclaimer
Not part of `make test` (needs Chrome); exits non-zero on any failure and saves a screenshot.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import uvicorn
from websockets.sync.client import connect

from contains import app as appmod
from contains.labels import ALLERGENS

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
APP_PORT, DBG_PORT = 8011, 9233
TEXT = "all-purpose flour, butter, eggs, worcestershire sauce, semisweet chocolate, cooking oil, pistachio spread, pesto, vanilla essence, milk"
SHOT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "ui4_desktop_full.png"


class Fake:
    name = "fake"

    def verdicts(self, text):
        return {a: "no" for a in ALLERGENS} | {"milk": "yes", "egg": "yes", "soy": "unsure", "peanuts": "unsure", "sesame": "yes"}


class CDP:
    def __init__(self, ws_url):
        self.ws, self.i = connect(ws_url, max_size=64 * 1024 * 1024), 0

    def call(self, method, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.i:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]

    def js(self, expr, await_=False):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=await_)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")

    def click(self, selector):
        box = self.js(f"(()=>{{const e=document.querySelector({json.dumps(selector)});e.scrollIntoView({{block:'center'}});"
                      "const r=e.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()")
        x, y = box
        for t in ("mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)
        time.sleep(0.3)


def main() -> int:
    appmod.app.state.sampler = Fake()
    srv = uvicorn.Server(uvicorn.Config(appmod.app, host="127.0.0.1", port=APP_PORT, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    base = f"http://127.0.0.1:{APP_PORT}"
    for _ in range(50):
        try:
            urllib.request.urlopen(base + "/healthz", timeout=1); break
        except Exception:
            time.sleep(0.2)
    prof = tempfile.mkdtemp()
    chrome = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", f"--remote-debugging-port={DBG_PORT}",
                               f"--user-data-dir={prof}", "--window-size=900,1100", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fails = []
    try:
        ws = None
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{DBG_PORT}/json", timeout=1))
                ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page"); break
            except Exception:
                time.sleep(0.2)
        cdp = CDP(ws)
        cdp.call("Page.enable")
        cdp.call("Browser.grantPermissions", permissions=["clipboardReadWrite", "clipboardSanitizedWrite"], origin=base)
        cdp.call("Page.navigate", url=base + "/#text=" + urllib.request.quote(TEXT))
        for _ in range(60):  # wait for the result cards to render
            if cdp.js("document.querySelectorAll('details.a').length") :
                break
            time.sleep(0.25)
        n = cdp.js("document.querySelectorAll('details.a').length")
        print(f"cards rendered: {n}")

        # 1) chevron collapse / expand (real click on the card header)
        def state():
            # a closed <details> keeps its child laid out, so measure the whole card (it shrinks) + visibility
            return cdp.js("(()=>{const d=document.querySelector('details.a');const ev=d.querySelector('.ev');"
                          "return {open:d.open,cardH:Math.round(d.getBoundingClientRect().height),"
                          "evVisible:ev.checkVisibility({contentVisibilityAuto:true,visibilityProperty:true})}})()")
        s0 = state(); cdp.click("details.a > summary .chev"); s1 = state(); cdp.click("details.a > summary .chev"); s2 = state()
        print("chevron: initial", s0, "after click 1", s1, "after click 2", s2)
        if not (s0["open"] and s0["evVisible"]): fails.append("evidence panel not expanded by default")
        if not (s1["open"] is False and not s1["evVisible"] and s1["cardH"] < s0["cardH"]): fails.append("click did not collapse the evidence panel")
        if not (s2["open"] and s2["evVisible"] and s2["cardH"] == s0["cardH"]): fails.append("second click did not re-expand")

        # 2) Copy -> clipboard
        want = cdp.js("document.querySelector('.label .line').textContent + '\\n' + document.querySelector('.label .verify').textContent")
        cdp.click(".label .copy")
        got = cdp.js("navigator.clipboard.readText()", await_=True)
        label = cdp.js("document.querySelector('.label .copy').textContent")
        print("clipboard:", repr(got)); print("button label after click:", repr(label))
        if got.replace("\r\n", "\n") != want: fails.append(f"clipboard mismatch: want {want!r}")
        if label.strip() != "Copied": fails.append("button did not show Copied")

        # 3) fixed disclaimer must not cover the last content
        cdp.js("window.scrollTo(0, document.documentElement.scrollHeight)")
        time.sleep(0.3)
        gap = cdp.js("(()=>{const d=document.querySelector('.disclaimer').getBoundingClientRect().top;"
                     "const f=document.querySelector('#foot').getBoundingClientRect().bottom;return d-f})()")
        print(f"gap between last content and disclaimer top: {gap:.0f}px")
        if gap < 0: fails.append("disclaimer overlaps the last content")

        # full-page screenshot (tall viewport so the fixed disclaimer sits at the very end)
        h = cdp.js("document.documentElement.scrollHeight")
        cdp.call("Emulation.setDeviceMetricsOverride", width=900, height=int(h) + 10, deviceScaleFactor=1, mobile=False)
        cdp.js("window.scrollTo(0,0)"); time.sleep(0.4)
        img = cdp.call("Page.captureScreenshot", format="png")["data"]
        SHOT.write_bytes(base64.b64decode(img)); print("screenshot:", SHOT)
    finally:
        chrome.terminate()
    print("RESULT:", "FAIL " + "; ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
