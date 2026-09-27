"""Tiny local server so you can run the scan from your phone.

Run it on the same computer that runs scan.py (it needs the internet for price
data). Your phone, on the SAME Wi-Fi, then talks to it - either through the
mobile/ Expo app, or just by opening the printed address in your phone browser.

    python server.py            # starts on port 8000
    python server.py 9000       # custom port

It prints the address to use on your phone. The data is read-only screening
output; there's no login, so only run it on a network you trust.

Uses only the Python standard library - no extra installs beyond what scan.py
already needs.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import scan
from daytrader.clock import describe_now
from daytrader.playbook import buy_now_candidates

# Map the phone's button names to scan views.
VIEW_ALIASES = {
    "buy": "intraday",
    "buynow": "intraday",
    "intraday": "intraday",
    "morning": "morning",
    "overnight": "overnight",
    "swing": "swing",
    "position": "position",
    "longhold": "position",
    "long": "position",
    "day": "day",
    "holdings": "holdings",
    "positions": "holdings",
    "mypositions": "holdings",
    "all": "all",
    "auto": "auto",
}


def scan_to_payload(view: str) -> dict:
    """Run a scan for one view and return a JSON-able dict (text + summary)."""
    resolved = VIEW_ALIASES.get(view.lower(), "auto")

    if resolved == "holdings":
        res = scan.scan_holdings_only()
        text = scan.build_holdings_text(res["now"], res["holdings"], res.get("holdings_issues"))
        holds = res["holdings"]
        n = len(holds)
        summary = f"{n} open position(s)" if n else "no open positions listed"
        return {
            "ok": True,
            "view": "holdings",
            "scanned_at": describe_now(res["now"]),
            "summary": summary,
            "tickers": n,
            "errors": [r["ticker"] for r in res["results"] if "error" in r],
            "text": text,
        }

    result = scan.run_scan(view=resolved)
    results, now, extra_views = result["results"], result["now"], result["extra_views"]
    text = scan.build_prompt_text(results, now, extra_views, holdings=result.get("holdings"),
                                  holdings_issues=result.get("holdings_issues"))
    n_buy = len(buy_now_candidates(results))
    errors = [r["ticker"] for r in results if "error" in r]

    summary = f"{n_buy} fresh buy signal(s)" if n_buy else "No fresh buy signals"
    if "overnight" in extra_views:
        summary += " - overnight list included"
    elif "morning" in extra_views:
        summary += " - morning list included"

    return {
        "ok": True,
        "view": resolved,
        "scanned_at": describe_now(now),
        "summary": summary,
        "tickers": len(results),
        "errors": errors,
        "text": text,
    }


def lan_ip() -> str:
    """Best-guess LAN IP address other devices on the Wi-Fi can reach."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _in_cgnat(ip: str) -> bool:
    """True if ip is in Tailscale's 100.64.0.0/10 range."""
    try:
        a, b = (int(x) for x in ip.split(".")[:2])
        return a == 100 and 64 <= b <= 127
    except Exception:
        return False


def tailscale_ip() -> str | None:
    """The laptop's Tailscale address (100.x.x.x), if Tailscale is installed and
    up. This is the address that works from your phone ANYWHERE, not just on the
    home Wi-Fi. Returns None if Tailscale isn't found.
    """
    exes = ["tailscale"]
    if sys.platform.startswith("win"):
        exes.append(r"C:\Program Files\Tailscale\tailscale.exe")
    for exe in exes:
        path = exe if (os.path.isabs(exe) and os.path.exists(exe)) else shutil.which(exe)
        if not path:
            continue
        try:
            out = subprocess.run([path, "ip", "-4"], capture_output=True, text=True, timeout=5)
            for line in out.stdout.splitlines():
                ip = line.strip()
                if _in_cgnat(ip):
                    return ip
        except Exception:
            pass
    # Fallback: scan our own interface addresses for a 100.x Tailscale address.
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if _in_cgnat(ip):
                return ip
    except Exception:
        pass
    return None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_OPTIONS(self) -> None:  # CORS preflight
        self._send(204, b"", "text/plain")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/health":
            self._json(200, {"ok": True})
            return

        if path == "/scan":
            view = parse_qs(parsed.query).get("view", ["auto"])[0]
            try:
                self._json(200, scan_to_payload(view))
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)})
            return

        self._json(404, {"ok": False, "error": "not found"})

    def log_message(self, *args) -> None:  # quieter console
        return


# Mobile-friendly fallback page (works in any phone browser; no app needed).
PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading Scan</title>
<style>
  :root { color-scheme: light dark; --blue:#2f6df6; --grey:#4b5563; --green:#1f9d55; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 16px;
         max-width: 640px; margin-inline: auto; }
  h1 { font-size: 1.15rem; margin: 0 0 12px; }
  .tabs { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
  .tab { padding: 12px; text-align: center; border-radius: 12px; font-weight: 600;
         background: transparent; border: 2px solid var(--grey); color: inherit; cursor: pointer; }
  .tab.active { background: var(--blue); border-color: var(--blue); color: #fff; }
  .panel { display: none; }
  .panel.active { display: block; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; }
  button.scan { font-size: 1rem; padding: 15px; border: 0; border-radius: 12px;
                background: var(--blue); color: #fff; width: 100%; }
  button.wide { grid-column: 1 / -1; }
  button.alt { background: var(--grey); }
  button:active { opacity: .7; }
  .hint { font-size: .8rem; opacity: .65; margin: 2px 2px 10px; }
  #status { font-size: .9rem; opacity: .85; min-height: 1.2em; margin: 12px 2px 8px; }
  textarea { width: 100%; height: 44vh; font-family: ui-monospace, monospace;
             font-size: .8rem; border-radius: 12px; padding: 10px; }
  .copy { margin-top: 10px; width: 100%; padding: 15px; border: 0; border-radius: 12px;
          background: var(--green); color: #fff; font-size: 1rem; }
</style></head><body>
<h1>Trading Scan</h1>

<div class="tabs">
  <div id="tab-day" class="tab active" onclick="showTab('day')">Day Trading</div>
  <div id="tab-long" class="tab" onclick="showTab('long')">Long Term</div>
</div>

<div id="panel-day" class="panel active">
  <div class="hint">Intraday and short holds - buy, sell same day or within a few days.</div>
  <div class="row">
    <button class="scan" onclick="run('buy')">Buy now</button>
    <button class="scan" onclick="run('morning')">Morning hold</button>
    <button class="scan" onclick="run('overnight')">Overnight</button>
    <button class="scan" onclick="run('swing')">Swing (days)</button>
    <button class="scan alt wide" onclick="run('day')">All day-trading views</button>
  </div>
</div>

<div id="panel-long" class="panel">
  <div class="hint">Positions held for weeks - uptrend pullbacks and tracking what you own.</div>
  <div class="row">
    <button class="scan wide" onclick="run('position')">Long hold (weeks)</button>
    <button class="scan alt wide" onclick="run('holdings')">My positions</button>
  </div>
</div>

<button class="scan alt" style="margin-bottom:6px" onclick="run('all')">Everything (both)</button>
<div id="status">Pick a tab, then tap a scan. A scan takes ~30-60 seconds.</div>
<textarea id="out" readonly placeholder="Result appears here, and is copied to your clipboard."></textarea>
<button class="copy" onclick="copyOut()">Copy result to clipboard</button>

<script>
function showTab(which) {
  document.getElementById('panel-day').classList.toggle('active', which === 'day');
  document.getElementById('panel-long').classList.toggle('active', which === 'long');
  document.getElementById('tab-day').classList.toggle('active', which === 'day');
  document.getElementById('tab-long').classList.toggle('active', which === 'long');
}
async function copyText(t) {
  let ok = false;
  try { await navigator.clipboard.writeText(t); ok = true; } catch (e) {}
  if (!ok) { const o = document.getElementById('out'); o.focus(); o.select();
    try { document.execCommand('copy'); ok = true; } catch (e) {} }
  return ok;
}
function copyOut() {
  const t = document.getElementById('out').value;
  if (!t) { document.getElementById('status').textContent = 'Nothing to copy yet - run a scan first.'; return; }
  copyText(t).then(ok => { document.getElementById('status').textContent =
    ok ? 'Copied to clipboard.' : 'Select the text above to copy.'; });
}
async function run(view) {
  const s = document.getElementById('status'), out = document.getElementById('out');
  s.textContent = 'Scanning (' + view + ')... ~30-60s'; out.value = '';
  try {
    const r = await fetch('/scan?view=' + view);
    const d = await r.json();
    if (!d.ok) { s.textContent = 'Error: ' + (d.error || 'failed'); return; }
    out.value = d.text;
    const copied = await copyText(d.text);
    s.textContent = (copied ? 'Copied to clipboard. ' : 'Tap "Copy result" below. ')
                    + d.scanned_at + ' - ' + d.summary;
  } catch (e) { s.textContent = 'Could not reach the scanner: ' + e; }
}
</script>
</body></html>
"""


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    ip = lan_ip()
    ts = tailscale_ip()
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("Day Trading scan server is running.")
    print(f"  On THIS computer:        http://localhost:{port}")
    print(f"  Phone on SAME Wi-Fi:     http://{ip}:{port}")
    if ts:
        print(f"  Phone ANYWHERE (Tailscale): http://{ts}:{port}")
    else:
        print("  Phone ANYWHERE: install Tailscale, then re-run")
        print("    this, or run 'tailscale ip -4' and use http://<that-100.x-address>:%d" % port)
    print("  (Use the same address as the Expo app's API URL, or open it in your phone browser.)")
    print("Leave this window open while you use it. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
