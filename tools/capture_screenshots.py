"""Capture application-documentation screenshots of the running web UI.

Drives the live Reflex app with a headless browser and writes PNGs into
``assets/screenshots/`` (served by Reflex at ``/screenshots/...`` and embedded
in the in-app ``/help`` pages).

Usage:
    1. Start the app in one terminal:  ``uv run reflex run``
    2. In another, run this script:
       ``uv run --frozen --group screenshots python tools/capture_screenshots.py``

The target URL defaults to ``http://localhost:3000`` and can be overridden with
the ``PATCH_SIM_URL`` environment variable.
"""

import os
import pathlib
import sys

from playwright.sync_api import (
    Page,
    sync_playwright,
)
from playwright.sync_api import (
    TimeoutError as PlaywrightTimeout,
)

BASE_URL = os.environ.get("PATCH_SIM_URL", "http://localhost:3000")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "assets" / "screenshots"

#: Element screenshots captured after a single-sweep action-potential run.
_ELEMENTS: list[tuple[str, str]] = [
    ("#ps-neuron-panel", "neuron-panel.png"),
    ("#ps-protocol-panel", "protocol-panel.png"),
    ("#ps-trace-plot", "trace-plot.png"),
    ("#ps-sweep-manager", "sweep-manager.png"),
    ("#ps-analysis-sidebar", "analysis.png"),
]


def _run_and_wait(page: Page) -> None:
    """Click Run and wait for the Plotly trace to finish rendering.

    Retries the click because an early click can land before the Reflex
    websocket has connected (the handler is not yet wired, so the event is
    dropped). While a run is in flight the Run button is replaced by a spinner,
    so a missing button just means we wait.

    Args:
        page: The Playwright page driving the app.

    Raises:
        RuntimeError: If the trace never renders after several attempts.
    """
    # Plotly.react() adds the ``js-plotly-plot`` class to the target div itself
    # (``#ps-trace-plot``), not to a descendant, so check the element directly.
    rendered = (
        "() => { const e = document.querySelector('#ps-trace-plot');"
        " return !!e && (e.classList.contains('js-plotly-plot')"
        " || !!e.querySelector('.js-plotly-plot')); }"
    )
    for _ in range(5):
        run_btn = page.get_by_role("button", name="Run", exact=True)
        if run_btn.count() > 0 and run_btn.first.is_visible():
            run_btn.first.click()
        try:
            page.wait_for_function(rendered, timeout=15_000)
            page.wait_for_timeout(1_500)
            return
        except PlaywrightTimeout:
            continue
    raise RuntimeError("trace plot did not render after multiple Run attempts")


def _select_protocol_preset(page: Page, name: str) -> None:
    """Choose a protocol preset from the Experiment panel dropdown.

    Args:
        page: The Playwright page driving the app.
        name: The exact preset label to select (e.g. ``"F-I Curve"``).
    """
    page.locator("#ps-protocol-panel").get_by_role("combobox").first.click()
    page.get_by_role("option", name=name, exact=True).click()
    page.wait_for_timeout(500)


def _capture(page: Page) -> None:
    """Capture the full screenshot set into ``OUTPUT_DIR``.

    Args:
        page: The Playwright page driving the app.
    """
    # Track the Reflex backend event socket so we do not click before it is
    # connected (an early click would be dropped, and the run never starts).
    event_socket_open = []
    page.on(
        "websocket",
        lambda ws: event_socket_open.append(ws.url) if "/_event/" in ws.url else None,
    )

    page.goto(BASE_URL, wait_until="networkidle")
    page.locator("#ps-sidebar").wait_for(state="visible", timeout=30_000)
    for _ in range(60):
        if event_socket_open:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(2_500)  # let the socket.io handshake settle

    # Default load is the squid axon + Action Potential preset.
    _run_and_wait(page)
    page.screenshot(path=str(OUTPUT_DIR / "overview.png"))
    for selector, filename in _ELEMENTS:
        page.locator(selector).screenshot(path=str(OUTPUT_DIR / filename))
        print(f"  wrote {filename}")
    print("  wrote overview.png")

    # A multi-sweep example for the trace-plot section.
    _select_protocol_preset(page, "F-I Curve")
    _run_and_wait(page)
    page.locator("#ps-trace-plot").screenshot(path=str(OUTPUT_DIR / "fi-curve.png"))
    print("  wrote fi-curve.png")


def main() -> int:
    """Capture all documentation screenshots.

    Returns:
        Process exit code (0 on success).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Capturing screenshots from {BASE_URL} into {OUTPUT_DIR}")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": 1500, "height": 950},
            device_scale_factor=2,
            color_scheme="light",
        )
        page = context.new_page()
        try:
            _capture(page)
        finally:
            context.close()
            browser.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
