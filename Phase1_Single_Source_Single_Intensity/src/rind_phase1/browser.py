"""Browse the same installed single-source data used by Phase1Dataset."""

import argparse
import os
import threading
import webbrowser

from rind_phase1.data import Phase1Dataset


def create_app(root=None):
    """Create the original RIND browser with the Phase I dataset and defaults."""
    dataset = Phase1Dataset(root)
    try:
        from rind_dataset.browser import create_app as create_dataset_browser
    except ImportError as exc:
        raise ImportError(
            "Browser dependencies are missing. Run ./scripts/browser.sh or "
            "uv sync --locked --extra browser") from exc
    return create_dataset_browser(str(dataset.root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", "--dataset", help="Data directory; defaults to Phase1Dataset's root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8766")))
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser tab automatically")
    args = parser.parse_args()
    try:
        app = create_app(args.data_root)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.exit(1, f"Cannot open RIND browser: {exc}\n")
    url = f"http://{args.host}:{args.port}"
    print(f"Starting RIND Phase I Browser: {url}", flush=True)
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
