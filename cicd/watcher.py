"""
cicd/watcher.py
File system watcher — auto-triggers pipeline when new .xlsx is dropped
Run with: python cicd/watcher.py
"""

import os
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from loguru import logger

WATCH_DIR = Path(os.getenv("PIPELINE_TRIGGER_DIR", "./data/uploads"))
WATCH_DIR.mkdir(parents=True, exist_ok=True)


class RequirementFileHandler(FileSystemEventHandler):
    def __init__(self):
        self._processed = set()

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix not in (".xlsx", ".xls"):
            return
        if str(path) in self._processed:
            return
        self._processed.add(str(path))
        logger.info(f"📂 New file detected: {path.name}")
        time.sleep(1)  # Wait for file to finish writing
        self._trigger_pipeline(path)

    def _trigger_pipeline(self, path: Path):
        logger.info(f"🚀 Triggering pipeline for: {path.name}")
        try:
            # Option 1: Run pipeline stages directly via Python
            result = subprocess.run(
                ["python", "-m", "cicd.run_pipeline", str(path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.success(f"✅ Pipeline complete for {path.name}")
                logger.info(result.stdout[-500:])
            else:
                logger.error(f"❌ Pipeline failed: {result.stderr[-300:]}")

        except subprocess.TimeoutExpired:
            logger.error("Pipeline timed out after 300s")
        except Exception as e:
            logger.error(f"Pipeline trigger error: {e}")


def main():
    logger.info(f"👁️ Watching for new requirement files in: {WATCH_DIR.resolve()}")
    logger.info("Drop a .xlsx file to trigger the pipeline automatically.")
    logger.info("Press Ctrl+C to stop.")

    handler = RequirementFileHandler()
    observer = Observer()
    observer.schedule(handler, str(WATCH_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Watcher stopped.")
    observer.join()


if __name__ == "__main__":
    main()
