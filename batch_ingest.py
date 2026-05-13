"""
Production batch ingestion script.
Run this to upload 1000+ files reliably with progress tracking.

Usage:
  python batch_ingest.py --folder "C:/your/documents" --workers 2
"""

import asyncio
import argparse
import httpx
import json
from pathlib import Path
from loguru import logger
import time

BACKEND_URL = "http://localhost:8000/api/v1"
ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv"}
MAX_FILE_SIZE_MB = 50


async def upload_file(client: httpx.AsyncClient, file_path: Path, semaphore: asyncio.Semaphore) -> dict:
    """Upload a single file with retry logic."""
    async with semaphore:
        for attempt in range(3):  # retry up to 3 times
            try:
                with open(file_path, "rb") as f:
                    files = {"file": (file_path.name, f, "application/octet-stream")}
                    response = await client.post(
                        f"{BACKEND_URL}/ingest",
                        files=files,
                        timeout=120.0,
                    )
                if response.status_code == 200:
                    logger.success(f"✓ Uploaded: {file_path.name}")
                    return {"file": file_path.name, "status": "success"}
                else:
                    logger.warning(f"Failed {file_path.name}: {response.text}")
                    return {"file": file_path.name, "status": "failed", "error": response.text}
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {file_path.name}: {e}")
                await asyncio.sleep(2 ** attempt)  # exponential backoff

        return {"file": file_path.name, "status": "failed", "error": "Max retries exceeded"}


async def batch_ingest(folder: str, max_workers: int = 2):
    """
    Upload all files in a folder.
    max_workers=2 means 2 files uploading simultaneously.
    Keep this low (2-3) on CPU to avoid memory overflow.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        logger.error(f"Folder not found: {folder}")
        return

    # Find all supported files
    all_files = []
    for ext in ALLOWED_EXTENSIONS:
        all_files.extend(folder_path.rglob(f"*{ext}"))

    # Filter out files that are too large
    valid_files = []
    skipped = []
    for f in all_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            skipped.append(f.name)
            logger.warning(f"Skipping {f.name} — {size_mb:.1f}MB exceeds {MAX_FILE_SIZE_MB}MB limit")
        else:
            valid_files.append(f)

    logger.info(f"Found {len(valid_files)} files to upload ({len(skipped)} skipped)")
    logger.info(f"Uploading with {max_workers} concurrent workers")

    if not valid_files:
        logger.error("No valid files found")
        return

    # Semaphore limits concurrent uploads
    semaphore = asyncio.Semaphore(max_workers)
    results = []
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        # Check backend is alive
        try:
            health = await client.get(f"http://localhost:8000/health", timeout=5.0)
            logger.info(f"Backend health: {health.json()}")
        except Exception:
            logger.error("Backend is not running! Start it with: python run.py")
            return

        # Create all upload tasks
        tasks = [upload_file(client, f, semaphore) for f in valid_files]

        # Run with progress tracking
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            pct = (completed / len(valid_files)) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (len(valid_files) - completed) if completed > 0 else 0
            logger.info(f"Progress: {completed}/{len(valid_files)} ({pct:.1f}%) — ETA: {eta:.0f}s")

    # Summary report
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    total_time = time.time() - start_time

    logger.info("=" * 50)
    logger.info(f"BATCH COMPLETE")
    logger.info(f"Total files:  {len(valid_files)}")
    logger.info(f"Successful:   {len(successful)}")
    logger.info(f"Failed:       {len(failed)}")
    logger.info(f"Skipped:      {len(skipped)}")
    logger.info(f"Total time:   {total_time:.1f}s")
    logger.info(f"Avg per file: {total_time/len(valid_files):.1f}s")

    # Save report
    report = {
        "total": len(valid_files),
        "successful": len(successful),
        "failed": len(failed),
        "skipped": skipped,
        "failed_files": failed,
        "time_seconds": round(total_time, 1),
    }
    with open("batch_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to batch_report.json")

    if failed:
        logger.warning("Failed files:")
        for f in failed:
            logger.warning(f"  - {f['file']}: {f.get('error', 'unknown')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ingest documents into RAG system")
    parser.add_argument("--folder", required=True, help="Folder containing PDF/Excel files")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent upload workers (default: 2)")
    args = parser.parse_args()
    asyncio.run(batch_ingest(args.folder, args.workers))