import asyncio
import concurrent.futures
import logging
import multiprocessing
from typing import Any, Callable, Coroutine, List, Optional, Union

from tqdm import tqdm

DEFAULT_NUM_WORKERS = 4

logger = logging.getLogger(__name__)


def run_async_job_in_thread(job: Union[Callable, Coroutine]) -> Any:
    try:
        loop = asyncio.get_running_loop()
        logger.debug("Using existing event loop for async job.")
        future = asyncio.ensure_future(job)
        return loop.run_until_complete(future)
    except RuntimeError:
        logger.debug("Creating new event loop for async job.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(job)
        finally:
            loop.close()
            logger.debug("Closed new event loop.")
        return result


def run_jobs_threadpool(
    jobs: List[Union[Callable, Coroutine]],
    workers: int = DEFAULT_NUM_WORKERS,
    show_progress: bool = False,
    desc: Optional[str] = None,
) -> List[Any]:
    results = []

    logger.info(f"Starting threadpool with {workers} workers and {len(jobs)} jobs.")
    if workers > len(jobs):
        workers = len(jobs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []

        for i, job in enumerate(jobs):
            if asyncio.iscoroutine(job):
                logger.debug(f"Submitting async job {i+1}/{len(jobs)} to thread pool.")
                futures.append(executor.submit(run_async_job_in_thread, job))
            else:
                logger.debug(f"Submitting sync job {i+1}/{len(jobs)} to thread pool.")
                futures.append(executor.submit(job))

        logger.info("All jobs submitted. Waiting for results...")

        try:
            if show_progress:
                results = [
                    future.result()
                    for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc=desc,
                    )
                ]
            else:
                results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]

        except Exception as e:
            logger.error(
                f"Error while executing jobs in thread pool: {e}", exc_info=True
            )

    logger.info("Threadpool execution completed.")
    return results


def _worker(job: Callable[[], Any]) -> Any:
    return job()


def run_jobs_multiprocessing(
    jobs: List[Callable[[], Any]],
    workers: int = DEFAULT_NUM_WORKERS,
    show_progress: bool = False,
    desc: Optional[str] = None,
) -> List[Any]:
    """Run LLM jobs using multiprocessing (for local models)."""

    with multiprocessing.Pool(processes=workers) as pool:
        if show_progress:
            results = list(tqdm(pool.imap(_worker, jobs), total=len(jobs), desc=desc))
        else:
            results = pool.map(_worker, jobs)

    return results
