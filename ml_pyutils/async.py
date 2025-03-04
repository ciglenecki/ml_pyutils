import concurrent
import concurrent.futures
import os
import threading
from contextlib import contextmanager
from functools import partial
from typing import Callable


@contextmanager
def multitask_executor_context(
    *tasks: partial | Callable,
    error_event: threading.Event | None = None,
    pool_executor_constructor: type[
        concurrent.futures.ThreadPoolExecutor
    ] = concurrent.futures.ThreadPoolExecutor,
):
    """Context manager that runs tasks concurrently using ProcessPoolExecutor | ThreadPoolExecutor. It submits all tasks, yields (perform some logic here), and then waits for all tasks to finish.

    Example usage:

    with multitask_executor_context(task1, task2, task3) as _:
        # Do something while tasks are running
    # All tasks are finished here

    Args:
        tasks: Iterable of tasks to run concurrently.

    Returns:
        List: Results of the tasks.
    """
    tasks_list = list(tasks)
    max_workers = min(os.cpu_count() or 1, len(tasks_list))
    pool_executor = pool_executor_constructor(max_workers=max_workers)
    with pool_executor as executor:
        futures = list[concurrent.futures.Future]()
        for task in tasks_list:
            future = executor.submit(task)
            futures.append(future)
        try:
            yield
        except Exception as e:
            # if yield raises an exception we want to stop all other tasks
            if error_event is not None:
                error_event.set()
            raise e
        finally:
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_EXCEPTION
            )
            for future in done:
                if (e := future.exception()) is not None:
                    # if exception is raised in one of the tasks
                    # we want to stop all other tasks
                    if error_event is not None:
                        error_event.set()
                    for future in futures:
                        future.cancel()
                    raise e
