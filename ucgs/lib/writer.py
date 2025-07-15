import queue
import time
import copy
import threading


class AsyncWriter:

    def __init__(self, max_size=8, sleep_sec=5, timeout=60):
        self.sleep_sec = sleep_sec
        self.timeout = timeout
        self._task_list = queue.Queue(maxsize=max_size)
        self._logs = []
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def add_writer_task(self, fn, *args, **kwargs):
        try:
            self._task_list.put((fn, args, kwargs), block=False)
        except queue.Full:
            return 'Add task {} failed. Full Queue'.format(fn.__name__)
        except Exception as e:
            return 'Add task {} failed. {}'.format(fn.__name__, str(e))

    def worker(self):
        while True:
            try:
                fn, args, kwargs = self._task_list.get()
                fn(*args, **kwargs)
                self._logs.append(fn.__name__)
                self._task_list.task_done()
            except queue.Empty:
                time.sleep(self.sleep_sec)

    def log(self, end='\n'):
        logs = copy.deepcopy(self._logs)
        self._logs = list()
        print('Solve tasks {' + ', '.join(logs) + '}', end=end)
