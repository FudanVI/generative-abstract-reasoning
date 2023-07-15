import threading
import queue


class QueueDataLoader:
    def __init__(self, data_loader, device, maxsize=3):
        self.queue = queue.Queue(maxsize=maxsize)
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.device = device
        self.split_size = len(data_loader)
        self.next_pointer = 0
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        while True:
            try:
                data = next(self.data_loader_iter)
            except StopIteration:
                self.data_loader_iter = iter(self.data_loader)
                data = next(self.data_loader_iter)
            if type(data) is tuple or type(data) is list:
                data = (d.to(self.device) for d in data)
            else:
                data = data.to(self.device)
            self.queue.put(data)

    def visit(self):
        for i in range(self.split_size):
            yield i, self.queue.get()

    def visit_one(self):
        return self.queue.get()

    def finish_task(self):
        self.queue.task_done()

    def get_dataset(self):
        return self.data_loader.dataset
