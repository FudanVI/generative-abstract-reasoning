import threading
import queue


class QueueDataLoader:
    def __init__(self, data_loader, device, maxsize=3):
        self.queue = queue.Queue(maxsize=maxsize)
        self.data_loader = data_loader
        self.data_loader_list = list(data_loader)
        self.device = device
        self.split_size = len(self.data_loader_list)
        self.next_pointer = 0
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        while True:
            data = self.data_loader_list[self.next_pointer]
            if type(data) is tuple or type(data) is list:
                data = self.get_dataset().process_input(*data)
                data = (d.to(self.device) for d in data)
            else:
                data = self.get_dataset().process_input(data)
                data = data.to(self.device)
            self.queue.put(data)
            self.next_pointer = (self.next_pointer + 1) % self.split_size

    def visit(self):
        for i in range(self.split_size):
            yield i, self.queue.get()

    def visit_one(self):
        return self.queue.get()

    def finish_task(self):
        self.queue.task_done()

    def get_dataset(self):
        return self.data_loader.dataset
