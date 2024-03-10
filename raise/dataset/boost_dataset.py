class BoostDataLoader:
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.device = device
        self.next_pointer = 0
        self.samples = []
        for data in self.data_loader_iter:
            if type(data) is tuple or type(data) is list:
                data = tuple([d.to(self.device) for d in data])
            else:
                data = data.to(self.device)
            self.samples.append(data)
        self.split_size = len(self.samples)

    def visit(self):
        return self.samples

    def visit_one(self):
        return self.samples[0]

    def get_dataset(self):
        return self.data_loader.dataset
