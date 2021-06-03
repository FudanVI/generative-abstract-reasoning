from torch.utils.data import Dataset
import os
import torch


class PolyMRDataset(Dataset):
    def __init__(self, root='cache', type='triangle_5000', set='train', size=64):
        super(PolyMRDataset, self).__init__()
        self.root = root
        self.type = type
        self.set = set
        self.size = size
        self.full_path = os.path.join(root, type)

        if self.check_file_exist():
            self.panel = torch.load(os.path.join(self.root, '%s_%s_%d.pt' % (self.type, self.set, self.size)))
            self.num_data = self.panel.size(0)
        else:
            print('Cannot find dataset.')
            exit(0)

    def __getitem__(self, index):
        return self.panel[index][:-1] / 255, self.panel[index][-1] / 255

    def __len__(self):
        return self.num_data

    def check_file_exist(self):
        return os.path.exists(os.path.join(self.root, '%s_%s_%d.pt' % (self.type, self.set, self.size)))


class PolyMRSelectionDataset(Dataset):
    def __init__(self, root='cache', type='position_triangle_5000', set="train", size=64):
        super(PolyMRSelectionDataset, self).__init__()
        self.root = root
        self.type = type
        self.set = set
        self.size = size
        self.panel_name = '%s_selection_%s_%d.pt' % (self.type, self.set, self.size)
        self.selection_name = '%s_selection_%s_%d.selection' % (self.type, self.set, self.size)
        self.answer_name = '%s_selection_%s_%d.answer' % (self.type, self.set, self.size)

        if self.check_file_exist():
            self.panel = torch.load(os.path.join(self.root, self.panel_name))
            self.selection = torch.load(os.path.join(self.root, self.selection_name))
            self.answer = torch.load(os.path.join(self.root, self.answer_name))
            self.num_data = self.panel.size(0)
        else:
            print('Cannot find dataset.')
            exit(0)

    def __getitem__(self, index):
        return self.panel[index][:-1] / 255, self.panel[index][-1] / 255,\
               self.selection[index] / 255, self.answer[index]

    def __len__(self):
        return self.num_data

    def check_file_exist(self):
        if not os.path.exists(os.path.join(self.root, self.panel_name)):
            return False
        if not os.path.exists(os.path.join(self.root, self.selection_name)):
            return False
        return True
