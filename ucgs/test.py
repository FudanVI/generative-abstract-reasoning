import os
import yaml
import argparse
import tqdm
import torch
import importlib
from dataset.get_dataset import get_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='ucgst')
parser.add_argument('--model', type=str, default='model_ucgst')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--size', type=int, default=128)
pre_args = parser.parse_args()


device = torch.device("cuda:%d" % pre_args.gpu) if pre_args.gpu >= 0 else torch.device("cpu")
exp_name, model_name = pre_args.exp_name, pre_args.model
save_path = os.path.join('saves', exp_name)
load_path = os.path.join(save_path, 'checkpoint.pt.tar')
save_file = None
while save_file is None:
    try:
        save_file = torch.load(load_path, map_location=device, weights_only=False)
    except:
        save_file = None
start_epoch = save_file['epoch']
args = save_file['args']
with open('config/main.yaml') as f:
    config = yaml.safe_load(f)
args.dataset_info = config['dataset_info']
args.gpu = pre_args.gpu
args.size = pre_args.size
print('Load model trained after {} epochs.'.format(start_epoch))
main = importlib.import_module('{}.main'.format(model_name))
importlib.reload(main)

# Load model parameters
model = main.MainNet(args, device, global_step=start_epoch)
model = model.to(device)
model.load_state_dict(save_file['model'])
model = model.eval()


args.dataset = 'RAVEN'
args.batch = 128
test_loader = get_dataset(args, test=True)
acc_list = []
with torch.no_grad():
    with tqdm.tqdm(total=len(test_loader[0])) as progress_bar:
        for i, data in enumerate(test_loader[0]):
            problem = data[0].to(device)
            answer = data[1].to(device)
            distractors = data[2].to(device)
            acc = model.metric(problem, answer, distractors)['ACC']
            acc_list.append(acc)
            progress_bar.update(1)
print('RAVEN:', sum(acc_list) / len(acc_list))


args.dataset = 'PGM'
args.batch = 128
test_loader = get_dataset(args, test=True)
acc_list = []
with torch.no_grad():
    with tqdm.tqdm(total=len(test_loader[0])) as progress_bar:
        for i, data in enumerate(test_loader[0]):
            problem = data[0].to(device)
            answer = data[1].to(device)
            distractors = data[2].to(device)
            acc = model.metric(problem, answer, distractors)['ACC']
            acc_list.append(acc)
            progress_bar.update(1)
print('PGM:', sum(acc_list) / len(acc_list))


args.dataset = 'RAVEN'
args.batch = 128
test_loader = get_dataset(args, test=True)
acc_list = []
with torch.no_grad():
    with tqdm.tqdm(total=len(test_loader[0])) as progress_bar:
        for i, data in enumerate(test_loader[0]):
            problem = data[0].to(device)
            answer = data[1].to(device)
            distractors = data[2].to(device)
            acc = model.metric_o3_raven(problem, answer, distractors)['ACC']
            acc_list.append(acc)
            progress_bar.update(1)
print('O3-ID:', sum(acc_list) / len(acc_list))


args.dataset = 'RAVEN'
args.batch = 128
test_loader = get_dataset(args, test=True)
acc_list = []
with torch.no_grad():
    with tqdm.tqdm(total=len(test_loader[0])) as progress_bar:
        for i, data in enumerate(test_loader[0]):
            problem = data[0].to(device)
            answer = data[1].to(device)
            distractors = data[2].to(device)
            acc = model.metric_vap_raven(problem, answer, distractors)['ACC']
            acc_list.append(acc)
            progress_bar.update(1)
print('VAP-ID:', sum(acc_list) / len(acc_list))


args.dataset = 'RAVEN'
args.batch = 128
test_loader = get_dataset(args, test=True)
acc_list = []
with torch.no_grad():
    with tqdm.tqdm(total=len(test_loader[0])) as progress_bar:
        for i, data in enumerate(test_loader[0]):
            problem = data[0].to(device)
            answer = data[1].to(device)
            distractors = data[2].to(device)
            acc = model.metric_bp_raven(problem, answer, distractors)['ACC']
            acc_list.append(acc)
            progress_bar.update(1)
print('SVRT-ID:', sum(acc_list) / len(acc_list))


args.dataset = 'G1SET'
args.batch = 1
test_loader = get_dataset(args, test=True)
acc_list = []
with torch.no_grad():
    with tqdm.tqdm(total=len(test_loader[0])) as progress_bar:
        for i, data in enumerate(test_loader[0]):
            problem = data[0].to(device)
            answer = data[1].to(device)
            acc = model.metric_o3(problem, answer)['ACC']
            acc_list.append(acc)
            progress_bar.update(1)
print('G1SET:', sum(acc_list) / len(acc_list))


args.dataset = 'VAP'
args.batch = 128
test_loader = get_dataset(args, test=True)
acc_list = []
with torch.no_grad():
    with tqdm.tqdm(total=len(test_loader[0])) as progress_bar:
        for i, data in enumerate(test_loader[0]):
            problem = data[0].to(device)
            answer = data[1].to(device)
            distractors = data[2].to(device)
            acc = model.metric(problem, answer, distractors)['ACC']
            acc_list.append(acc)
            progress_bar.update(1)
print('VAP:', sum(acc_list) / len(acc_list))


args.dataset = 'SVRT'
args.batch = 128
test_loader = get_dataset(args, test=True)
acc_list = []
with torch.no_grad():
    with tqdm.tqdm(total=len(test_loader[0])) as progress_bar:
        for i, data in enumerate(test_loader[0]):
            problem = data[0].to(device)
            answer = data[1].to(device)
            distractors = data[2].to(device)
            acc = model.metric_bp(problem, answer, distractors)['ACC']
            acc_list.append(acc)
            progress_bar.update(1)
print('SVRT:', sum(acc_list) / len(acc_list))
