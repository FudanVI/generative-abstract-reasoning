import numpy as np
from get_drawer import get_drawer
import random
import torch


def compute_factor_vae(dataset, representation_function,  batch_size=512, num_train=1000, num_eval=500,
                       num_variance_estimate=1000, n_iter=10):

    scores_arr = []
    for i in range(n_iter):
        data_drawer, attr, bound = get_drawer(dataset)
        global_variances = compute_variances(data_drawer, attr, bound, representation_function, num_variance_estimate)
        active_dims = prune_dims(global_variances)
        scores_dict = {}

        if not active_dims.any():
            scores_dict["train_accuracy"] = 0.
            scores_dict["eval_accuracy"] = 0.
            scores_dict["num_active_dims"] = 0
            continue

        training_votes = generate_batch(data_drawer, attr, bound, representation_function, batch_size,
                                        num_train, global_variances, active_dims)
        classifier = np.argmax(training_votes, axis=0)
        other_index = np.arange(training_votes.shape[1])

        train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)

        eval_votes = generate_batch(data_drawer, attr, bound, representation_function, batch_size,
                                    num_eval, global_variances, active_dims)

        eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)
        scores_dict["train_accuracy"] = train_accuracy
        scores_dict["eval_accuracy"] = eval_accuracy
        scores_dict["num_active_dims"] = len(active_dims)
        scores_arr.append(scores_dict["eval_accuracy"].item())
        print('%d/%d' % (i + 1, n_iter), end='\r')
    var, mean = torch.var_mean(torch.tensor(scores_arr).float())
    return mean.item(), var.sqrt().item()


def prune_dims(variances, threshold=0.):
    scale_z = np.sqrt(variances)
    return scale_z >= threshold


def compute_variances(drawer, attr, bound, represent_fn, batch_size):
    factors = np.random.rand(batch_size, len(attr))
    for i, ele in enumerate(attr):
        factors[:, i] = factors[:, i] * (bound[ele][1] - bound[ele][0]) + bound[ele][0]
    pics = []
    for i in range(factors.shape[0]):
        pic = drawer.draw({attr[k]: factors[i][k] for k in range(len(attr))})
        pics.append(pic[None, None, ...])
    observations = np.concatenate(pics, axis=0)
    representations = represent_fn(observations)
    assert representations.shape[0] == batch_size
    return np.var(representations, axis=0, ddof=1)


def generate_sample(drawer, attr, bound, represent_fn, batch_size, global_var, active_dims):
    factor_index = random.randint(0, len(attr) - 1)
    factors = np.random.rand(batch_size, len(attr))
    for i, ele in enumerate(attr):
        factors[:, i] = factors[:, i] * (bound[ele][1] - bound[ele][0]) + bound[ele][0]
    factors[:, factor_index] = factors[0, factor_index]
    pics = []
    for i in range(factors.shape[0]):
        pic = drawer.draw({attr[k]: factors[i][k] for k in range(len(attr))})
        pics.append(pic[None, None, ...])
    observations = np.concatenate(pics, axis=0)
    representations = represent_fn(observations)
    # representations = np.random.rand(batch_size, representations.shape[1])
    local_var = np.var(representations, axis=0, ddof=1)
    argmin = np.argmin(local_var[active_dims] / global_var[active_dims])
    return factor_index, argmin


def generate_batch(drawer, attr, bound, represent_fn, batch_size, num_points, global_var, active_dims):
    votes = np.zeros((len(attr), global_var.shape[0]), dtype=np.int64)
    for _ in range(num_points):
        factor_index, argmin = generate_sample(drawer, attr, bound, represent_fn, batch_size, global_var, active_dims)
        votes[factor_index, argmin] += 1
    return votes
