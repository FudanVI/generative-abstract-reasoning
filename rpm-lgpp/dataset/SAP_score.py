import numpy as np
from six.moves import range
from get_drawer import get_drawer
import torch


def compute_sap(dataset, representation_function, num_train=10000, batch_size=512, n_iter=10):
    score_arr = []
    for i in range(n_iter):
        data_drawer, attr, bound = get_drawer(dataset)
        mus, ys = generate_batch_factor_code(data_drawer, attr, bound, representation_function, num_train, batch_size)
        score_arr.append(_compute_sap(mus, ys)['SAP_score'].item())
        print('%d/%d' % (i + 1, n_iter), end='\r')
    var, mean = torch.var_mean(torch.tensor(score_arr).float())
    return mean.item(), var.sqrt().item()


def _compute_sap(mus, ys):
    score_matrix = compute_score_matrix(mus, ys)
    assert score_matrix.shape[0] == mus.shape[0]
    assert score_matrix.shape[1] == ys.shape[0]
    scores_dict = {
        'SAP_score': compute_avg_diff_top_two(score_matrix)
    }
    return scores_dict


def compute_score_matrix(mus, ys):
    num_latents = mus.shape[0]
    num_factors = ys.shape[0]
    score_matrix = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
        for j in range(num_factors):
            mu_i = mus[i, :]
            y_j = ys[j, :]
            # Attribute is considered continuous.
            cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
            cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
            var_mu = cov_mu_i_y_j[0, 0]
            var_y = cov_mu_i_y_j[1, 1]
            if var_mu > 1e-12:
                score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
            else:
                score_matrix[i, j] = 0.
    return score_matrix


def compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])


def generate_batch_factor_code(drawer, attr, bound, represent_fn, num_points, batch_size):
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors = np.random.rand(batch_size, len(attr))
        for j, ele in enumerate(attr):
            current_factors[:, j] = current_factors[:, j] * (bound[ele][1] - bound[ele][0]) + bound[ele][0]
        pics = []
        for j in range(current_factors.shape[0]):
            pic = drawer.draw({attr[k]: current_factors[j][k] for k in range(len(attr))})
            pics.append(pic[None, None, ...])
        current_observations = np.concatenate(pics, axis=0)

        if i == 0:
            factors = current_factors
            representations = represent_fn(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations, represent_fn(current_observations)))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)
