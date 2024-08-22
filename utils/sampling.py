import numpy as np


def cifar_iid(dataset, num_users):
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def non_iid_dirichlet_sampling(dataset, num_classes, num_users, seed, alpha_dirichlet):
    np.random.seed(seed)
    p = 1
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))
    n_classes_per_client = np.sum(Phi, axis=1)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client == 0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j] == 1)[0]) for j in range(num_classes)]
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {i: set() for i in range(num_users)}
    user_class_counts = {i: {class_i: 0 for class_i in range(num_classes)} for i in range(num_users)}

    for class_i in range(num_classes):
        all_idxs = np.where(np.array(dataset.targets) == class_i)[0]
        if len(all_idxs) == 0:
            continue
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

        for client_k in Psi[class_i]:
            client_idxs = all_idxs[assignment == client_k]
            if client_k in dict_users:
                dict_users[client_k] = dict_users[client_k] | set(client_idxs)
            else:
                dict_users[client_k] = set(client_idxs)
            user_class_counts[client_k][class_i] += len(client_idxs)

    for user in dict_users:
        dict_users[user] = list(dict_users[user])

    return dict_users, user_class_counts
