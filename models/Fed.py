import copy


def FedAvg_noniid(w, dict_len):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * dict_len[0]
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dict_len[i]
            # w_avg[k] += w[i][k]
        # w_avg[k] = w_avg[k] / len(w)
        w_avg[k] = w_avg[k] / sum(dict_len)
    return w_avg


def FedWeightAvg(backbone_w_locals, linear_w_locals, dict_len):
    backbone_w_avg = FedAvg_noniid(backbone_w_locals, dict_len)
    linear_w_avg = FedAvg_noniid(linear_w_locals, dict_len)
    return backbone_w_avg, linear_w_avg
