import numpy as np


def calculate_macro_metrics(confusion_matrix):

    num_classes = confusion_matrix.shape[0]

    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        TP = confusion_matrix[i, 0]
        FP = confusion_matrix[i, 1]
        FN = confusion_matrix[i, 2]
        TN = confusion_matrix[i, 3]

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        precision_per_class[i] = precision
        recall_per_class[i] = recall
        f1_per_class[i] = f1

    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)

    return macro_precision, macro_recall, macro_f1
