import string
import torch
import numpy as np

def get_char2idx_dict():
     # Prepare char_dict
    count = 0
    char_dict = {}
    labels = list(string.digits)
    labels.extend(list(string.ascii_letters))
    
    for x in labels:
        char_dict[str(x)] = count
        count += 1

    char_dict['0_deg'] = count
    count += 1
    char_dict['90_deg'] = count
    count += 1
    char_dict['180_deg'] = count
    count += 1
    char_dict['270_deg'] = count

    return char_dict

def get_idx2char_dict():
     # Prepare char_dict
    count = 0
    char_dict = {}
    labels = list(string.digits)
    labels.extend(list(string.ascii_letters))
    
    for x in labels:
        char_dict[count] = str(x)
        count += 1

    char_dict[count] = '0_deg'
    count += 1
    char_dict[count] = '90_deg'
    count += 1
    char_dict[count] = '180_deg'
    count += 1
    char_dict[count] = '270_deg'

    return char_dict

def shot_acc(preds, targets, train_labels, many_shot_thr = 100, low_shot_thr=20):
    train_class_count = []
    test_class_count = []
    class_correct = []
    
    if not isinstance(train_labels, np.ndarray):
        train_labels = np.array(train_labels)
    
    for l in torch.unique(targets):
        l = int(l)
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(targets[targets == l]))
        class_correct.append((preds[targets == l] == targets[targets == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)

def calibration(pred_labels, true_labels, confidences, num_bins=15):
    ### https://github.com/dvlab-research/MiSLAS/blob/main/utils/metric.py
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """  
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)
    
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)
    
    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)
    
    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)
            
    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)
    
    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)
    
    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }
    