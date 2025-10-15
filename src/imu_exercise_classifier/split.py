import numpy as np
import warnings
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_session_split(data, labels, train_ratio=0.8, seed=42):
    labels = np.asarray(labels)
    idx_all = np.arange(len(labels))
    cls, counts = np.unique(labels, return_counts=True)
    rare = {c for c, k in zip(cls, counts) if k < 2}

    forced_train = np.array([i for i in idx_all if labels[i] in rare], dtype=int)
    rest_idx = np.array([i for i in idx_all if labels[i] not in rare], dtype=int)

    target_train = int(train_ratio * len(labels))
    remaining_train_needed = max(0, target_train - len(forced_train))
    remaining_total = len(rest_idx)

    if remaining_total == 0 or remaining_train_needed == 0:
        train_idx = forced_train
        test_idx = np.setdiff1d(idx_all, train_idx)
        return np.sort(train_idx), np.sort(test_idx)

    adjusted_train_ratio = min(1.0, remaining_train_needed / remaining_total)
    adjusted_test_size = 1.0 - adjusted_train_ratio

    sss = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_test_size, random_state=seed)
    (rest_train_sel, rest_test_sel), = sss.split(rest_idx, labels[rest_idx])

    train_idx = np.concatenate([forced_train, rest_idx[rest_train_sel]])
    test_idx  = rest_idx[rest_test_sel]

    present_all = set(np.unique(labels))
    present_train = set(np.unique(labels[train_idx]))
    missing_in_train = present_all - present_train
    if missing_in_train:
        warnings.warn(f"Moving one session per missing class to TRAIN: {missing_in_train}")
        for cls in list(missing_in_train):
            cand = np.where(labels[test_idx] == cls)[0]
            if len(cand) > 0:
                move = test_idx[cand[0]]
                train_idx = np.append(train_idx, move)
                test_idx = np.delete(test_idx, cand[0])

    return np.sort(train_idx), np.sort(test_idx)
