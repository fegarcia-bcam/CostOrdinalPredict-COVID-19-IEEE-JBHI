import pandas as pd

from sklearn.model_selection import StratifiedKFold


def stratify_classes(df_data, feat_class, n_folds, shuffle, random_seq):
    if not isinstance(df_data, pd.DataFrame):
        raise TypeError
    if feat_class not in df_data.columns:
        raise ValueError

    s_classes = df_data[feat_class]

    l_df_train = []
    l_df_test = []

    # generate random state: first create child sequence by spawning, then draw a sample
    random_state = random_seq.spawn(1)[0].generate_state(1)[0]

    # stratification by classes
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    for train_idx, test_idx in skf.split(df_data, s_classes):
        df_train_fold = df_data.loc[train_idx].reset_index(drop=True)
        df_test_fold = df_data.loc[test_idx].reset_index(drop=True)
        # samples within fold are not shuffled by sklearn
        l_df_train.append(df_train_fold)
        l_df_test.append(df_test_fold)

    return l_df_train, l_df_test


def stratify_groups_classes(df_data, feat_group, feat_class, n_folds, shuffle, random_seq):
    if not isinstance(df_data, pd.DataFrame):
        raise TypeError
    if (feat_group not in df_data.columns) or (feat_class not in df_data.columns):
        raise ValueError

    s_groups = df_data[feat_group]
    s_classes = df_data[feat_class]
    set_groups = s_groups.unique()
    set_classes = s_classes.unique()
    num_groups = set_groups.size
    num_classes = set_classes.size

    # generate random states: first create child sequence by spawning, then draw num_groups samples
    random_states_groups = random_seq.spawn(1)[0].generate_state(num_groups)

    # split by groups
    l_df_train = [[] for _ in range(n_folds)]
    l_df_test = [[] for _ in range(n_folds)]
    for gr, random_state in zip(set_groups, random_states_groups):
        idx_gr = (s_groups == gr)
        df_gr = df_data[idx_gr].reset_index(drop=True)
        s_class_gr = s_classes[idx_gr].reset_index(drop=True)

        # for each group, perform a separate stratification attending to classes
        skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        for nf, (train_idx, test_idx) in enumerate(skf.split(df_gr, s_class_gr)):
            df_train_gr = df_gr.loc[train_idx].reset_index(drop=True)
            df_test_gr = df_gr.loc[test_idx].reset_index(drop=True)
            # samples within fold are not shuffled by sklearn
            l_df_train[nf].append(df_train_gr)
            l_df_test[nf].append(df_test_gr)

    # merge groups per fold
    for nf in range(n_folds):
        df_train_fold = pd.concat(l_df_train[nf], ignore_index=True)
        df_test_fold = pd.concat(l_df_test[nf], ignore_index=True)
        l_df_train[nf] = df_train_fold
        l_df_test[nf] = df_test_fold

    return l_df_train, l_df_test
