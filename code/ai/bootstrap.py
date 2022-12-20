import numpy as np
import pandas as pd

from sklearn.utils import resample


def bootstrap_classes(num_iters, df_data, col_class, random_seq):
    if not isinstance(df_data, pd.DataFrame):
        raise TypeError
    if col_class not in df_data.columns:
        raise ValueError

    s_classes = df_data[col_class]

    # generate random state: first create child sequence by spawning, then draw num_iters samples
    random_states_boot = random_seq.spawn(1)[0].generate_state(num_iters)

    l_df_data_boot = []
    for random_state in random_states_boot:
        # stratified resampling with replacement and equal size
        np_data = df_data.to_numpy()
        np_boot = resample(np_data, replace=True, n_samples=None, stratify=s_classes, random_state=random_state)
        df_boot = pd.DataFrame(data=np_boot, columns=df_data.columns)
        l_df_data_boot.append(df_boot)

    return l_df_data_boot


def bootstrap_groups_classes(num_iters, df_data, col_group, col_class, random_seq):
    if not isinstance(df_data, pd.DataFrame):
        raise TypeError
    if (col_group not in df_data.columns) or (col_class not in df_data.columns):
        raise ValueError

    s_groups = df_data[col_group]
    s_classes = df_data[col_class]
    set_groups = s_groups.unique()
    set_classes = s_classes.unique()
    num_groups = set_groups.size
    num_classes = set_classes.size

    # generate random state: first create child sequence by spawning, then draw num_iters samples
    random_seqs_boot = random_seq.spawn(num_iters)

    l_df_data_boot = []
    for random_seq_boot in random_seqs_boot:
        random_states_groups = random_seq_boot.spawn(1)[0].generate_state(num_groups)
        l_np_boot = []
        for gr, random_state in zip(set_groups, random_states_groups):
            idx_gr = (s_groups == gr)
            df_gr = df_data[idx_gr].reset_index(drop=True)
            s_class_gr = s_classes[idx_gr].reset_index(drop=True)

            # for each group, perform a separate bootstrap, stratified attending to classes
            np_data_gr = df_gr.to_numpy()
            np_boot_gr = resample(np_data_gr, replace=True, n_samples=None, stratify=s_class_gr,
                                  random_state=random_state)
            l_np_boot.append(np_boot_gr)

        # merge groups
        np_boot = np.concatenate(tuple(l_np_boot), axis=0)
        df_boot = pd.DataFrame(data=np_boot, columns=df_data.columns)
        l_df_data_boot.append(df_boot)

    return l_df_data_boot
