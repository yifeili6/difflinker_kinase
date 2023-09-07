from models import MQAModel
from util import load_checkpoint
import tensorflow as tf
import mdtraj as md
import numpy as np
import os
from glob import glob
from tensorflow import keras as keras
from tqdm import tqdm
from validate_performance_on_xtals import process_strucs, process_paths, predict_on_xtals


if __name__ == '__main__':
    label_dictionary = np.load('/project/bowmore/ameller/projects/pocket_prediction/data/val_label_dictionary.npy',
                               allow_pickle=True).item()
    val_set_apo_ids = np.load('/project/bowmore/ameller/projects/pocket_prediction/data/val_apo_ids.npy')
    val_set_apo_ids_with_chainids = np.load('/project/bowmore/ameller/projects/'
                                            'pocket_prediction/data/val_apo_ids_with_chainids.npy')

    l_max = max([len(l) for l in label_dictionary.values()])

    label_mask = np.zeros([len(label_dictionary.keys()), l_max], dtype=bool)
    for i, l in enumerate(label_dictionary.values()):
        label_mask[i] = np.pad(l != 2, [[0, l_max - len(l)]])

    true_labels = np.zeros([len(label_dictionary.keys()), l_max], dtype=int)
    for i, l in enumerate(label_dictionary.values()):
        true_labels[i] = np.pad(l, [[0, l_max - len(l)]])

    print(np.sum(true_labels[label_mask] == 0))
    print(np.sum(true_labels[label_mask] == 1))

    # cryptosite exclusions
    cryptosite_exclude = ['1rrg', '2bu8', '2ohg', '2wgb', '1ok8', '1k3f']
    cryptosite_mask = np.zeros([len(label_dictionary.keys()), l_max], dtype=bool)
    for i, (p, l) in enumerate(label_dictionary.items()):
        if p not in cryptosite_exclude:
            cryptosite_mask[i] = np.pad(l != 2, [[0, l_max - len(l)]])
        else:
            print(f'excluding {p} from cryptosite analysis')

    # Create model
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100

    # get NN directories
    nn_dirs = glob('/project/bowmanlab/ameller/gvp/task2/*/*')

    X, S, mask = process_paths(val_set_apo_ids_with_chainids, use_tensors=False)
    _, S_lm, _ = process_paths(val_set_apo_ids_with_chainids, use_tensors=False, use_lm=True)


    # nn_dirs = [
    #     "/project/bowmanlab/ameller/gvp/task2/train-with-4-residue-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b4resis_b1proteins_30epoch_feat_method_nearby-pv-procedure_rank_7_stride_1_window_40_pos_87",
    #     "/project/bowmanlab/ameller/gvp/task2/train-with-4-residue-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b4resis_b1proteins_20epoch_feat_method_gp-to-nearest-resi-procedure_rank_7_stride_1_window_40_pos_20",
    #     "/project/bowmanlab/ameller/gvp/task2/train-with-4-residue-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b4resis_b1proteins_20epoch_feat_method_gp-to-nearest-resi-procedure_rank_7_stride_1_window_40_pos_20_pretrained"
    #     "/project/bowmanlab/ameller/gvp/task2/train-with-4-residue-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b4resis_b1proteins_20epoch_feat_method_gp-to-nearest-resi-procedure_rank_7_stride_1_window_40_pos_30",
    #     "/project/bowmanlab/ameller/gvp/task2/train-with-4-residue-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b4resis_b1proteins_20epoch_feat_method_gp-to-nearest-resi-procedure_rank_7_stride_1_window_40_pos_20_refine_feat_method_fpocket_drug_scores_max_window_40_cutoff_0.3_stride_5",
    #     "/project/bowmanlab/ameller/gvp/task2/train-with-4-residue-batches-constant-size-balanced-640-resi-draws-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b4resis_b4proteins_30epoch_feat_method_nearby-pv-procedure_rank_7_stride_1_window_40_pos_87"
    # ]

    for nn_dir in nn_dirs:
        # if sidechain is in name, need to use tensors
        if 'sidechain' in nn_dir:
            continue

        # determine number of compeleted epochs
        val_files = glob(f"{nn_dir}/val_pr_auc_*.npy")

        if len(val_files) == 0:
            continue

        # Determine network name
        index_filenames = glob(f"{nn_dir}/*.index")
        nn_id = os.path.basename(index_filenames[0]).split('_')[0]

        if "lm" in nn_dir:
            if 'squeeze' in nn_dir:
                model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                                 hidden_dim=(16, HIDDEN_DIM),
                                 num_layers=NUM_LAYERS, dropout=DROPOUT_RATE,
                                 use_lm=True, squeeze_lm=True)
            else:
                model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                                 hidden_dim=(16, HIDDEN_DIM),
                                 num_layers=NUM_LAYERS, dropout=DROPOUT_RATE,
                                 use_lm=True, squeeze_lm=False)
        elif "pretrain" in nn_dir:
            model = MQAModel(node_features=(8, 100), edge_features=(1,32),
                             hidden_dim=(16,100), dropout=1e-3)
        else:
            model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                             hidden_dim=(16, HIDDEN_DIM),
                             num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)

        # Determine which network to use (i.e. epoch with best AUC)
        pr_auc = []
        auc = []
        auc_metric = keras.metrics.AUC(name='auc')
        pr_auc_metric = keras.metrics.AUC(curve='PR', name='pr_auc')

        for epoch in tqdm(range(len(val_files))):
            nn_path = f"{nn_dir}/{nn_id}_{str(epoch).zfill(3)}"
            if 'lm' in nn_dir:
                predictions = predict_on_xtals(model, nn_path, X, S_lm, mask)
            else:
                predictions = predict_on_xtals(model, nn_path, X, S, mask)

            y_pred = predictions[mask.astype(bool) & label_mask]
            y_true = true_labels[label_mask]

            auc_metric.update_state(y_true, y_pred)
            pr_auc_metric.update_state(y_true, y_pred)

            np.save(os.path.join(nn_dir, f"val_new_auc_{epoch}.npy"), auc_metric.result().numpy())
            np.save(os.path.join(nn_dir, f"val_new_pr_auc_{epoch}.npy"), pr_auc_metric.result().numpy())
            np.save(os.path.join(nn_dir, f"val_new_y_pred_{epoch}.npy"), y_pred)
            np.save(os.path.join(nn_dir, f"val_new_y_true_{epoch}.npy"), y_true)

            pr_auc.append(pr_auc_metric.result().numpy())
            auc.append(auc_metric.result().numpy())

            # reset AUC and PR-AUC
            pr_auc_metric.reset_state()
            auc_metric.reset_state()

        print(auc)
        best_epoch = np.argmax(auc)
        nn_path = f"{nn_dir}/{nn_id}_{str(best_epoch).zfill(3)}"

        if 'lm' in nn_dir:
            predictions = predict_on_xtals(model, nn_path, X, S_lm, mask)
        else:
            predictions = predict_on_xtals(model, nn_path, X, S, mask)

        np.save(f'{nn_dir}/new_val_set_y_pred_best_epoch.npy', predictions)

        # find performance for cryptosite subset
        y_pred_subset = predictions[mask.astype(bool) & cryptosite_mask]
        y_true_subset = true_labels[cryptosite_mask]

        auc_metric.update_state(y_true_subset, y_pred_subset)
        pr_auc_metric.update_state(y_true_subset, y_pred_subset)

        np.save(f'{nn_dir}/new_val_set_cryptosite_subset_auc.npy', auc_metric.result().numpy())
        np.save(f'{nn_dir}/new_val_set_cryptosite_subset_pr_auc.npy', pr_auc_metric.result().numpy())


