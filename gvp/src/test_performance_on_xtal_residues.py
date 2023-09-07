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
from optimal_threshold_protein_performance import determine_optimal_threshold, determine_recall_or_sensitivity

if __name__ == '__main__':
    val_label_dictionary = np.load('/project/bowmore/ameller/projects/pocket_prediction/data/val_label_dictionary.npy',
                               allow_pickle=True).item()

    # Load validation apo ids to select best epoch
    val_set_apo_ids_with_chainids = np.load('/project/bowmore/ameller/projects/'
                                            'pocket_prediction/data/val_apo_ids_with_chainids.npy')

    # Load test apo ids for evaluation
    val_l_max = max([len(l) for l in val_label_dictionary.values()])

    val_label_mask = np.zeros([len(val_label_dictionary.keys()), val_l_max], dtype=bool)
    for i, l in enumerate(val_label_dictionary.values()):
        val_label_mask[i] = np.pad(l != 2, [[0, val_l_max - len(l)]])

    val_true_labels = np.zeros([len(val_label_dictionary.keys()), val_l_max], dtype=int)
    for i, l in enumerate(val_label_dictionary.values()):
        val_true_labels[i] = np.pad(l, [[0, val_l_max - len(l)]])

    print('Balance in val set')
    print(f'Total negative residues in val: {np.sum(val_true_labels[val_label_mask] == 0)}')
    print(f'Total positive residues in val: {np.sum(val_true_labels[val_label_mask] == 1)}')

    # Load test set input
    test_label_dictionary = np.load('/project/bowmore/ameller/projects/pocket_prediction/data/test_label_dictionary.npy',
                                    allow_pickle=True).item()
    test_set_apo_ids_with_chainids = np.load('/project/bowmore/ameller/projects/'
                                             'pocket_prediction/data/test_apo_ids_with_chainids.npy')

    # Load test apo ids for evaluation
    test_l_max = max([len(l) for l in test_label_dictionary.values()])

    test_label_mask = np.zeros([len(test_label_dictionary.keys()), test_l_max], dtype=bool)
    for i, l in enumerate(test_label_dictionary.values()):
        test_label_mask[i] = np.pad(l != 2, [[0, test_l_max - len(l)]])

    test_true_labels = np.zeros([len(test_label_dictionary.keys()), test_l_max], dtype=int)
    for i, l in enumerate(test_label_dictionary.values()):
        test_true_labels[i] = np.pad(l, [[0, test_l_max - len(l)]])

    print('Balance in test set')
    print(f'Total negative residues in test: {np.sum(test_true_labels[test_label_mask] == 0)}')
    print(f'Total positive residues in test: {np.sum(test_true_labels[test_label_mask] == 1)}')


    # Create model
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100

    # get NN directories
    nn_dirs = [
        '/project/bowmanlab/ameller/gvp/task2/train-with-4-residue-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b4resis_b1proteins_20epoch_feat_method_gp-to-nearest-resi-procedure_rank_7_stride_1_window_40_pos_20_refine_feat_method_fpocket_drug_scores_max_window_40_cutoff_0.3_stride_5'
    ]

    X_val, S_val, mask_val = process_paths(val_set_apo_ids_with_chainids, use_tensors=False)
    _, S_val_lm, _ = process_paths(val_set_apo_ids_with_chainids, use_tensors=False, use_lm=True)

    X_test, S_test, mask_test = process_paths(test_set_apo_ids_with_chainids, use_tensors=False)
    # print(np.sum(test_label_mask, axis=1))
    # print(np.sum(mask_test, axis=1))

    print([(p, s1, s2) for p, s1, s2 in
           zip(test_set_apo_ids_with_chainids, np.sum(mask_test.astype(bool) & test_label_mask, axis=1), np.sum(test_label_mask, axis=1))])
    # print(np.sum(test_label_mask, axis=1))

    # _, S_val_lm, _ = process_paths(val_set_apo_ids_with_chainids, use_tensors=False, use_lm=True)


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

        val_optimal_thresholds = []

        # for epoch in tqdm(range(len(val_files))):
        for epoch in tqdm(range(3)):
            nn_path = f"{nn_dir}/{nn_id}_{str(epoch).zfill(3)}"
            if 'lm' in nn_dir:
                predictions = predict_on_xtals(model, nn_path, X_val, S_val_lm, mask_val)
            else:
                predictions = predict_on_xtals(model, nn_path, X_val, S_val, mask_val)

            y_pred = predictions[mask_val.astype(bool) & val_label_mask]
            y_true = val_true_labels[val_label_mask]

            auc_metric.update_state(y_true, y_pred)
            pr_auc_metric.update_state(y_true, y_pred)

            pr_auc.append(pr_auc_metric.result().numpy())
            auc.append(auc_metric.result().numpy())

            # determine optimal threshold for validation set
            val_optimal_thresholds.append(determine_optimal_threshold(y_pred, y_true))

            # reset AUC and PR-AUC
            pr_auc_metric.reset_state()
            auc_metric.reset_state()

        print(val_optimal_thresholds)
        print(f'Validation ROC-AUCs: {auc}')
        best_epoch = np.argmax(auc)
        print(f'best epoch is {best_epoch}')
        nn_path = f"{nn_dir}/{nn_id}_{str(best_epoch).zfill(3)}"
        print(f'using network {nn_path}')

        # Make predictions on test set
        if 'lm' in nn_dir:
            predictions = predict_on_xtals(model, nn_path, X_test, S_test_lm, mask_test)
        else:
            predictions = predict_on_xtals(model, nn_path, X_test, S_test, mask_test)

        test_y_pred = predictions[mask_test.astype(bool) & test_label_mask]
        test_y_true = test_true_labels[test_label_mask]

        np.save(f'{nn_dir}/test_y_pred.npy', test_y_pred)
        np.save(f'{nn_dir}/test_y_true.npy', test_y_true)

        lengths = np.sum(mask_test.astype(bool), axis=1)
        print(lengths)
        parsed_y_pred = [np.array(y[:l]) for y, l in zip(predictions, lengths)]
        np.save(f'{nn_dir}/test_y_pred_parsed.npy', parsed_y_pred)

        auc_metric.update_state(test_y_true, test_y_pred)
        pr_auc_metric.update_state(test_y_true, test_y_pred)

        print(f'Test AUC: {auc_metric.result().numpy()}')
        print(f'Test PR-AUC: {pr_auc_metric.result().numpy()}')

        np.save(f'{nn_dir}/test_auc.npy', auc_metric.result().numpy())
        np.save(f'{nn_dir}/test_pr_auc.npy', pr_auc_metric.result().numpy())

        # Determine protein-level performance
        # optimal_threshold = determine_optimal_threshold(test_y_pred, test_y_true)
        optimal_threshold = val_optimal_thresholds[best_epoch]
        print(optimal_threshold)

        # parse only those predictions for cryptic residues and negative residues
        parsed_y_pred = [np.array(y)[m] for y, m in zip(predictions, test_label_mask)]
        parsed_y_true = [y[m] for y, m in zip(test_true_labels, test_label_mask)]
        print(parsed_y_pred[0])
        print(parsed_y_true[0])

        protein_performance = determine_recall_or_sensitivity(parsed_y_pred, parsed_y_true, optimal_threshold)

        np.save(os.path.join(nn_dir, 'test_protein_performance.npy'), protein_performance)

        print([(p, round(v, 3)) for p, v in zip(test_label_dictionary.keys(), protein_performance)])


