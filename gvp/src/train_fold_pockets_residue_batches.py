import sys
import yaml
import tensorflow as tf
from datetime import datetime
from datasets import *
import tqdm, sys
import util, pdb
from tensorflow import keras as keras
from models import *
import os
from util import save_checkpoint, load_checkpoint
import random
import math
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# utility method for splitting lists
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# utility method for continuing trainings
def get_training_data_for_restart(model, opt, nn_dir):
    # Determine network name
    index_filenames = glob(f"{nn_dir}/*.index")
    # check if there is only one network in the folder
    if len(np.unique([os.path.basename(fn).split('_')[0] for fn in index_filenames])) == 1:
        nn_path = index_filenames[-1].split('.index')[0]
        load_checkpoint(model, opt, nn_path)
    else:
        print('multiple networks in folder')
        nn_path = index_filenames[-1].split('.index')[0]
        load_checkpoint(model, opt, nn_path)
        print(f'selecting following network id: {os.path.basename(nn_path)}')

    model_id = os.path.basename(index_filenames[-1]).split('_')[0]
    last_epoch = int(os.path.basename(index_filenames[-1]).split('_')[1].split('.')[0])

    # need to also determine best epoch, best val, and best auc
    best_epoch, best_val, best_pr_auc = 0, np.inf, 0
    val_losses = []
    train_losses = []
    for epoch in range(last_epoch):
        train_loss = np.load(os.path.join(nn_dir, f"train_loss_{epoch}.npy"))
        train_losses.append(train_loss)
        val_loss = np.load(os.path.join(nn_dir, f"val_loss_{epoch}.npy"))
        val_losses.append(val_loss)
        pr_auc = np.load(os.path.join(nn_dir, f"val_pr_auc_{epoch}.npy"))
        if val_loss < best_val:
            best_val = val_loss
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch

    return model_id, last_epoch, best_epoch, best_val, best_pr_auc, val_losses, train_losses


def make_model():
    #Emailed the lead author for what these values should be, these are good defaults.
    model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                     hidden_dim=(16, HIDDEN_DIM),
                     num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)
    return model

def main():
    # Prepare data
    trainset, valset, testset = pockets_dataset_fold(BATCH_SIZE, FILESTEM) # batch size = N proteins
    if weight_globally:
        if train_on_intermediates:
            positive_weight, negative_weight = determine_global_weights(FILESTEM, pos_thresh, pos_thresh)
        else:
            positive_weight, negative_weight = determine_global_weights(FILESTEM, pos_thresh, neg_thresh)

    # Set optimizer and make model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model = make_model()

    loop_func = loop

    if continue_previous_training:
        training_params = get_training_data_for_restart(model, optimizer, outdir)
        model_id, last_epoch, best_epoch, best_val, best_pr_auc, val_losses, train_losses = training_params
        start_epoch = last_epoch
    else:
        model_id = int(datetime.timestamp(datetime.now()))
        best_epoch, best_val, best_pr_auc = 0, np.inf, 0
        val_losses = []
        train_losses = []
        start_epoch = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        if weight_globally:
            loss, y_pred, y_true = loop_func(trainset, model, train=True, optimizer=optimizer,
                                             positive_weight=positive_weight,
                                             negative_weight=negative_weight)
        else:
            loss, y_pred, y_true = loop_func(trainset, model, train=True, optimizer=optimizer)
        train_losses.append(loss)
        print('EPOCH {} training loss: {}'.format(epoch, loss))

        # For debugging save out train predictions and true values
        np.save(os.path.join(outdir, f"train_loss_{epoch}.npy"), loss)
        np.save(os.path.join(outdir, f"train_y_pred_{epoch}.npy"), y_pred)
        np.save(os.path.join(outdir, f"train_y_true_{epoch}.npy"), y_true)

        save_checkpoint(model_path, model, optimizer, model_id, epoch)

        # Determine validation loss and performance statistics
        loss, auc, pr_auc, y_pred, y_true, meta_d = loop_func(valset, model, train=False, test=False)

        # Save out validation metrics for this epoch
        np.save(os.path.join(outdir, f"val_loss_{epoch}.npy"), loss)
        np.save(os.path.join(outdir, f"val_auc_{epoch}.npy"), auc)
        np.save(os.path.join(outdir, f"val_pr_auc_{epoch}.npy"), pr_auc)
        np.save(os.path.join(outdir, f"val_y_pred_{epoch}.npy"), y_pred)
        np.save(os.path.join(outdir, f"val_y_true_{epoch}.npy"), y_true)
        np.save(os.path.join(outdir, f"val_meta_d_{epoch}.npy"), meta_d)

        val_losses.append(loss)
        print(' EPOCH {} validation loss: {}'.format(epoch, loss))
        if loss < best_val:
            best_val = loss

        # Update best PR AUC to keep track of best model
        if pr_auc > best_pr_auc:
            best_epoch, best_pr_auc = epoch, pr_auc

    # Test with best validation loss
    print(f'Best AUC is in epoch {best_epoch}')
    path = model_path.format(str(model_id).zfill(3), str(best_epoch).zfill(3))

    # Save out training and validation losses
    np.save(f'{outdir}/cv_loss.npy', val_losses)
    np.save(f'{outdir}/train_loss.npy', train_losses)

    load_checkpoint(model, optimizer, path)

    loss, tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true, meta_d = loop_func(testset, model, train=False, test=True)
    print('EPOCH TEST {:.4f} {:.4f}'.format(loss, acc))

    # Validate performance on xtals
    predictions = predict_on_xtals(model, xtal_validation_path)
    np.save(f'{outdir}/xtal_val_set_predictions.npy', predictions)

    return loss, tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true, meta_d


def loop(dataset, model, train=False, optimizer=None, alpha=1, test=False,
         positive_weight=1, negative_weight=1):
    # if running validation or testing
    if test:
        tp_metric.reset_states()
        fp_metric.reset_states()
        tn_metric.reset_states()
        fn_metric.reset_states()
        acc_metric.reset_states()
        prec_metric.reset_states()
        recall_metric.reset_states()
        auc_metric.reset_states()
        pr_auc_metric.reset_states()
    if not train and not test:
        auc_metric.reset_states()
        pr_auc_metric.reset_states()

    losses = []
    y_pred, y_true, meta_d, targets = [], [], [], []
    batch_num = 0

    # for batch in tqdm.tqdm(dataset):
    for batch in dataset:
        X, S, y, meta, M = batch
        if train:
            if balance_classes:
                # Grab balanced set of residues
                if undersample:
                    iis = choose_balanced_inds_undersampling(y)
                if oversample:
                    iis = choose_balanced_inds_oversampling(y)
                if constant_size_balanced_sets:
                    iis = choose_balanced_inds_constant_size(y, NUMBER_RESIDUES_PER_DRAW)
            else:
                if weight_loss:
                    if weight_globally:
                        iis, weights = use_global_weights(y, positive_weight, negative_weight)
                    # determine weights at protein level
                    else:
                        iis, weights = get_weights(y)
                else:
                    iis = get_indices(y)

            # split into approximately equal sized batches
            # print('protein iis:')
            # print(len(iis), iis)
            # print('split iis:')
            num_batches = int(math.ceil(len(iis) / NUMBER_RESIDUES_PER_BATCH))
            iis_split = list(split(iis, num_batches))
            if weight_loss:
                weights_split = list(split(weights, num_batches))
            for i, iis in enumerate(iis_split):
                # print(len(iis), iis)
                with tf.GradientTape() as tape:
                    prediction = model(X, S, M, train=True, res_level=True)
                    if weight_loss:
                        y_sel = tf.gather_nd(y, indices=iis)
                        y_sel = y_sel >= pos_thresh
                        y_sel = tf.cast(y_sel, tf.float32)
                        prediction = tf.gather_nd(prediction, indices=iis)
                        # Convert tensors of shape (n,) to (1xn)
                        prediction = tf.expand_dims(prediction, 1)
                        y_sel = tf.expand_dims(y_sel, 1)
                        loss_value = loss_fn(y_sel, prediction,
                                             sample_weight=weights_split[i])
                    else:
                        y_sel = tf.gather_nd(y, indices=iis)
                        y_sel = y_sel >= pos_thresh
                        y_sel = tf.cast(y_sel, tf.float32)
                        prediction = tf.gather_nd(prediction, indices=iis)
                        loss_value = loss_fn(y_sel, prediction)

                assert(np.isfinite(float(loss_value)))
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # print('----------------')
        else:
            # test and validation sets (select all examples except for intermediate values)
            # note - can be refactored to remove unnecessary code
            prediction = model(X, S, M, train=False, res_level=True)
            iis = convert_test_targs(y)
            y = tf.gather_nd(y, indices=iis)
            y = y >= pos_thresh
            y = tf.cast(y, tf.float32)
            prediction = tf.gather_nd(prediction, indices=iis)
            loss_value = loss_fn(y, prediction)
            #to be able to identify each y value with its protein and resid
            meta_pairs = [(meta[ind[0]].numpy(), ind[1]) for ind in iis]
            meta_d.extend(meta_pairs)

        losses.append(float(loss_value))
        # if batch_num % 5 == 0:
        #     print('--------Printing loss values--------')
        #     print(loss_value)
        #     print('---------end loss values------------')
        y_pred.extend(prediction.numpy().tolist())
        y_true.extend(y.numpy().tolist())

        if test:
            tp_metric.update_state(y, prediction)
            fp_metric.update_state(y, prediction)
            tn_metric.update_state(y, prediction)
            fn_metric.update_state(y, prediction)
            acc_metric.update_state(y, prediction)
            prec_metric.update_state(y, prediction)
            recall_metric.update_state(y, prediction)
            auc_metric.update_state(y, prediction)
            pr_auc_metric.update_state(y, prediction)
        # if validation
        if not train and not test:
            auc_metric.update_state(y, prediction)
            pr_auc_metric.update_state(y, prediction)

        batch_num += 1
    if test:
        tp = tp_metric.result().numpy()
        fp = fp_metric.result().numpy()
        tn = tn_metric.result().numpy()
        fn = fn_metric.result().numpy()
        acc = acc_metric.result().numpy()
        prec = prec_metric.result().numpy()
        recall = recall_metric.result().numpy()
        auc = auc_metric.result().numpy()
        pr_auc = pr_auc_metric.result().numpy()
    # if validation
    if not train and not test:
        auc = auc_metric.result().numpy()
        pr_auc = pr_auc_metric.result().numpy()

    if train:
        return np.mean(losses), y_pred, y_true
    elif test:
        return np.mean(losses), tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true, meta_d
    # validation call
    else:
        return np.mean(losses), auc, pr_auc, y_pred, y_true, meta_d


def convert_test_targs(y):
    # Need to convert targs (volumes) to 1s and 0s but also discard
    # intermediate values
    if discard_intermediates_in_testing:
        iis_pos = [np.where(np.array(i) >= pos_thresh)[0] for i in y]
        iis_neg = [np.where(np.array(i) < neg_thresh)[0] for i in y]
        iis = []
        count = 0
        for i, j in zip(iis_pos,iis_neg):
            subset_iis = [[count,s] for s in j]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count,s] for s in i]
            for pair in subset_iis:
                iis.append(pair)
            count+=1
        return iis
    else:
        iis = [[struct_index, res_index]
               for struct_index, y_vals in enumerate(y)
               for res_index in range(len(y_vals))]
        return iis


def get_indices(y):
    if train_on_intermediates:
        iis = [[struct_index, res_index]
               for struct_index, y_vals in enumerate(y)
               for res_index in range(len(y_vals))]
    else:
        iis = [[struct_index, res_index]
               for struct_index, y_vals in enumerate(y)
               for res_index, res_example in enumerate(y_vals)
               if res_example >= pos_thresh or res_example < neg_thresh]
    return iis

def use_global_weights(y, positive_weight, negative_weight):
    if train_on_intermediates:
        iis = [[struct_index, res_index]
               for struct_index, y_vals in enumerate(y)
               for res_index in range(len(y_vals))]
        # Determine weights based on y values
        weights = [
            positive_weight if res_example >= pos_thresh
            else negative_weight
            for y_vals in y
            for res_example in y_vals
        ]
    else:
        iis = [[struct_index, res_index]
               for struct_index, y_vals in enumerate(y)
               for res_index, res_example in enumerate(y_vals)
               if res_example >= pos_thresh or res_example < neg_thresh]
        # Determine weights based on y values
        weights = [
            positive_weight if res_example >= pos_thresh
            else negative_weight
            for y_vals in y
            for res_example in y_vals
            if res_example >= pos_thresh or res_example < neg_thresh
        ]
    return iis, weights

def get_weights(y):
    iis = []
    weights = []
    for struct_index, example in enumerate(y):
        pos_residue_count = np.sum(np.array(example) >= pos_thresh)
        if train_on_intermediates:
            neg_residue_count = np.sum(np.array(example) < pos_thresh)
        else:
            neg_residue_count = np.sum(np.array(example) < neg_thresh)
        total_residue_count = pos_residue_count + neg_residue_count
        if train_on_intermediates:
            weights.extend([
                1 / pos_residue_count * (total_residue_count / 2.0)
                if y_vals >= pos_thresh
                else 1 / neg_residue_count * (total_residue_count / 2.0)
                for y_vals in example
            ])
            iis.extend([
                [struct_index, res_index]
                for res_index in range(len(example))
            ])
        else:
            weights.extend([
                1 / pos_residue_count * (total_residue_count / 2.0)
                if y_vals >= pos_thresh
                else 1 / neg_residue_count * (total_residue_count / 2.0)
                for y_vals in example
                if y_vals >= pos_thresh or y_vals < neg_thresh
            ])
            iis.extend([
                [struct_index, res_index]
                for res_index in range(len(example))
                if example[res_index] >= pos_thresh or example[res_index] < neg_thresh
            ])
        if pos_residue_count == 0 or neg_residue_count == 0:
            assert np.all([w == 1.0 for w in weights])
    return iis, weights

def choose_balanced_inds_constant_size(y, n_residues):
    '''
    We will undersample or oversample as needed to generate
    a set of indices with equal number of positive and
    negative examples. The iis will be shuffled to ensure
    that batches constructed from subsets of these indices
    are mixed. If there are only negative or positive examples,
    up to n_residues will be returned since balancing is not possible.

    n_residues : int
        specifies the amount of residues that should be returned
    '''
    pos_mask = np.array(y) >= pos_thresh
    if train_on_intermediates:
        neg_mask = np.array(y) < pos_thresh
    else:
        neg_mask = np.array(y) < neg_thresh

    positive_example_count = pos_mask.sum()
    negative_example_count = neg_mask.sum()

    # require both positive and negative example to return balanced inds
    if (positive_example_count > 0 and negative_example_count > 0):
        struct_indices, residue_indices = np.where(pos_mask)
        # creates a N x 2 list where N is the number of examples
        pos_indices = np.array([[si, ri] for si, ri in
                                zip(struct_indices, residue_indices)])
        struct_indices, residue_indices = np.where(neg_mask)
        neg_indices = np.array([[si, ri] for si, ri in
                                zip(struct_indices, residue_indices)])

        # we will target n_residues / 2 positive and negative examples
        target_size = int(n_residues / 2)

        # if we have more positive examples than the target
        # then we will undersample
        if positive_example_count > target_size:
            pos_selection = pos_indices[np.random.choice(range(positive_example_count),
                                                         target_size, replace=False)]
        # if we have fewer positive examples than the target
        # then we will oversample
        elif positive_example_count < target_size:
            pos_selection = pos_indices[np.random.choice(range(positive_example_count),
                                                         target_size)]
        else:
            pos_selection = pos_indices

        # if we have more negative examples than the target
        # then we will undersample
        if negative_example_count > target_size:
            neg_selection = neg_indices[np.random.choice(range(negative_example_count),
                                                         target_size, replace=False)]
        # if we have fewer negative examles than the target
        # then we will oversample
        elif negative_example_count < target_size:
            neg_selection = neg_indices[np.random.choice(range(negative_example_count),
                                                         target_size)]
        else:
            neg_selection = neg_indices

        selection = np.concatenate((pos_selection, neg_selection))
        # shuffle to ensure that positive and negative examples are well-mixed
        np.random.shuffle(selection)
        # assert that number of selected residues is 2 x the number of examples
        # in the majority class
        assert selection.shape[0] == n_residues
        return selection.tolist()
    # if there are only positive or only negative examples
    # return n_residue random indices
    else:
        iis = np.array([[struct_index, res_index]
                        for struct_index, y_vals in enumerate(y)
                        for res_index in range(len(y_vals))])
        # if there are more residues in this example than n_residues
        # we will take a random sample of size n_residues (undersampling)
        if len(iis) > n_residues:
            iis = iis[np.random.choice(range(len(iis)), n_residues, replace=False)]
        return iis.tolist()


def choose_balanced_inds_oversampling(y):
    pos_mask = np.array(y) >= pos_thresh
    if train_on_intermediates:
        neg_mask = np.array(y) < pos_thresh
    else:
        neg_mask = np.array(y) < neg_thresh

    positive_example_count = pos_mask.sum()
    negative_example_count = neg_mask.sum()
    if (positive_example_count > 0 and negative_example_count > 0):
        struct_indices, residue_indices = np.where(pos_mask)
        # creates a N x 2 list where N is the number of examples
        pos_indices = np.array([[si, ri] for si, ri in
                                zip(struct_indices, residue_indices)])
        struct_indices, residue_indices = np.where(neg_mask)
        neg_indices = np.array([[si, ri] for si, ri in
                                zip(struct_indices, residue_indices)])
        # combine into single selection where the number of positive
        # examples matches the number of negative examples
        # if there are m and n examples from the two classes respectively and m < n,
        # we select a random set of n examples from the minority class
        pos_selection = (pos_indices if positive_example_count >= negative_example_count
                         else pos_indices[np.random.choice(range(positive_example_count),
                                                           negative_example_count)])
        neg_selection = (neg_indices if positive_example_count <= negative_example_count
                         else neg_indices[np.random.choice(range(negative_example_count),
                                                           positive_example_count)])
        selection = np.concatenate((pos_selection, neg_selection))
        # shuffle to ensure that positive and negative examples are well-mixed
        np.random.shuffle(selection)
        # assert that number of selected residues is 2 x the number of examples
        # in the majority class
        assert selection.shape[0] == max(negative_example_count, positive_example_count) * 2
        return selection.tolist()
    # if there are no negative or positive examples in the batch return all indices
    else:
        return [[struct_index, res_index]
                for struct_index, y_vals in enumerate(y)
                for res_index in range(len(y_vals))]


def choose_balanced_inds_undersampling(y):
    pos_mask = np.array(y) >= pos_thresh
    if train_on_intermediates:
        neg_mask = np.array(y) < pos_thresh
    else:
        neg_mask = np.array(y) < neg_thresh

    positive_example_count = pos_mask.sum()
    negative_example_count = neg_mask.sum()
    if (positive_example_count > 0 and negative_example_count > 0):
        struct_indices, residue_indices = np.where(pos_mask)
        # creates a N x 2 list where N is the number of examples
        pos_indices = np.array([[si, ri] for si, ri in
                                zip(struct_indices, residue_indices)])
        struct_indices, residue_indices = np.where(neg_mask)
        neg_indices = np.array([[si, ri] for si, ri in
                                zip(struct_indices, residue_indices)])
        # combine into single selection where the number of positive
        # examples matches the number of negative examples
        # if there are m and n examples from the two classes respectively and m < n,
        # we select a random set of m examples from the majority class
        pos_selection = (pos_indices if positive_example_count <= negative_example_count
                         else pos_indices[np.random.choice(range(positive_example_count),
                                                           negative_example_count, replace=False)])
        neg_selection = (neg_indices if positive_example_count >= negative_example_count
                         else neg_indices[np.random.choice(range(negative_example_count),
                                                           positive_example_count, replace=False)])
        selection = np.concatenate((pos_selection, neg_selection))
        # shuffle to ensure that positive and negative examples are well-mixed
        np.random.shuffle(selection)

        # assert that number of selected residues is 2 x the number of examples
        # in the minority class
        assert selection.shape[0] == min(negative_example_count, positive_example_count) * 2
        return selection.tolist()
    # if there are no negative or positive examples in the batch return all indices
    else:
        return [[struct_index, res_index]
                for struct_index, y_vals in enumerate(y)
                for res_index in range(len(y_vals))]

def process_struc(strucs):
    """Takes a list of single frame md.Trajectory objects
    """

    pdbs = []
    for s in strucs:
        prot_iis = s.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = s.atom_slice(prot_iis)
        pdbs.append(prot_bb)

    B = len(strucs)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])
    print(L_max)
    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)

    for i, prot_bb in enumerate(pdbs):
        l = prot_bb.top.n_residues
        xyz = prot_bb.xyz.reshape(l, 4, 3)

        seq = [r.name for r in prot_bb.top.residues]
        S[i, :l] = np.asarray([lookup[abbrev[a]] for a in seq], dtype=np.int32)
        X[i] = np.pad(xyz, [[0,L_max-l], [0,0], [0,0]],
                      'constant', constant_values=(np.nan, ))

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)

    return X, S, mask

def predict_on_xtals(model, xtal_set_path):
    '''
    xtal_set_path : string
        path to npy file containing array of tuples
        where the first entry is a path to a crystal
        structure and the second entry is the labels
        for that xtal (where 1 indicates a cryptic
        residue)
    '''
    val_set = np.load(xtal_set_path, allow_pickle=True)
    strucs = [md.load(p[0]) for p in val_set]
    X, S, mask = process_struc(strucs)
    prediction = model(X, S, mask, train=False, res_level=True)
    return prediction


######### INPUTS ##########
## Define global variables
# from python call
yaml_filename = sys.argv[1]

with open(yaml_filename, "r") as stream:
    try:
        training_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# ------TRAINING PARAMETERS----- #
NUM_EPOCHS = training_config['NUM_EPOCHS']
# BATCH_SIZE specifies the number of proteins that are
# featurized and drawn each iteration of the training loop
BATCH_SIZE = training_config['BATCH_SIZE']
# NUMBER_RESIDUES_PER_BATCH specifies the number of residues
# that are used for an error calculation
NUMBER_RESIDUES_PER_BATCH = training_config['NUMBER_RESIDUES_PER_BATCH']

LEARNING_RATE = training_config['LEARNING_RATE']

continue_previous_training = training_config['continue_previous_training']
if continue_previous_training:
    previous_epoch_count = training_config['previous_epoch_count']

# should generally be set to False
discard_intermediates_in_testing = training_config['discard_intermediates_in_testing']
# balance positive and negative examples in each batch
balance_classes = training_config['balance_classes']
# with or without weighting
weight_loss = training_config['weight_loss']
# weight per protein or globally
# if global, weights are determined based on the number
# of positive and negative examples in the entire dataset
weight_globally = training_config['weight_globally']

# if not training with all data, do we oversample
# minority class or undersample majority class
# to maintain class balance
oversample = training_config['oversample']
undersample = training_config['undersample']
constant_size_balanced_sets = training_config['constant_size_balanced_sets']
if constant_size_balanced_sets:
    # NUMBER_RESIDUES_PER_DRAW specifies the number of residues
    # that are selected from the above number of proteins
    NUMBER_RESIDUES_PER_DRAW = training_config['NUMBER_RESIDUES_PER_DRAW']

# you must set balance classes to True if you
# wish to run with oversampling or undersampling
assert ~(balance_classes ^ (oversample or undersample or constant_size_balanced_sets))
# you cannot run with both
if balance_classes:
    assert (oversample ^ undersample) ^ constant_size_balanced_sets

train_on_intermediates = training_config['train_on_intermediates']
fold = training_config['fold']
# ----------------------------- #

# ------- LIGSITE INPUT PARAMETERS ---- #
featurization_method = training_config['featurization_method']
min_rank = training_config['min_rank']
stride = training_config['stride']
pos_thresh = training_config['pos_thresh']
neg_thresh = training_config['neg_thresh']
window = training_config['window']
# ----END LIGSITE INPUT PARAMETERS ---- #

# ------- GVP INPUT PARAMETERS ---- #
DROPOUT_RATE = training_config['DROPOUT_RATE']
HIDDEN_DIM = training_config['HIDDEN_DIM']
NUM_LAYERS = training_config['NUM_LAYERS']
# -----END GVP INPUT PARAMETERS ---- #

# ---- XTAL VALIDATION SET -----#
# file contains pdb path as well as labels
xtal_validation_path = training_config['xtal_validation_path']
# ---- END XTAL VALIDATION SET -----#
base_path = training_config['base_path']
######### END INPUTS ##############


####### CREATE OUTPUT FILENAMES #####
# base_path = "/project/bowmanlab/ameller/gvp/task1-final-folds-window-40-nearby-pv-procedure"

subdir_name = f'train-with-{NUMBER_RESIDUES_PER_BATCH}-residue-batches-'
if balance_classes:
    if oversample:
        subdir_name += 'oversample'
    elif undersample:
        subdir_name += 'undersample'
    elif constant_size_balanced_sets:
        subdir_name += f'constant-size-balanced-{NUMBER_RESIDUES_PER_DRAW}-resi-draws'
else:
    subdir_name += "no-balancing"
    if weight_loss:
        subdir_name += "-weight-loss"
        if weight_globally:
            subdir_name += "-global-weights"
        else:
            subdir_name += "-by-protein"

if train_on_intermediates:
    subdir_name += '-intermediates-in-training'
else:
    subdir_name += '-no-intermediates-in-training'

if discard_intermediates_in_testing:
    subdir_name += '-discard-intermediates-in-testing'

nn_name = (f"net_8-50_1-32_16-100_"
           f"dr_{DROPOUT_RATE}_"
           f"nl_{NUM_LAYERS}_hd_{HIDDEN_DIM}_"
           f"lr_{LEARNING_RATE}_b{NUMBER_RESIDUES_PER_BATCH}resis_"
           f"b{BATCH_SIZE}proteins_{NUM_EPOCHS}epoch_"
           f"feat_method_{featurization_method}_rank_{min_rank}_"
           f"stride_{stride}_window_{window}_pos_{pos_thresh}")

if not train_on_intermediates or discard_intermediates_in_testing:
    nn_name += f"_neg_{neg_thresh}"

outdir = f'{base_path}/{subdir_name}/{nn_name}_{fold}/'

####### END CREATE OUTPUT FILENAMES #####

if continue_previous_training:
    old_nn_name = (f"net_8-50_1-32_16-100_"
                   f"dr_{DROPOUT_RATE}_"
                   f"nl_{NUM_LAYERS}_hd_{HIDDEN_DIM}_"
                   f"lr_{LEARNING_RATE}_b{NUMBER_RESIDUES_PER_BATCH}resis_{previous_epoch_count}epoch_"
                   f"feat_method_{featurization_method}_rank_{min_rank}_"
                   f"stride_{stride}_window_{window}_pos_{pos_thresh}")
    old_outdir = f'{base_path}/{subdir_name}/{old_nn_name}_{fold}/'
    os.system(f'mv {old_outdir} {outdir}')

print(outdir)

os.makedirs(outdir, exist_ok=True)
model_path = outdir + '{}_{}'
FILESTEM = f'{featurization_method}-min-rank-{min_rank}-window-{window}-stride-{stride}-final-task1-fold-{fold}'

#### GPU INFO ####
#tf.debugging.enable_check_numerics()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
#####################

##### PERFORMANCE METRICS ########
tp_metric = keras.metrics.TruePositives(name='tp')
fp_metric = keras.metrics.FalsePositives(name='fp')
tn_metric = keras.metrics.TrueNegatives(name='tn')
fn_metric = keras.metrics.FalseNegatives(name='fn')
acc_metric = keras.metrics.BinaryAccuracy(name='accuracy')
prec_metric = keras.metrics.Precision(name='precision')
recall_metric = keras.metrics.Recall(name='recall')
auc_metric = keras.metrics.AUC(name='auc')
pr_auc_metric = keras.metrics.AUC(curve='PR', name='pr_auc')

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

##################################

######## RUN TRAINING ############
loss, tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true, meta_d = main()
#################################

####### SAVE TEST RESULTS ########
np.save(os.path.join(outdir,"test_loss.npy"), loss)
np.save(os.path.join(outdir,"test_tp.npy"), tp)
np.save(os.path.join(outdir,"test_fp.npy"), fp)
np.save(os.path.join(outdir,"test_tn.npy"), tn)
np.save(os.path.join(outdir,"test_fn.npy"), fn)
np.save(os.path.join(outdir,"test_acc.npy"), acc)
np.save(os.path.join(outdir,"test_prec.npy"), prec)
np.save(os.path.join(outdir,"test_recall.npy"), recall)
np.save(os.path.join(outdir,"test_auc.npy"), auc)
np.save(os.path.join(outdir,"test_pr_auc.npy"), pr_auc)
np.save(os.path.join(outdir,"test_y_pred.npy"), y_pred)
np.save(os.path.join(outdir,"test_y_true.npy"), y_true)
np.save(os.path.join(outdir,"test_meta_d.npy"), meta_d)
##################################

####### MOVE TRAINING YAML FILE ###
os.system(f"mv {yaml_filename} {os.path.join(outdir, 'training.yml')}")
###################################
