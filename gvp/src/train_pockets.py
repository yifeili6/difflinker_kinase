import sys
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def make_model():
    #Emailed the lead author for what these values should be, these are good defaults.
    model = MQAModel(node_features=(8,50), edge_features=(1,32), hidden_dim=(16,100),
                     num_layers=NUM_LAYERS, dropout=DROPOUT_RATE)
    return model

def main():

    trainset, valset, testset = pockets_dataset(BATCH_SIZE, FILESTEM) # batch size = N proteins
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model = make_model()

    model_id = int(datetime.timestamp(datetime.now()))

    loop_func = loop
    best_epoch, best_val = 0, np.inf

    # Calculate validation loss before any training
    val_losses = []
    loss, auc, pr_auc, y_pred, y_true, meta_d = loop_func(valset, model, train=False, test=False)
    val_losses.append(loss)
    # Save out validation metrics before training
    # np.save(os.path.join(outdir, f"val_loss_pre_train.npy"), loss)
    # np.save(os.path.join(outdir, f"val_auc_pre_train.npy"), auc)
    # np.save(os.path.join(outdir, f"val_pr_auc_pre_train.npy"), pr_auc)
    # np.save(os.path.join(outdir, f"val_y_pred_pre_train.npy"), y_pred)
    # np.save(os.path.join(outdir, f"val_y_true_pre_train.npy"), y_true)
    # np.save(os.path.join(outdir, f"val_meta_d_pre_train.npy"), meta_d)

    train_losses = []

    for epoch in range(NUM_EPOCHS):
        loss = loop_func(trainset, model, train=True, optimizer=optimizer)
        train_losses.append(loss)

        print('EPOCH {} training loss: {}'.format(epoch, loss))
        save_checkpoint(model_path, model, optimizer, model_id, epoch)
        print('EPOCH {} TRAIN {:.4f}'.format(epoch, loss))
        #util.save_confusion(confusion)
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
            #Could play with this parameter here. Instead of saving best NN based on loss
            #we could save it based on precision/auc/recall/etc. 
            best_epoch, best_val = epoch, loss
        print('EPOCH {} VAL {:.4f}'.format(epoch, loss))
        #util.save_confusion(confusion)

    # Test with best validation loss
    path = model_path.format(str(model_id).zfill(3), str(best_epoch).zfill(3))

    # Save out training and validation losses
    np.save(f'{outdir}/cv_loss.npy', val_losses)
    np.save(f'{outdir}/train_loss.npy', train_losses)

    load_checkpoint(model, optimizer, path)

    loss, tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true, meta_d = loop_func(testset, model, train=False, test=True)
    print('EPOCH TEST {:.4f} {:.4f}'.format(loss, acc))
    #util.save_confusion(confusion)
    return loss, tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true, meta_d


def loop(dataset, model, train=False, optimizer=None, alpha=1, test=False):
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
            with tf.GradientTape() as tape:
                prediction = model(X, S, M, train=True, res_level=True)
                #Grab balanced set of residues
                iis = choose_balanced_inds(y)
                y = tf.gather_nd(y, indices=iis)
                # print('Targets should be balanced: ', y)
                y = y >= pos_thresh
                y = tf.cast(y, tf.float32)
                prediction = tf.gather_nd(prediction, indices=iis)
                loss_value = loss_fn(y, prediction)
        else:
            # test and validation sets (select all examples except for intermediate values)    
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
        if train:
            assert(np.isfinite(float(loss_value)))
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

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
        return np.mean(losses)
    elif test:
        return np.mean(losses), tp, fp, tn, fn, acc, prec, recall, auc, pr_auc, y_pred, y_true, meta_d
    # validation call
    else:
        return np.mean(losses), auc, pr_auc, y_pred, y_true, meta_d


def convert_test_targs(y):
    # Need to convert targs (volumes) to 1s and 0s but also discard
    # intermediate values
    iis_pos = [np.where(np.array(i) >= pos_thresh)[0] for i in y]
    if discard_intermediates:
        iis_neg = [np.where(np.array(i) < neg_thresh)[0] for i in y]
    else:
        iis_neg = [np.where(np.array(i) < pos_thresh)[0] for i in y]
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

def choose_balanced_inds(y):
    iis_pos = [np.where(np.array(i) >= pos_thresh)[0] for i in y]
    iis_neg = [np.where(np.array(i) < neg_thresh)[0] for i in y]
    count = 0
    iis = []
    for i, j in zip(iis_pos, iis_neg):
        # print(len(i),len(j))
        if len(i) < len(j):
            subset = np.random.choice(j, len(i), replace=False)
            subset_iis = [[count,s] for s in subset]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count,s] for s in i]
            for pair in subset_iis:
                iis.append(pair)
        elif len(j) < len(i):
            subset = np.random.choice(i, len(j), replace=False)
            subset_iis = [[count,s] for s in subset]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count,s] for s in j]
            for pair in subset_iis:
                iis.append(pair)
        else:
            subset_iis = [[count,s] for s in j]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count,s] for s in i]
            for pair in subset_iis:
                iis.append(pair)

        count+=1
    # select a random residue when there are no positive examples (or negative)
    # for a given structure
    if len(iis) == 0:
        # if there are multiple structures in batch simply use the first
        iis = [[0, random.choice(range(len(y[0])))]]
        print(f'selected random resid {iis[0][1]}')
        # iis = [[0, 0]]

    return iis


######### INPUTS ##########
## Define global variables

NUM_EPOCHS = 20
pos_thresh = 116

neg_thresh = 20
# neg_thresh = int(pos_thresh * 0.8)
# neg_thresh = 116

BATCH_SIZE = 1
LEARNING_RATE = 0.00029
DROPOUT_RATE = 0.13225
NUM_LAYERS = 4
window = int(sys.argv[1])
#####

min_rank = 7
test_string = 'TEM-1MY0-1BSQ-nsp5-il6-2OFV'
stride = 1
featurization_method = 'nearby-pv-procedure'

discard_intermediates = False

if discard_intermediates:
    outdir = (f"/project/bowmanlab/ameller/gvp/net_8-50_1-32_16-100_"
              f"nl_{NUM_LAYERS}_lr_{LEARNING_RATE}_dr_{DROPOUT_RATE}_"
              f"b{BATCH_SIZE}_{NUM_EPOCHS}epoch_"
              f"feat_method_{featurization_method}_rank_{min_rank}_"
              f"stride_{stride}_"
              f"pos{pos_thresh}_neg{neg_thresh}_window_{window}ns_"
              f"test_{test_string}_discard_intermediates/")
else:
    outdir = (f"/project/bowmanlab/ameller/gvp/net_8-50_1-32_16-100_"
              f"nl_{NUM_LAYERS}_lr_{LEARNING_RATE}_dr_{DROPOUT_RATE}_"
              f"b{BATCH_SIZE}_{NUM_EPOCHS}epoch_"
              f"feat_method_{featurization_method}_rank_{min_rank}_"
              f"stride_{stride}_"
              f"pos{pos_thresh}_neg{neg_thresh}_window_{window}ns_"
              f"test_{test_string}/")


###########################

print(outdir)

os.makedirs(outdir, exist_ok=True)
model_path = outdir + '{}_{}'
FILESTEM = f'{featurization_method}-min-rank-{min_rank}-window-{window}-stride-{stride}-test-{test_string}'

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
