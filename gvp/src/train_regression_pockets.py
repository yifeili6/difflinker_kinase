import sys
import tensorflow as tf
import tensorflow_addons as tfa
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
                     num_layers=NUM_LAYERS, dropout=DROPOUT_RATE, regression=True)
    return model

def main():

    trainset, valset, testset = pockets_dataset(BATCH_SIZE, FILESTEM) # batch size = N proteins
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model = make_model()

    model_id = int(datetime.timestamp(datetime.now()))

    loop_func = loop
    best_epoch, best_val = 0, np.inf

    val_losses = []
    train_losses = []

    for epoch in range(NUM_EPOCHS):
        loss = loop_func(trainset, model, train=True, optimizer=optimizer)
        train_losses.append(loss)

        print('EPOCH {} training loss: {}'.format(epoch, loss))
        save_checkpoint(model_path, model, optimizer, model_id, epoch)
        print('EPOCH {} TRAIN {:.4f}'.format(epoch, loss))

        loss, r2, y_pred, y_true, meta_d = loop_func(valset, model, train=False, test=False)

        # Save out validation metrics for this epoch
        np.save(os.path.join(outdir, f"val_r_squared_{epoch}.npy"), r2)
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

    loss, r2, y_pred, y_true, meta_d = loop_func(testset, model, train=False, test=True)
    print('EPOCH TEST {:.4f} {:.4f}'.format(loss, acc))
    #util.save_confusion(confusion)
    return loss, r2, y_pred, y_true, meta_d


def loop(dataset, model, train=False, optimizer=None, alpha=1, test=False):
    if not train:
        r2_metric.reset_states()

    losses = []
    y_pred, y_true, meta_d, targets = [], [], [], []

    # for batch in tqdm.tqdm(dataset):
    for batch in dataset:
        X, S, y, meta, M = batch
        if train:
            with tf.GradientTape() as tape:
                prediction = model(X, S, M, train=True, res_level=True)
                iis = choose_regress_inds(y)
                y = tf.gather_nd(y, indices=iis)
                prediction = tf.gather_nd(prediction, indices=iis)
                loss_value = loss_fn(y, prediction)
        else:
            # test and validation sets (select all examples except for intermediate values)    
            prediction = model(X, S, M, train=False, res_level=True)
            iis = choose_regress_inds(y)
            y = tf.gather_nd(y, indices=iis)
            prediction = tf.gather_nd(prediction, indices=iis)
            loss_value = loss_fn(y, prediction)
            # to be able to identify each y value with its protein and resid
            meta_pairs = [(meta[ind[0]].numpy(), ind[1]) for ind in iis]
            meta_d.extend(meta_pairs)
        if train:
            assert(np.isfinite(float(loss_value)))
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        losses.append(float(loss_value))

        y_pred.extend(prediction.numpy().tolist())
        y_true.extend(y.numpy().tolist())

        if not train:
            r2_metric.update_state(y, prediction)

    if not train:
        r2 = r2_metric.result().numpy()

    if train:
        return np.mean(losses)
    else:
        return np.mean(losses), r2, y_pred, y_true, meta_d

def choose_regress_inds(y):
    iis_list = [np.where(np.array(i) > -1)[0] for i in y]
    paired_iis = []
    count = 0
    for iis in iis_list:
        for i in iis:
            paired_iis.append([count,i])
    return paired_iis


######### INPUTS ##########
## Define global variables

NUM_EPOCHS = 250

BATCH_SIZE = 1
LEARNING_RATE = 0.00001
DROPOUT_RATE = 0.13225
NUM_LAYERS = 4
window = int(sys.argv[1])
#####

min_rank = 7
test_string = 'TEM-1MY0-1BSQ-nsp5-il6-2OFV'
stride = 1
featurization_method = 'nearby-pv-procedure'

outdir = (f"/project/bowmanlab/ameller/gvp/net_8-50_1-32_16-100_"
          f"nl_{NUM_LAYERS}_lr_{LEARNING_RATE}_dr_{DROPOUT_RATE}_"
          f"b{BATCH_SIZE}_{NUM_EPOCHS}epoch_"
          f"feat_method_{featurization_method}_rank_{min_rank}_"
          f"stride_{stride}_window_{window}ns_regression_"
          f"test_{test_string}_loss_huber_delta_50/")

loss_fn = tf.keras.losses.Huber(delta=50.0)
# loss_fn = tf.keras.losses.MeanSquaredError()


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
r2_metric = tfa.metrics.r_square.RSquare()
##################################

######## RUN TRAINING ############
loss, r2, y_pred, y_true, meta_d = main()
#################################

####### SAVE TEST RESULTS ########
np.save(os.path.join(outdir,"test_loss.npy"), loss)
np.save(os.path.join(outdir,"test_r_squared.npy"), r2)
np.save(os.path.join(outdir,"test_y_pred.npy"), y_pred)
np.save(os.path.join(outdir,"test_y_true.npy"), y_true)
np.save(os.path.join(outdir,"test_meta_d.npy"), meta_d)
##################################
