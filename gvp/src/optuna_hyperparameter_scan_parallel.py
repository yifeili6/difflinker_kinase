import os

import threading
import random
import mlflow
import optuna
import tensorflow as tf
from tensorflow import keras as keras

from datasets import *
from models import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

NUM_EPOCHS = 20
pos_thresh = 20
neg_thresh = 5

## Vary throughout task 1 training ##
BATCH_SIZE = 1
window = 20

# Input data specs
min_rank = 6
test_string = 'TEM-1MY0-1BSQ-nsp5-il6'
featurization_method = 'gp-to-nearest-resi'
stride = 1
FILESTEM = f'{featurization_method}-min-rank-{min_rank}-window-{window}-stride-{stride}-test-{test_string}'
# Output data directory
outdir = f"/project/bowmanlab/ameller/gvp/optuna/{FILESTEM}"

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def make_model(dropout_rate, num_layers, hidden_dim):
    model = MQAModel(
        node_features=(8, 50),
        edge_features=(1, 32),
        hidden_dim=hidden_dim,
        dropout=dropout_rate,
        num_layers=num_layers,
    )
    return model


def main_cv(learning_rate, dropout_rate, num_layers, hidden_layers, cv_fold):

    trainset, valset, testset = pockets_dataset(BATCH_SIZE, FILESTEM, cv_fold)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = make_model(
        dropout_rate=dropout_rate, num_layers=num_layers, hidden_dim=(16, hidden_layers)
    )

    loop_func = loop
    best_epoch, best_val = 0, np.inf
    val_losses = []

    # Maintain ROC AUC, PR AUC metrics throughout training
    auc_metric = keras.metrics.AUC(name='auc')
    pr_auc_metric = keras.metrics.AUC(curve='PR', name='pr_auc')

    for epoch in range(NUM_EPOCHS):
        loss, y_pred, y_true, meta_d = loop_func(trainset, model, train=True, optimizer=optimizer)
        print("CV FOLD {} EPOCH {} training loss: {:.4f}".format(cv_fold, epoch, loss))
        loss, y_pred, y_true, meta_d = loop_func(valset, model, train=False, val=False)
        val_losses.append(loss)
        print("CV FOLD {} EPOCH {} validation loss: {:.4f}".format(cv_fold, epoch, loss))
        if loss < best_val:
            best_epoch, best_val = epoch, loss

        # Save out validation metrics for each epoch
        auc_metric.update_state(y_true, y_pred)
        pr_auc_metric.update_state(y_true, y_pred)

        np.save(os.path.join(outdir,
                             f"val_cv_fold_{cv_fold}_lr_{learning_rate:.5f}_"
                             f"dr_{dropout_rate:.5f}_nl_{num_layers}_"
                             f"hl_{hidden_layers}_loss_{epoch}.npy"),
                loss)
        np.save(os.path.join(outdir,
                             f"val_cv_fold_{cv_fold}_lr_{learning_rate:.5f}_"
                             f"dr_{dropout_rate:.5f}_nl_{num_layers}_"
                             f"hl_{hidden_layers}_auc_{epoch}.npy"),
                auc_metric.result().numpy())
        np.save(os.path.join(outdir,
                             f"val_cv_fold_{cv_fold}_lr_{learning_rate:.5f}_"
                             f"dr_{dropout_rate:.5f}_nl_{num_layers}_"
                             f"hl_{hidden_layers}_pr_auc_{epoch}.npy"),
                pr_auc_metric.result().numpy())
        np.save(os.path.join(outdir,
                             f"val_cv_fold_{cv_fold}_lr_{learning_rate:.5f}_"
                             f"dr_{dropout_rate:.5f}_nl_{num_layers}_"
                             f"hl_{hidden_layers}_y_pred_{epoch}.npy"),
                y_pred)
        np.save(os.path.join(outdir,
                             f"val_cv_fold_{cv_fold}_lr_{learning_rate:.5f}_"
                             f"dr_{dropout_rate:.5f}_nl_{num_layers}_"
                             f"hl_{hidden_layers}_y_true_{epoch}.npy"),
                y_true)
        np.save(os.path.join(outdir,
                             f"val_cv_fold_{cv_fold}_lr_{learning_rate:.5f}_"
                             f"dr_{dropout_rate:.5f}_nl_{num_layers}_"
                             f"hl_{hidden_layers}_meta_d_{epoch}.npy"),
                meta_d)

        auc_metric.reset_states()
        pr_auc_metric.reset_states()

    # Save out validation losses
    np.save(f"{outdir}/cv_fold{cv_fold}_val_lr_{learning_rate:.5f}_dr_{dropout_rate:.5f}_nl_{num_layers}_hl_{hidden_layers}_cv_loss.npy",
            val_losses)

    return best_val


def loop(dataset, model, train=False, optimizer=None, alpha=1, val=False):

    losses = []
    y_pred, y_true, meta_d, targets = [], [], [], []
    batch_num = 0
    # for batch in tqdm.tqdm(dataset):
    for batch in dataset:
        X, S, y, meta, M = batch
        if train:
            with tf.GradientTape() as tape:
                prediction = model(X, S, M, train=True, res_level=True)
                # Grab balanced set of residues
                iis = choose_balanced_inds(y)
                y = tf.gather_nd(y, indices=iis)
                y = y >= pos_thresh
                y = tf.cast(y, tf.float32)
                prediction = tf.gather_nd(prediction, indices=iis)
                loss_value = loss_fn(y, prediction)
        else:
            # we set train to False
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
            assert np.isfinite(float(loss_value))
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        losses.append(float(loss_value))
        y_pred.extend(prediction.numpy().tolist())
        y_true.extend(y.numpy().tolist())

        batch_num += 1

    return np.mean(losses), y_pred, y_true, meta_d


def convert_test_targs(y):
    # Need to convert targs (volumes) to 1s and 0s but also discard
    # intermediate values
    iis_pos = [np.where(np.array(i) >= pos_thresh)[0] for i in y]
    iis_neg = [np.where((np.array(i) < neg_thresh) & (np.array(i) >= 0))[0] for i in y]
    iis = []
    count = 0
    for i, j in zip(iis_pos, iis_neg):
        subset_iis = [[count, s] for s in j]
        for pair in subset_iis:
            iis.append(pair)
        subset_iis = [[count, s] for s in i]
        for pair in subset_iis:
            iis.append(pair)
        count += 1

    return iis


def choose_balanced_inds(y):
    iis_pos = [np.where(np.array(i) >= pos_thresh)[0] for i in y]
    iis_neg = [np.where((np.array(i) < neg_thresh) & (np.array(i) > -1))[0] for i in y]
    count = 0
    iis = []
    for i, j in zip(iis_pos, iis_neg):
        # print(len(i),len(j))
        if len(i) < len(j):
            subset = np.random.choice(j, len(i), replace=False)
            subset_iis = [[count, s] for s in subset]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count, s] for s in i]
            for pair in subset_iis:
                iis.append(pair)
        elif len(j) < len(i):
            subset = np.random.choice(i, len(j), replace=False)
            subset_iis = [[count, s] for s in subset]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count, s] for s in j]
            for pair in subset_iis:
                iis.append(pair)
        else:
            subset_iis = [[count, s] for s in j]
            for pair in subset_iis:
                iis.append(pair)
            subset_iis = [[count, s] for s in i]
            for pair in subset_iis:
                iis.append(pair)

        count += 1
    # select a random residue when there are no positive examples (or negative)
    # for a given structure
    if len(iis) == 0:
        # if there are multiple structures in batch simply use the first
        iis = [[0, random.choice(range(len(y[0])))]]
        print(f'selected random resid {iis[0][1]}')
        # iis = [[0, 0]]

    return iis


def mlflow_callback(study, trial):
    "Saves the parameters for each optuna run, along with the best loss for each"
    trial_value = trial.value if trial.value is not None else float("nan")
    with mlflow.start_run(run_name=f"optuna_CV"):
        mlflow.log_params(trial.params)
        mlflow.log_metrics({"loss": trial_value})


def train_function(learning_rate, dropout_rate, num_layers, hidden_layers, cv_fold, results):
    '''
        Callable training function that is run by a thread
        results : dictionary storing cv loss for multiple threads
    '''
    with tf.device("/gpu:%d" % cv_fold):
        # This print is likely printing gpu device 0 in all cases
        # gpu_device_name returns the name of a GPU device if available or the empty string
        print(f'running training for {cv_fold} using {tf.test.gpu_device_name()}')
        cv_loss = main_cv(
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            num_layers=num_layers,
            hidden_layers=hidden_layers,
            cv_fold=cv_fold,)
        results[cv_fold] = cv_loss


def objective(trial):
    "Defines objective function for optuna, uses mean best vallidation loss across folds"

    # similar to the original gvp paper
    # lowered learning rate min to reflect improved performance at lower learning rates
    # increased dropout rate max since the default is 0.1 in MQAModel
    # Also note that the hidden layers here refers to the number of hidden channels to use
    # for the scalar embedding. This does not apply to the hidden dimension
    # of the vector embedding which is currently set to a constant of 16.
    learning_rate = trial.suggest_loguniform("lr", 5e-5, 1e-2)
    dropout_rate = trial.suggest_loguniform("dropout", 1e-4, 2e-1)
    num_layers = trial.suggest_categorical("num_layers", [3, 4, 5, 6])
    hidden_layers = trial.suggest_categorical("hid_layers", [50, 100, 200])

    print(f'running trial with learning rate: {learning_rate: .4f}, dropout rate: {dropout_rate: .4f}'
          f', number of layers: {num_layers}, and hidden layers: {hidden_layers}')

    # loss = np.inf

    # sess = tf.compat.v1.Session()

    # "This loops computes loss from cross validation, alternatively you can use simply a single test splits, this is done in the GVP paper"
    # train_ops = []
    # for cv_fold in range(0, 5):
    #     "Here parallelize training by sending to different GPUs."
    #     with tf.device("/gpu:%d" % cv_fold):
    #         train_ops.append(
    #             main_cv(
    #                 learning_rate=learning_rate,
    #                 dropout_rate=dropout_rate,
    #                 num_layers=num_layers,
    #                 hidden_layers=hidden_layers,
    #                 cv_fold=cv_fold,)
    #         )
            # loss = main_cv(
            #     learning_rate=learning_rate,
            #     dropout_rate=dropout_rate,
            #     num_layers=num_layers,
            #     hidden_layers=hidden_layers,
            #     cv_fold=cv_fold,
            # )
            # cv_losses.append(loss)


    # Create multiple training threads

    # need a mutable object to pass to thread constructor
    # that can store results of training run
    cv_losses = {}

    train_threads = []
    for cv_fold in range(5):
        train_threads.append(threading.Thread(target=train_function,
                                              args=(learning_rate, dropout_rate,
                                                    num_layers, hidden_layers,
                                                    cv_fold, cv_losses)))

    # Start threads and block on their completion
    for t in train_threads:
        t.start()
    for t in train_threads:
        t.join()

    # Returns mean val loss across CV folds
    return np.mean([loss for loss in cv_losses.values()])


if __name__ == "__main__":
    #### GPU INFO ####
    #tf.debugging.enable_check_numerics()
    os.makedirs(outdir, exist_ok=True)
    model_path = outdir + '{}_{}'

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

    study = optuna.create_study()
    study.optimize(objective, n_trials=15, callbacks=[mlflow_callback])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))