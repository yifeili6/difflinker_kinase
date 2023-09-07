from glob import glob
import os
import numpy as np

training_directories = glob('/project/bowmanlab/ameller/gvp/task1-final-folds-window-40-nearby-pv-procedure/*')

complete_subdirs = []
incomplete_subdirs = []
missing_fold_subdirs = []
for td in training_directories:
    subdirs = glob(f'{td}/*30epoch*')
    if len(subdirs) == 5:
        test_files = glob(f'{td}/*30epoch*/test_pr_auc.npy')
        if len(test_files) == 5:
            complete_subdirs.append(os.path.basename(td))
        else:
            incomplete_subdirs.append(os.path.basename(td))

    else:
        missing_fold_subdirs.append(os.path.basename(td))


print(f'Complete subdirs ({len(complete_subdirs)}): {complete_subdirs}')
print(f'Incomplete subdirs: {incomplete_subdirs}')
print(f'Subdirs with missing folds: {missing_fold_subdirs}')

for td in training_directories:
    subdirs = glob(f'{td}/*30epoch*')
    if len(subdirs) < 5:
        folds = [int(os.path.basename(folder).split('_')[-1]) for folder in subdirs]
        missing_folds = np.arange(5)[~np.in1d(range(5), folds)]
        # print(folds, missing_folds)
        print(f'{os.path.basename(td)} is missing folds {missing_folds}')
