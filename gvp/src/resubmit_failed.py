import os

NUM_EPOCHS = 50
pos_thresh = 20
min_rank = 6
test_string = 'TEM-1MY0-1BSQ-nsp5-il6'
featurization_method = 'gp-to-nearest-resi'
LEARNING_RATE = 0.0002

for BATCH_SIZE in [1, 2, 4]:
	for cv_fold in range(5):
		for window in [10, 20, 40]:
			nn_dir = (f"/project/bowmanlab/ameller/gvp/net_8-50_1-32_16-100_"
			          f"feat_method_{featurization_method}_rank_{min_rank}_"
			          f"pos{pos_thresh}_window_{window}ns_"
			          f"{NUM_EPOCHS}epoch_b{BATCH_SIZE}_lr{LEARNING_RATE}_"
			          f"cv_fold{cv_fold}_test_{test_string}/")
			if not os.path.exists(os.path.join(nn_dir, 'test_auc.npy')):
				print(BATCH_SIZE, cv_fold, window)
				os.system(f'sed -i "s/python train_pockets.py.*/python train_pockets.py {BATCH_SIZE} {cv_fold} {window}/g" train-submit-amd.sh;')
				os.system('bsub < submit-gvp-amd.sh')
				os.system('sleep 7m')

			# echo $batch $fold $window;
			# sed -i "s/python train_pockets.py.*/python train_pockets.py $batch $fold $window/g" train-submit-amd.sh;
			# bsub < submit-gvp-amd.sh;
			# sleep 7m;
		