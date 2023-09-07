import os

incomplete = []
for batch in [1, 2, 4]:
	for window in [10, 20, 40]:
		for cv in range(5):
			fname = (f'/project/bowmanlab/ameller/gvp/net_8-50_1-32_16-100_feat_method_gp-to-nearest-resi_rank_6_pos20_'
					 f'window_{window}ns_50epoch_b{batch}_cv_fold{cv}_test_TEM-1MY0-1BSQ-nsp5-il6/auc.npy')
			if not os.path.isfile(fname):
				incomplete.append((batch, cv, window))

print(incomplete)
print(len(incomplete))

unstarted = []
for batch, cv, window in incomplete:
	dir_name = (f'/project/bowmanlab/ameller/gvp/net_8-50_1-32_16-100_feat_method_gp-to-nearest-resi_rank_6_pos20_'
		   		f'window_{window}ns_50epoch_b{batch}_cv_fold{cv}_test_TEM-1MY0-1BSQ-nsp5-il6')
	if not os.path.exists(dir_name):
		unstarted.append((batch, cv, window))

print(unstarted)
print(len(unstarted))


'''
Ran the following to figure out which jobs were still actively running:
for i in `bjobs | grep training | grep 'Aug 13' | cut -d' ' -f1`; do grep -B 1 $i job_tracking.log; done

Output (Format is $batch $fold $window):
1 0 10
Job <1346098> is submitted to queue <bowman>.
1 1 10
Job <1346101> is submitted to queue <bowman>.
1 4 10
Job <1346111> is submitted to queue <bowman>.
1 3 10
Job <1346107> is submitted to queue <bowman>.
'''

'''
Currently running also includes unstarted jobs that are already running:
2 10 2
Job <1346195> is submitted to queue <bowman>.
2 20 2
Job <1346196> is submitted to queue <bowman>.
2 20 3
Job <1346197> is submitted to queue <bowman>.
2 40 4
Job <1346200> is submitted to queue <bowman>.
'''

currently_running = [
	(1, 0, 10),
	(1, 1, 10),
	(1, 4, 10),
	(1, 3, 10),
	(2, 2, 10),
	(2, 2, 20),
	(2, 3, 20),
	(2, 4, 40),
]

for batch, cv, window in incomplete:
	if (batch, cv, window) not in currently_running:
		print(f'To run {batch} {cv} {window}')


# for batch, cv, window in incomplete:
# 	if (batch, cv, window) not in currently_running:
# 		print(batch, cv, window)
# 		os.system(f'sed -i "s/python train_pockets.py.*/python train_pockets.py {batch} {cv} {window}/g" train-submit-amd.sh')
# 		os.system('bsub < submit-gvp-amd.sh')
# 		os.system('sleep 7m')

