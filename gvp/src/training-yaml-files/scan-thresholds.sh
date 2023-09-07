for tf in `ls task2-schemes`; do
	for threshold in 20 30 40; do
		sed -i "s/pos_thresh.*/pos_thresh: $threshold/g" task2-schemes/$tf;
		cp task2-schemes/$tf to-run-task2-gp-to-nearest-resi/$(basename "$tf" .yaml)_pos_thresh_$threshold.yaml;
	done;
done;
		
