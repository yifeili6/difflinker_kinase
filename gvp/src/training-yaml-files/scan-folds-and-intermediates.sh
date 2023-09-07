for tf in `ls schemes`; do
	for intermediates in False True; do
		for fold in {0..4}; do
			sed -i "s/train_on_intermediates.*/train_on_intermediates: $intermediates/g" schemes/$tf;
			sed -i "s/fold:.*/fold: $fold/g" schemes/$tf;
			cp schemes/$tf to-run/$(basename "$tf" .yaml)_intermediates_${intermediates}_fold_${fold}.yaml;
		done;
	done;
done;
		
