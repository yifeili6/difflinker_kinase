#!/bin/bash
#BSUB -L /bin/bash
#BSUB -q bowman
#BSUB -gpu "num=1"
#BSUB -R "select[defined(rocm)] rusage[rocm=1]"
#BSUB -J training
#BSUB -o logs/training-%J.log
#BSUB -e logs/training-%J.log

# things to do before Singularity
# sleep 7m
export HIP_VISIBLE_DEVICES=$(/opt/rocm-common/bin/find_rocm.sh -n 1)
date
hostname
echo "HIP ROCM REQUESTS $HIP_VISIBLE_DEVICES"

#The Singularity run
#singularity exec --no-home -H /project/bowmanlab/ameller/tf-singularity-home/ -B /project:/project /project/bowmanlab/rocm/tensorflow-rocm.sif python train_fpocket_drug_score_labels.py training-yaml-files/to-run-task2-fpocket/train_1prot_fpocket_diff_refine.yaml 


singularity exec --no-home -H /project/bowmanlab/ameller/tf-singularity-home/ -B /project:/project /project/bowmanlab/rocm/tensorflow-rocm.sif python train_xtal_predictor.py training-yaml-files/to-run-task2-gp-to-nearest-resi/train_1prot_4resis_gp_to_nearest_resi_no_balance_include_intermediates_pos_thresh_30_lm_squeeze.yaml

#singularity exec --no-home -H /project/bowmanlab/ameller/tf-singularity-home/ -B /project:/project /project/bowmanlab/rocm/tensorflow-rocm.sif python train_xtal_predictor.py training-yaml-files/pretrained-models/train_4prot_4resis_constant_draw_640_resis_include_intermediates_pos_thresh_87.yaml

#singularity exec --no-home -H /project/bowmanlab/ameller/tf-singularity-home/ -B /project:/project /project/bowmanlab/rocm/tensorflow-rocm.sif python train_cryptic_labels.py training-yaml-files/to-run-task2-gp-to-nearest-resi/train_1prot_4resis_gp_to_nearest_resi_no_balance_include_intermediates_pos_thresh_30.yaml

#Things to do after singularity
echo Done
