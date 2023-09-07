# IMPORTS
import numpy as np

# filepaths
upperpath = "/home/jonathanb/mount" #"X:/project"
structurepath = f"{upperpath}/bowmanlab/borowsky.jonathan/FAST-cs/pocket-tracking/all-structures"
valapopath = f"{upperpath}/bowmanlab/borowsky.jonathan/FAST-cs/protein-sets/new_pockets/labels/validation_apo_ids_all.npy"
truelabelpath = f"{upperpath}/bowmanlab/borowsky.jonathan/FAST-cs/protein-sets"
predlabelpath = f"{upperpath}/bowmanlab/ameller/gvp/task2"

# parameters
binarize = False
binthresh = 0.5

# list of training schemes/network versions
label_trainschemes = [\
"train-with-4-residue-batches-constant-size-balanced-640-resi-draws-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b4resis_b4proteins_30epoch_feat_method_nearby-pv-procedure_rank_7_stride_1_window_40_pos_87",\
f"train-with-1-protein-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b1proteins_30epoch_feat_method_nearby-pv-procedure_rank_7_stride_1_window_40_pos_87",\
f"train-with-1-protein-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b1proteins_30epoch_feat_method_nearby-pv-procedure_rank_7_stride_1_window_40_pos_116",\
f"train-with-1-protein-batches-no-balancing-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b1proteins_30epoch_feat_method_nearby-pv-procedure_rank_7_stride_1_window_40_pos_145",\
f"train-with-1-protein-batches-undersample-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b1proteins_30epoch_feat_method_nearby-pv-procedure_rank_7_stride_1_window_40_pos_87",\
f"train-with-1-protein-batches-undersample-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b1proteins_30epoch_feat_method_nearby-pv-procedure_rank_7_stride_1_window_40_pos_116",\
f"train-with-1-protein-batches-undersample-intermediates-in-training/net_8-50_1-32_16-100_dr_0.1_nl_4_hd_100_lr_2e-05_b1proteins_30epoch_feat_method_nearby-pv-procedure_rank_7_stride_1_window_40_pos_145"]

# global load commands

  # proteins in the validation set
val_apo_ids = np.load(val_apo_path)
val_apo_ids_nochain = [i[0:4] for i in val_apo_ids]

  # true labels
truelabels_all = np.load(f"{upperpath}/bowmanlab/borowsky.jonathan/FAST-cs/protein-sets/all-GVP-project-ligand-resis.npy", allow_pickle=True)
truelabels_all_apo = [i[0][0].lower() for i in truelabels_all[0]]

  # predicted labels
#identify the best epoch for each training parameter set; deprecated as Artur's code now takes care of this
#best_epochs = []

#for ts in label_trainschemes:

#    best_epoch = -1
#    best_pr_auc = 0

#    for epoch in range(0,30):
#        predprauc_mean = np.mean(np.load(f"{predlabelpath}/{ts}/val_protein_pr_aucs_{epoch}.npy"))
#        if predprauc_mean > best_pr_auc:
#            best_pr_auc = predprauc_mean
#            best_epoch = epoch

#    best_epochs.append(best_epoch)

#load the predicted labels and pr aucs for the epoch with the best average pr auc
#predlabels_all = [np.load(f"{predlabelpath}/{i}/val_y_pred_{best_epochs[x]}.npy") for x, i in enumerate(label_trainschemes)]
#predpraucs_all = [np.load(f"{predlabelpath}/{i}/val_protein_pr_aucs_{best_epochs[x]}.npy") for x, i in enumerate(label_trainschemes)]

predlabels_all = [np.load(f"{predlabelpath}/{i}/full_val_set_y_pred.npy") for x, i in enumerate(label_trainschemes)]
predpraucs_all = [np.load(f"{predlabelpath}/{i}/full_val_set_protein_pr_aucs.npy") for x, i in enumerate(label_trainschemes)]

# graphics settings; presently unused

def set_graphics():
    cmd.do("bg grey50")
    cmd.do("set orthoscopic,1")
    cmd.do("set depth_cue,0")
    cmd.do("set auto_zoom, off")
    cmd.do("set sphere_quality, 5")
    cmd.do("set opaque_background, off")
    cmd.do("set ray_opaque_background, off")
    cmd.do("set antialias, 2")
    cmd.do("set ray_trace_mode, 1")
    cmd.do("set ray_trace_color, black")

#summary information to print when creating the macro

print(val_apo_ids)
print("Input:\nt2mapper ['t' to color labels on a [0, 1] scale, anything else to color them on a [0, max_label] scale], [index of the training scheme of interest; see above], [protein apo pdb id, or index in validation set protein list]")

#define method
@cmd.extend
def t2mapper(abscale, label_ts_ind, protin):

    #identify and process inputs
    if abscale == "t":
        absolute_scale = True
    elif abscale == "f":
        absolute_scale = False
    else:
        absolute_scale = True
        print("invalid scale argument, enter 't' or 'f', defaulting to True")

    try:
        protx = int(protin)
        prot = val_apo_ids_nochain[protx]
    except ValueError:
        prot = protin
        protx = val_apo_ids_nochain.index(prot.lower())

    #get protein information
    prot_index = truelabels_all_apo.index(prot)
    prot_info = truelabels_all[0][prot_index][0]
    truelabels = truelabels_all[0][prot_index][4]

    #print summary information
    #print(f"epoch = {best_epochs[int(label_ts_ind)]}")
    print(f"PR-AUC = {predpraucs_all[int(label_ts_ind)][protx]}")
    print(prot_info)

    #delete all and load apo and holo structures
    cmd.delete("all")

    aponame = prot_info[0]+prot_info[1]+"_clean_h"
    holoname = prot_info[2]+prot_info[3]+"_clean_ligand_h"

    cmd.load(f"{structurepath}/{aponame}.pdb")
    cmd.load(f"{structurepath}/{holoname}.pdb")
    cmd.align(aponame, holoname, cycles = 0)

    #misc graphics settings
    util.cbac(holoname)
    cmd.hide("cartoon", holoname)

    #show true labels as lines
    for i in truelabels:
        cmd.show("lines", f"{aponame} and resi {i}")

    #show predicted labels by color

        #get initial pdb residue number
    init_resi = -999
    for line in open(f"{structurepath}/{aponame}.pdb"):
        if init_resi == -999 and line[0:4] == "ATOM":
            init_resi = int(line[22:26])
            break

    predlabels = predlabels_all[int(label_ts_ind)][protx]

        #set label mapping to maximum color value
    if absolute_scale:
        scalemax = 1
    else:
        scalemax = max(predlabels)

        #color residues
    for resi, label in enumerate(predlabels):

        if binarize:
            if label > binthresh:
                label = 1
            else:
                label = 0

        grp = f"{aponame} and resi {int(resi)+init_resi}"
        s = f"b={label}"
        cmd.alter(grp, s)

    cmd.spectrum("b", "red_white_blue", aponame, 0, scalemax)
