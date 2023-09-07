import numpy as np
import random

def choose_balanced_inds(y):
    pos_mask = np.array(y) >= pos_thresh
    if train_on_intermediates:
        neg_mask = np.array(y) < pos_thresh
    else:
        neg_mask = np.array(y) < neg_thresh

    positive_example_count = pos_mask.sum()
    negative_example_count = neg_mask.sum()
    if (positive_example_count > 0 and negative_example_count > 0):
        struct_indices, residue_indices = np.where(pos_mask)
        # creates a N x 2 list where N is the number of examples
        pos_indices = np.array([[si, ri] for si, ri in
                                zip(struct_indices, residue_indices)])
        struct_indices, residue_indices = np.where(neg_mask)
        neg_indices = np.array([[si, ri] for si, ri in
                                zip(struct_indices, residue_indices)])
        # combine into single selection where the number of positive
        # examples matches the number of negative examples
        # if there are m and n examples from the two classes respectively and m < n,
        # we select a random set of m examples from the majority class
        pos_selection = (pos_indices if positive_example_count <= negative_example_count
                         else pos_indices[np.random.choice(range(positive_example_count),
                                                           negative_example_count, replace=False)])
        neg_selection = (neg_indices if positive_example_count >= negative_example_count
                         else neg_indices[np.random.choice(range(negative_example_count),
                                                           positive_example_count, replace=False)])
        selection = np.concatenate((pos_selection, neg_selection))

        # assert that number of selected residues is 2 x the number of examples
        # in the minority class
        assert selection.shape[0] == min(negative_example_count, positive_example_count) * 2
        return selection.tolist()
    # if there are no negative or positive examples in the batch return all indices
    else:
        return [[struct_index, res_index]
                for struct_index, y_vals in enumerate(y)
                for res_index in range(len(y_vals))]

def choose_balanced_inds_old(y):
    iis_pos = [np.where(np.array(i) >= pos_thresh)[0] for i in y]
    if train_on_intermediates:
        iis_neg = [np.where(np.array(i) < pos_thresh)[0] for i in y]
    else:
        iis_neg = [np.where(np.array(i) < neg_thresh)[0] for i in y]
    count = 0
    iis = []
    for i, j in zip(iis_pos, iis_neg):
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
        iis = [[0, random.choice(range(len(y[0])))]]
        # print(f'selected random resid {iis[0][1]}')

    return iis

def choose_balanced_inds_oversampling(y):
    pos_mask = np.array(y) >= pos_thresh
    if train_on_intermediates:
        neg_mask = np.array(y) < pos_thresh
    else:
        neg_mask = np.array(y) < neg_thresh

    positive_example_count = pos_mask.sum()
    negative_example_count = neg_mask.sum()
    if (positive_example_count > 0 and negative_example_count > 0):
        struct_indices, residue_indices = np.where(pos_mask)
        # creates a N x 2 list where N is the number of examples
        pos_indices = np.array([[si, ri] for si, ri in
                                zip(struct_indices, residue_indices)])
        struct_indices, residue_indices = np.where(neg_mask)
        neg_indices = np.array([[si, ri] for si, ri in
                                zip(struct_indices, residue_indices)])
        # combine into single selection where the number of positive
        # examples matches the number of negative examples
        # if there are m and n examples from the two classes respectively and m < n,
        # we select a random set of n examples from the minority class
        pos_selection = (pos_indices if positive_example_count >= negative_example_count
                         else pos_indices[np.random.choice(range(positive_example_count),
                                                           negative_example_count)])
        neg_selection = (neg_indices if positive_example_count <= negative_example_count
                         else neg_indices[np.random.choice(range(negative_example_count),
                                                           positive_example_count)])
        selection = np.concatenate((pos_selection, neg_selection))

        # assert that number of selected residues is 2 x the number of examples
        # in the majority class
        assert selection.shape[0] == max(negative_example_count, positive_example_count) * 2
        return selection.tolist()
    # if there are no negative or positive examples in the batch return all indices
    else:
        return [[struct_index, res_index]
                for struct_index, y_vals in enumerate(y)
                for res_index in range(len(y_vals))]

pos_thresh = 70
neg_thresh = 60
train_on_intermediates = True
# test = [[100, 50, 30], [120, 10, 20]]
test = [[110, 50, 130], [10, 120, 120]]
print(choose_balanced_inds_oversampling(test))
print(choose_balanced_inds_old(test))