import numpy as np

from configs.config_utils import parNN_list


def build_feature_for_model_train(exp_dataset, aux_dataset):
    return exp_dataset + aux_dataset


def build_target_for_model_loss(exp_dataset, aux_dataset):
    ## target structure
    is_exp_mask_1          = np.ones_like(exp_dataset, shape=(exp_dataset.n_samples, 1))    # 1 for dim 1 because the NN's output is 1D.
    not_is_aux             = np.zeros_like(aux_dataset, shape=(aux_dataset.n_samples, 1))
    is_exp_mask            = np.concatenate((is_exp_mask_1, not_is_aux), axis=0)
    
    is_exp_mask_2          = np.ones_like(exp_dataset, shape=(exp_dataset.n_samples, 1))
    weights_mask           = np.ones_like(aux_dataset, shape=(aux_dataset.n_samples, 1)) \
        * exp_dataset.n_samples * 1. / aux_dataset.n_samples
    weights                = np.concatenate((is_exp_mask_2, weights_mask), axis=0)
    
    loss_mask              = np.concatenate((is_exp_mask, weights), axis=1)
    
    return loss_mask


def build_shape_dictionary_list():
    # todo: this should be drawn from the config
    return [parNN_list['scale']]  # todo: this should be of the length of deltas? Look @ imperfect_model impolementation

