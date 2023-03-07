from OrdinalClassifPipeline import PIPELINE_STAGES as PIPELINE_STAGES_CLASSIF

from sklearn.model_selection import ParameterGrid, ParameterSampler

HYPERPARAMS_SEARCH = ['grid', 'random']

KEY_SEP = ':'


# returns a flattened version of the hyperparameters set
def hyperparams_flatten(hyperparams_unflat, is_classif=True):
    if not isinstance(hyperparams_unflat, dict):
        raise TypeError

    pipeline_valid_stages = PIPELINE_STAGES_CLASSIF
    if not is_classif:
        raise RuntimeError

    hyperparams_flat = {}
    for key, val_dict in hyperparams_unflat.items():
        if key not in pipeline_valid_stages:
            raise ValueError
        if not isinstance(val_dict, dict):
            raise TypeError

        if not val_dict:  # empty dict
            hyperparams_flat[key] = [{}]
        else:
            for tag, val in val_dict.items():
                if not isinstance(val, list):
                    val = [val]
                hyperparams_flat[key + KEY_SEP + tag] = val

    return hyperparams_flat


# returns an unflattened version of the hyperparameters set
def hyperparams_unflatten(hyperparams_flat, is_classif=True):
    if not isinstance(hyperparams_flat, dict):
        raise TypeError

    pipeline_valid_stages = PIPELINE_STAGES_CLASSIF
    if not is_classif:
        raise RuntimeError

    hyperparams_unflat = {st: {} for st in pipeline_valid_stages}
    for key_tag, val in hyperparams_flat.items():
        if key_tag in pipeline_valid_stages:  # empty hyperparams
            continue

        key, tag = key_tag.split(KEY_SEP, 1)
        if key not in pipeline_valid_stages:
            raise ValueError

        val_dict = hyperparams_unflat[key]
        val_dict.update({tag: val})
        hyperparams_unflat[key] = val_dict

    return hyperparams_unflat


# generate a sample collection of hyperparameters by either grid or random distribution
def hyperparams_sample(hyperparam_set, search, iters, random_seq):
    l_hyperparams = []
    if search not in HYPERPARAMS_SEARCH:
        raise ValueError
    elif search == 'grid':
        l_hyperparams = ParameterGrid(hyperparam_set)
    elif search == 'random':
        # generate random state: first create child sequence by spawning, then draw a sample
        random_state = random_seq.spawn(1)[0].generate_state(1)[0]
        l_hyperparams = ParameterSampler(hyperparam_set, n_iter=iters, random_state=random_state)

    l_hyperparams = list(l_hyperparams)
    return l_hyperparams
