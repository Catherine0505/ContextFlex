import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import *
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib
import skimage
import os
import re
import pathlib
import pickle 
data_gen = torch.Generator().manual_seed(4)
np.random.seed(4)

from . import losses
            

loss_fn_dict = {
    "mean_softdice_loss": losses.MeanDice,  
    "mean_ce_loss": losses.MeanCE 
}

def get_model_root_folder(args): 
    def get_dataset_folder(): 
        dataset_name = args.dataset_name 
        slice_num = args.slice_num if args.slice_num is not None else 0
        num_classes = args.num_classes
        dataset_folder_params = [dataset_name, slice_num, num_classes]
        assert_all_exist(dataset_folder_params, "dataset")
        dataset_folder_params = [dataset_name, f"slice{slice_num}", f"{num_classes}classes"]
        return "_".join(dataset_folder_params)
    
    def get_modality_folder(): 
        modalities = args.modalities or ["default"]
        modality_folder_params = sorted(modalities)
        assert_all_exist(modality_folder_params, "modalities")
        return "+".join(modality_folder_params)
    
    def get_corruption_folder(): 
        corruption_folder = f"corrupt_{args.num_corruptions}" if args.corrupt else "nocorrupt"
        corruption_types = []
        if args.corrupt: 
            corruption_types += get_randomnoise_folder()
            corruption_types += get_gaussianblur_folder()
            corruption_types += get_motion_corruption_folder()
        corruption_type_folder = "_".join(corruption_types)
        return corruption_folder + "/" + corruption_type_folder
    
    def get_grouping_folder(): 
        group_size = args.group_size
        grouping_criterion = args.grouping_criterion
        average_group = args.average_group
        assert_all_exist([group_size, grouping_criterion, average_group], "grouping")
        n_gs = f"gs{group_size[0]}to{group_size[-1]}"
        n_averagegroup = ["averagegroup"] if average_group else []
        grouping_params = [grouping_criterion, n_gs] + n_averagegroup
        return "_".join(grouping_params)
        

    def get_randomnoise_folder(): 
        return [f"noiseratio{args.noise_ratio}"] if args.do_random_noise else []
    def get_gaussianblur_folder(): 
        return [f"blursigma{args.sigma}truncate{args.truncate}"] if args.do_gaussian_blur else [] 
    def get_motion_corruption_folder(): 
        return [f"motiontranslation{args.max_translation}" \
                f"rotation{args.max_rotation}" \
                f"freq{args.freq_cutoff_idx_range[0]}to{args.freq_cutoff_idx_range[-1]}"] \
                if args.do_motion_corruption else [] 
    
    model_save_folder = pathlib.Path(args.save_root_folder) 
    return model_save_folder / get_dataset_folder() / get_modality_folder() / get_corruption_folder() / \
        get_grouping_folder()


def get_model_folder(args, network_size): 
    return pathlib.Path("_".join([args.aggregate_type, args.fusion_type, str(network_size), \
                                  f"lr{args.lr}", f"seed{args.seed}"]))
    

def get_wandb_projname(args): 
    dataset_name = args.dataset_name 
    slice_num = args.__dict__.get("slice_num") or str(0)
    num_classes = str(args.__dict__.get("num_classes"))
    wandb_projname_params = [dataset_name, f"slice{slice_num}", f"{num_classes}classes"]
    assert_all_exist(wandb_projname_params, "dataset")
    return "_".join(wandb_projname_params)

def get_wandb_config(args): 
    return {k:v if v else "None" for (k, v) in args.__dict__.items()}


def get_loss_fn_dict(args): 
    return_dict = {}
    for loss_fn_name in args.loss_function: 
        return_dict[loss_fn_name] = loss_fn_dict[loss_fn_name]
    return return_dict


def assert_in_range(tensor, range, name='tensor'):
    assert len(range) == 2, 'range should be in form [min, max]'
    assert tensor.min() >= range[0], f'{name} should be in {range}, found: {tensor.min()}'
    assert tensor.max() <= range[1], f'{name} should be in {range}, found: {tensor.max()}'

def assert_all_exist(param_list, name): 
    assert all([p is not None for p in param_list]), \
            f"Expect arguments to construct {name} folder to exist but got: {param_list}."


def get_corruption_setting_lst(args): 
    corruption_setting_lst = []
    if not args.corrupt: 
        corruption_setting_lst.append("nocorrupt")
    else: 
        corruption_setting_lst.append(f"numcorrupt{args.num_corruptions}")
        if not args.do_motion_corruption:
            corruption_setting_lst.append(f"motion-None")
        else: 
            corruption_setting_lst.append(f"motion-{args.max_translation}" \
                                f"-{args.max_roration}" \
                                f"-{args.freq_cutoff_idx_range[0]}" \
                                    f"-{args.freq_cutoff_idx_range[1]}")
        if not args.do_gaussian_blur is None: 
            corruption_setting_lst.append(f"blur-None")
        else: 
            corruption_setting_lst.append(f"blur-{args.sigma}" \
                                f"-{args.truncate}")
        if not args.do_random_noise: 
            corruption_setting_lst.append(f"noise-None")
        else: 
            corruption_setting_lst.append(f"noise-{args.noise_ratio}")
    return corruption_setting_lst


def get_dataloading_setting_lst(args): 
    dataloading_setting_lst = []
    modalities = sorted(args.modalities)
    dataloading_setting_lst.append("+".join(modalities))
    return dataloading_setting_lst


def get_grouping_setting_lst(args): 
    grouping_setting_lst = []
    grouping_setting_lst.append(f"groupby{args.grouping_criterion}")

    return grouping_setting_lst 


def get_modelinit_training_setting_lst(args): 
    modelinit_training_setting_lst = []
    if args.seed_init > 0: 
        modelinit_training_setting_lst.append("seedinit")
    modelinit_training_setting_lst.append(args.loss_function.replace("_", ""))   # Loss function
    modelinit_training_setting_lst.append(f"lambdace{args.lambda_ce if args.loss_function == 'softdice_ce' else 0}")  # lambda for regularization
    modelinit_training_setting_lst.append(f"valby{args.val_by}")
    modelinit_training_setting_lst.append(f"valprecision{args.validation_precision}")
    modelinit_training_setting_lst.append(f"valthreshold{args.nondecreasing_threshold}")

    return modelinit_training_setting_lst


def load_label_colormap(dataset_name, num_classes): 
    """
    Returns a list of (R, G, B) values representing label-color correspondence. 
        (R, G, B) values are in floating point. 
        The i-th element is the color of Class i. 
    Arguments: 
        dataset_name: String. Indicates dataset name for finding its corresponding colormap. 
    """
    current_path = os.path.realpath(__file__) # Real path (not sym-link!) of current file 
    current_path = pathlib.Path(current_path)
    colormaps_path = current_path.parent / "data" / "configs" / "label_colormap.pickle"

    with open(colormaps_path, "rb") as handle: 
        dataset_colormap_dict = pickle.load(handle)

    k = f"{dataset_name}_{num_classes}classes"
    if k not in dataset_colormap_dict.keys(): 
        raise ValueError(f"{k} not found. " \
                         f"Available datasets are: {dataset_colormap_dict.keys()}.")
    
    return dataset_colormap_dict[k]


def get_loss_list(predy, y, return_by_sample = False, checkpoint_folder = None, 
    num_classes = 1, class_weights = None, lambda_ce = 1., include_background = True, background_class = 0):
    """
    Return a list of losses. Used for creating model comparison CSV file. 

    Arguments: 
        predy: raw network output. torch.Tensor. shape: (b n c h w) for GroupNets. (b c h w) for SimpleUNets. 
        y: ground truth labels. torch.Tensor. shape: (b n c h w) for GroupNets. (b c h w) for SimpleUNets. 
        return_by_sample: boolean. If true, all losses returned are recorded for each sample. Otherwise, return averaged loss over 
            all samples. 
        checkpoint_folder: Python string. Checkpoint folder name. Used to determine converting method from raw logits to one-hot predictions.
            If "groupconv" in checkpoint_folder, max over dimension 2. Otherwise max over dimension 1. 

    Returns: 
        loss_lst: Python list of numpy.array. Losses are in the order of: 
            [loss_softdice, loss_ce, loss_lossfunc, loss_harddice, loss_softdice_class, loss_harddice_class]
            The first four are averaged loss across all classes. 
            The last two are SoftDice Loss and HardDice Loss for each class. 
            Shapes: 
                First four (averaged across class): (b*n 1) if return by sample. (1, 1) number if not return by sample. 
                Last two (for each class): (b*n c) if return by sample. (1, c) if not return by sample. 
    """
    
    # Load criterions. 
    if "groupconv" in checkpoint_folder: 
        criterion_softdice = losses.MeanDice(dim = 2, from_logits = True)
        criterion_ce = losses.MeanCE(dim = 2, weight = class_weights)
        criterion_harddice = losses.MeanDice(dim = 2, from_logits = False)
    elif "simpleunet" in checkpoint_folder:
        criterion_softdice = losses.MeanDice(dim = 1, from_logits = True)
        criterion_ce = losses.MeanCE(dim = 1, weight = class_weights)
        criterion_harddice = losses.MeanDice(dim = 1, from_logits = False)

    loss_softdice = criterion_softdice(predy, y, return_by_sample = return_by_sample, include_background = include_background, background_class = background_class).detach().cpu().numpy()
    loss_ce = np.expand_dims(criterion_ce(predy, y, return_by_sample = True).detach().cpu().numpy(), -1)
    loss_softdice = loss_softdice.reshape((-1, loss_softdice.shape[-1]))
    loss_lossfunc = loss_softdice + lambda_ce * loss_ce
    # SoftDice Loss for each class
    loss_softdice_class = criterion_softdice(predy, y, return_by_sample = return_by_sample, return_by_class = True).detach().cpu().numpy()
    loss_softdice_class = loss_softdice_class.reshape((-1, loss_softdice_class.shape[-1]))

    if "groupconv" in checkpoint_folder: 
        predy_hard = F.one_hot(predy.argmax(dim = 2), num_classes + 1).permute((0, 1, 4, 2, 3))
    elif "simpleunet" in checkpoint_folder:
        predy_hard = F.one_hot(predy.argmax(dim = 1), num_classes + 1).permute((0, 3, 1, 2))
    loss_harddice = criterion_harddice(predy_hard, y, return_by_sample = return_by_sample, include_background = include_background, background_class = background_class).detach().cpu().numpy()
    loss_harddice = loss_harddice.reshape((-1, loss_harddice.shape[-1]))
    # HardDice Loss for each class
    loss_harddice_class = criterion_harddice(predy_hard, y, return_by_sample = return_by_sample, return_by_class = True).detach().cpu().numpy()
    loss_harddice_class = loss_harddice_class.reshape((-1, loss_harddice_class.shape[-1]))

    return [loss_softdice, loss_ce, loss_lossfunc, loss_harddice, loss_softdice_class, loss_harddice_class]


def plt_performance_vs_networksize_groupdfs(grouped_dfs_dict, constraints, summary_type,
                                            performance_metric_column, precision, 
                                            fig, title, ylim_lower, ylim_upper): 
    # `grouped_dfs`: a dictionary with group size as key and grouped model performance 
        # dataframe as value. Model performance dataframe grouped by "model_type", "aggregate_type", 
        # "fusion_type". 
    # `constraints`: list of constraint column names in each grouped_df. 
        # Named with "constrain_[i]" with "i" 
        # being the constraint ID. Current constraints are `bs`, `lr`
    # `summary_type`: One group may contain multiple constraint values. `summary_type` represents how
        # these constraint values should be combined. Current choices are `best`, `average`, `separate`. 
    # `performance_metric_column`: column name of the performance metric. (y-axis of the plot) 

    assert summary_type in ["separate", "average", "best"] 

    metric_stats = {}
    for dataset_key in sorted(grouped_dfs_dict.keys()):

        grouped_dfs = grouped_dfs_dict[dataset_key]

        for (group_size, grouped_df) in grouped_dfs.items(): 
            grouped_df = sorted(grouped_df, key=lambda x: x[0])
            metric_stats[group_size] = {}

            for (k, v) in grouped_df: 
                grouped_df_key = [dataset_key]
                grouped_df_key = grouped_df_key + list(k)
                # print(k)

                grouped_df_constraints = v.groupby(constraints)
                grouped_df_constraints = sorted(grouped_df_constraints, key=lambda x: x[0])

                metric_mean_constraints = {}
                metric_std_constraints = {}
                network_sizes_start = []
                for (constraint, grouped_df_constraint) in grouped_df_constraints: 
                    network_sizes = sorted(list(grouped_df_constraint["network_size"].unique()))
                    if summary_type == "separate": 
                        grouped_df_key = tuple(k + list(constraint))
                        if len(network_sizes_start) != 0: 
                            assert network_sizes_start == network_sizes, f"{network_sizes_start}, {network_sizes}"

                    # Sanity check that for a given model type, network sizes are the same for all constraints.
                    
                    network_sizes_start = network_sizes 

                    metric_mean = []
                    metric_std = []
                    for network_size in network_sizes: 
                    # print(len(v[v["network_size"] == network_size][performance_metric_column]))
                        metric_mean.append(grouped_df_constraint[grouped_df_constraint["network_size"] == network_size][performance_metric_column].mean())
                        metric_std.append(grouped_df_constraint[grouped_df_constraint["network_size"] == network_size][performance_metric_column].std())
                    metric_mean = np.array(metric_mean)
                    metric_std = np.array(metric_std) 
                    for i in range(len(network_sizes)): 
                        network_size = network_sizes[i]
                        if network_size in metric_mean_constraints.keys(): 
                            metric_mean_constraints[network_size].append(metric_mean[i])
                        else: 
                            metric_mean_constraints[network_size] = [metric_mean[i]]
                        if network_size in metric_std_constraints.keys(): 
                            metric_std_constraints[network_size].append(metric_std[i])
                        else: 
                            metric_std_constraints[network_size] = [metric_std[i]]
                    if summary_type == "separate": 
                        metric_stats[group_size][tuple(grouped_df_key)] = \
                            [network_sizes, metric_mean, metric_std]
                network_sizes_mean = sorted(list(metric_mean_constraints.keys())) 
                network_sizes_std = sorted(list(metric_std_constraints.keys()))
                assert network_sizes_mean == network_sizes_std
                mean_arr, std_arr = [], []
                for network_size in network_sizes_mean: 
                    
                    if summary_type == "best" and "dice" in performance_metric_column: 
                        mean_arr.append(np.max(metric_mean_constraints[network_size]))
                        std_arr.append(metric_std_constraints[network_size][np.argmax(metric_mean_constraints[network_size])])
                    elif summary_type == "best" and "loss" in performance_metric_column: 
                        mean_arr.append(np.min(metric_mean_constraints[network_size]))
                        std_arr.append(metric_std_constraints[network_size][np.argmin(metric_mean_constraints[network_size])])
                    elif summary_type == "average": 
                        mean_arr, std_arr = np.mean(metric_mean_constraints[network_size]), \
                            np.mean(metric_std_constraints[network_size])
                if summary_type in ["average", "best"]:
                    metric_stats[group_size][tuple(grouped_df_key)] = [network_sizes_mean, mean_arr, std_arr]

        # print(metric_stats[1])
        start = [] 
        for (_, grouped_metric) in metric_stats.items(): 
            keys = sorted([i[0] for i  in grouped_metric]) 
            if len(start) != 0: 
                assert start == keys 
            start = keys 

        for label in list(metric_stats.items())[0][1].keys(): 
            mean_arr = []
            std_arr = []
            network_sizes_start = []
            for group_size in sorted(list(metric_stats.keys())): 
                mean_arr.append(metric_stats[group_size][label][1])
                std_arr.append(metric_stats[group_size][label][2])
                network_sizes = metric_stats[group_size][label][0]
                if len(network_sizes_start) != 0: 
                    assert network_sizes_start == network_sizes 
                network_sizes_start = network_sizes
            mean_arr = np.array(mean_arr)
            std_arr = np.array(std_arr)
            if "dice" in performance_metric_column: 
                metric_mean = np.max(mean_arr, axis = 0)
                metric_std = std_arr[np.argmax(mean_arr, axis = 0), np.arange(mean_arr.shape[1])]
            elif "loss" in performance_metric_column: 
                metric_mean = np.min(mean_arr, axis = 0)
                metric_std = std_arr[np.argmin(mean_arr, axis = 0), np.arange(mean_arr.shape[1])]
                
            fig.plot(np.array(network_sizes) / precision, metric_mean, 
                marker = ".", label = "_".join(label))
            fig.fill_between(np.array(network_sizes) / precision, 
                                metric_mean - metric_std, 
                                metric_mean + metric_std, 
                                alpha = 0.4)
        fig.set_xlabel(f"Network Size (by {precision})")
        fig.set_ylabel(performance_metric_column)
        
        # fig.set_ylim(ylim_lower, ylim_upper)
            
        fig.set_title(title)
        fig.grid()
        fig.legend(prop = {"size": 6})



def plt_performance_vs_networksize_groupdfs_tidied(grouped_dfs_lst, label_prefixes, 
                                            performance_metric_column, precision, 
                                            fig, title): 
    # `grouped_dfs_dict`: a dictionary with group size as key and grouped model performance 
        # dataframe as value. Model performance dataframe grouped by "model_type", "aggregate_type", 
        # "fusion_type". 
    # `performance_metric_column`: column name of the performance metric. (y-axis of the plot) 

    if label_prefixes is not None: 
        assert len(grouped_dfs_lst) == len(label_prefixes), \
            f"The length of grouped dataframes should have the same length as label prefixes. But got " \
            f"{len(grouped_dfs_lst)} and {len(label_prefixes)}"

    
    performance_record = {}
    for i in range(len(grouped_dfs_lst)):

        grouped_dfs = grouped_dfs_lst[i]
        prefix = label_prefixes[i] if label_prefixes is not None else ""

        metric_stats = {}
        for (group_size, grouped_df) in grouped_dfs.items(): 
            grouped_df = sorted(grouped_df, key=lambda x: x[0])
            metric_stats[group_size] = {}

            for (k, v) in grouped_df: 
                model_key = k 
                model_df = v 

                network_sizes = sorted(model_df["network_size"].unique()) 
                performance_lst = []
                performance_lst.append(np.array(network_sizes))
                performance_metric_lst = []
                for network_size in network_sizes: 
                    networksize_df = v[v["network_size"] == network_size]
                    rungrouped_df = networksize_df.groupby("run_id")
                    rungrouped_df = sorted(rungrouped_df, key=lambda x: x[0])
                    if "loss" in performance_metric_column:
                        performance_metric_lst.append([i[1][performance_metric_column].min() for i in rungrouped_df])
                    elif "dice" in performance_metric_column: 
                        performance_metric_lst.append([i[1][performance_metric_column].max() for i in rungrouped_df])
                performance_lst.append(np.array(performance_metric_lst))

                metric_stats[group_size][model_key] = performance_lst

        model_keys_start = [] 
        for (group_size, grouped_df) in metric_stats.items(): 
            model_keys = sorted(metric_stats[group_size].keys()) 
            if len(model_keys_start) != 0: 
                assert model_keys_start == model_keys 
            model_keys_start = model_keys 
        
        for model_key in model_keys_start: 
            network_sizes = []
            performance_metric_lsts = []
            for group_size in sorted(metric_stats.keys()): 
                network_sizes.append(metric_stats[group_size][model_key][0])
                performance_metric_lsts.append(metric_stats[group_size][model_key][1])
            network_sizes, performance_metric_lsts = np.stack(network_sizes, axis=-1), \
                np.stack(performance_metric_lsts, axis=-1)
            assert (network_sizes.min(axis=-1) == network_sizes.max(axis=-1)).all()
            if len(prefix) != 0: 
                performance_record["_".join([prefix] + list(model_key))] = [network_sizes.min(axis=-1)]
            else: 
                performance_record["_".join(list(model_key))] = [network_sizes.min(axis=-1)]
            
            if len(prefix) != 0: 
                if "loss" in performance_metric_column:
                    performance_record["_".join([prefix] + list(model_key))].append(
                        performance_metric_lsts.min(axis=-1))
                elif "dice" in performance_metric_column: 
                    performance_record["_".join([prefix] + list(model_key))].append(
                        performance_metric_lsts.max(axis=-1))
            else: 
                if "loss" in performance_metric_column:
                    performance_record["_".join(list(model_key))].append(
                        performance_metric_lsts.min(axis=-1))
                elif "dice" in performance_metric_column: 
                    performance_record["_".join(list(model_key))].append(
                        performance_metric_lsts.max(axis=-1)) 
                    
    for label in performance_record.keys(): 
        network_sizes = performance_record[label][0]
        metric_mean = performance_record[label][1].mean(axis=-1)
        metric_std = performance_record[label][1].std(axis=-1)
        fig.plot(network_sizes / precision, metric_mean, 
            marker = ".", label = label)
        fig.fill_between(network_sizes / precision, 
                            metric_mean - metric_std, 
                            metric_mean + metric_std, 
                            alpha = 0.4)
    fig.set_xlabel(f"Network Size (by {precision})")
    fig.set_ylabel(performance_metric_column)
    fig.set_title(title)
    fig.grid()
    fig.legend(prop = {"size": 6})
    



def plt_performance_vs_groupsize_groupdfs(grouped_dfs, group_name, 
                                          constraints, summary_type,
                                          performance_metric_column, 
                                          fig, title, ylim_lower, ylim_upper): 
    # Passed in `grouped_dfs` is a dictionary with group size as key and grouped model performance 
        # dataframe as value. Model performance dataframe grouped by "model_type", "aggregate_type", 
        # "fusion_type", "constraints" (currently bs and lr). 
    
    if constraints is not None: 
        assert summary_type is not None
    
    start = [] 
    for (_, grouped_df) in grouped_dfs.items(): 
        keys = sorted([i[0] for i in grouped_df]) 
        if len(start) != 0: 
            assert start == keys 
            start = keys 
    
    grouped_dfs_filterbygroupname = {}
    for (group_size, grouped_df) in grouped_dfs.items(): 
        filtered_df = [i[1] for i in grouped_df if i[0] == group_name]
        assert len(filtered_df) == 1
        grouped_dfs_filterbygroupname[group_size] = filtered_df[0]

    metric_stats = {}
    for (group_size, grouped_df) in grouped_dfs_filterbygroupname.items(): 
        if constraints is None: 
            network_sizes = sorted(list(grouped_df["network_size"].unique()))
            metric_mean = []
            metric_std = []
            for network_size in network_sizes: 
                # print(len(v[v["network_size"] == network_size][performance_metric_column]))
                metric_mean.append(grouped_df[grouped_df["network_size"] == network_size][performance_metric_column].mean())
                metric_std.append(grouped_df[grouped_df["network_size"] == network_size][performance_metric_column].std())
            metric_mean = np.array(metric_mean)
            metric_std = np.array(metric_std) 
            network_sizes = np.array(network_sizes)
            metric_stats[group_size] = [network_sizes, metric_mean, metric_std]
            
        else: 
            grouped_df_constraints = grouped_df.groupby(constraints)
            grouped_df_constraints = sorted(grouped_df_constraints, key=lambda x: x[0])

            metric_mean_constraints = [] 
            metric_std_constraints = []
            network_sizes_start = []
            for (constraint, grouped_df_constraint) in grouped_df_constraints: 
                network_sizes = sorted(list(grouped_df_constraint["network_size"].unique()))

                # Sanity check that for a given model type, network sizes are the same for all constraints.
                if len(network_sizes_start) != 0: 
                    assert network_sizes_start == network_sizes
                network_sizes_start = network_sizes 

                metric_mean = []
                metric_std = []
                for network_size in network_sizes: 
                # print(len(v[v["network_size"] == network_size][performance_metric_column]))
                    metric_mean.append(grouped_df_constraint[grouped_df_constraint["network_size"] == network_size][performance_metric_column].mean())
                    metric_std.append(grouped_df_constraint[grouped_df_constraint["network_size"] == network_size][performance_metric_column].std())
                metric_mean = np.array(metric_mean)
                metric_std = np.array(metric_std) 
                metric_mean_constraints.append(metric_mean)
                metric_std_constraints.append(metric_std) 
            
            metric_mean_constraints = np.array(metric_mean_constraints)
            metric_std_constraints = np.array(metric_std_constraints)
            if summary_type == "best" and "dice" in performance_metric_column: 
                mean_arr = np.max(metric_mean_constraints, axis=0)
                std_arr = metric_std_constraints[np.argmax(metric_mean_constraints, axis = 0), 
                                                    np.arange(metric_mean_constraints.shape[1])]
            elif summary_type == "best" and "loss" in performance_metric_column: 
                mean_arr = np.min(metric_mean_constraints, axis=0)
                std_arr = metric_std_constraints[np.argmin(metric_mean_constraints, axis = 0), 
                                                    np.arange(metric_mean_constraints.shape[1])]
            elif summary_type == "average": 
                mean_arr, std_arr = metric_mean_constraints.mean(axis=0), \
                    metric_std_constraints.mean(axis=0)
            if summary_type in ["average", "best"]:
                metric_stats[group_size] = [network_sizes, mean_arr, std_arr]

    
    start = [] 
    for (_, metric_stat) in metric_stats.items(): 
        network_sizes = sorted(metric_stat[0]) 
        if len(start) != 0: 
            assert start == network_sizes 
            start = network_sizes 

    network_sizes = list(metric_stats.items())[0][1][0]
    for i in range(len(network_sizes)): 
        network_size = network_sizes[i]
        mean_arr = []
        std_arr = []
        group_sizes = []
        for (group_size, metric_stat) in metric_stats.items(): 
            mean_arr.append(metric_stat[1][i])
            std_arr.append(metric_stat[2][i])
            group_sizes.append(group_size)
        metric_mean = np.array(mean_arr)
        metric_std = np.array(std_arr) 
        group_sizes = np.array(group_sizes)

        group_sizes_sorted = np.sort(group_sizes)
        metric_mean = metric_mean[np.argsort(group_sizes)]
        metric_std = metric_std[np.argsort(group_sizes)]

        fig.plot(group_sizes_sorted, metric_mean, 
            marker = ".", label = network_size)
        fig.fill_between(group_sizes_sorted, 
                            metric_mean - metric_std, 
                            metric_mean + metric_std, 
                            alpha = 0.4)
        fig.set_xlabel(f"Group size")
        fig.set_ylabel(performance_metric_column)
                
    fig.set_title(title)
    fig.grid()
    fig.legend(prop = {"size": 6})




def plot_performance_by_networksize(result_dfs, eval_metric, class_name, precision, 
    colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k', 'w'], linestyles = None, label_postfixes = None, force_range_off = False, save_file = ""):
    if class_name != "all":
        loss_field = f"loss_{eval_metric}_class{class_name}"
    else: 
        loss_field = f"loss_{eval_metric}"
    for i in range(len(result_dfs)):
        result_df = result_dfs[i]
        if label_postfixes is not None:
            label_postfix = label_postfixes[i]
            if label_postfix[0] != "_":
                label_postfix = f"_{label_postfix}"
        else: 
            label_postfix = ""
        if linestyles is not None: 
            linestyle = linestyles[i]
        else: 
            linestyle = "-"
        
        if loss_field in result_df.columns: 
            plot_range = False 
        else: 
            assert f"{loss_field}_mean" in result_df.columns and f"{loss_field}_std" in result_df.columns, \
                f"Current DataFrame has columns {result_df.columns}, which does not include loss field of interest. Please double check!"
            plot_range = True

        for j in range(len(result_df["model_name"].unique())):
            # print(sorted(result_df["model_name"].unique())[j])
            model_name = sorted(result_df["model_name"].unique())[j]
            model_df = result_df[result_df["model_name"] == model_name]
            x = []
            y = []
            y_upper = []
            y_lower = []
            for model_size in sorted(model_df["model_size"].unique()):
                x.append(int(model_size))
                if plot_range:
                    avg_loss_min = model_df[model_df[f"model_size"] == model_size][f"loss_{eval_metric}_mean"].min()
                    loss_min = model_df[model_df[f"model_size"] == model_size][f"{loss_field}_mean"].min()
                    # loss_min = model_df[(model_df[f"model_size"] == model_size) & (model_df[f"loss_{eval_metric}_mean"] == avg_loss_min)][f"{loss_field}_mean"].tolist()[0]
                    y.append(loss_min)
                    loss_std = model_df[(model_df[f"model_size"] == model_size) & (model_df[f"{loss_field}_mean"] == loss_min)][f"{loss_field}_std"].tolist()[0]
                    # loss_std = model_df[(model_df[f"model_size"] == model_size) & (model_df[f"loss_{eval_metric}_mean"] == avg_loss_min)][f"{loss_field}_std"].tolist()[0]
                    y_upper.append(loss_min + loss_std)
                    y_lower.append(loss_min - loss_std)
                else:
                    avg_loss_min = model_df[model_df[f"model_size"] == model_size][f"loss_{eval_metric}"].min()
                    # loss_min = model_df[(model_df[f"model_size"] == model_size) & (model_df[f"loss_{eval_metric}"] == avg_loss_min)][loss_field].iloc[0]
                    loss_min = model_df[model_df[f"model_size"] == model_size][loss_field].min()
                    y.append(loss_min)
            # print(x)
            # print(y)
            if model_name == "simpleunet":
                assert len(model_df["gs"].unique()) == 1 and int(model_df["gs"].unique()[0]) == 1, \
                    f"Current model: SimpleUNet. Expect records to have only training group size of 1. But got: {model_df['gs'].unique()}."
                if len(model_df["bs"].unique()) == 1: 
                    plt.plot([ms // precision for ms in x], y, linestyle = linestyle, color = colors[j], marker = ".", 
                        label = model_name + f"_bs{model_df['bs'].unique()[0]}_gs1" + label_postfix)
                else: 
                    plt.plot([ms // precision for ms in x], y, linestyle = linestyle, color = colors[j], marker = ".", 
                        label = model_name + f"_bsall_gs1" + label_postfix)
            else: 
                assert len(model_df["bs"].unique()) == 1, \
                    f"Current model: {model_name}. Expect records to have only 1 training batch size. But got: {model_df['bs'].unique()}."
                assert len(model_df["gs"].unique()) == 1, \
                    f"Current model: {model_name}. Expect records to have only 1 training group size. But got: {model_df['gs'].unique()}."
                plt.plot([ms // precision for ms in x], y, linestyle = linestyle, color = colors[j], marker = ".", 
                    label = model_name + f"_bs{model_df['bs'].unique()[0]}_gs{model_df['gs'].unique()[0]}" + label_postfix)
            if plot_range and not force_range_off: 
                plt.fill_between([ms // precision for ms in x], y_upper, y_lower, alpha = 0.2, color = colors[j])
        plt.xlabel(f"Number of Params (by {precision})")
        if eval_metric == "harddice":
            plt.ylabel("Hard DiceLoss")
            plt.title("Hard DiceLoss vs. Network Size")
        elif eval_metric == "softdice":
            plt.ylabel("Soft DiceLoss")
            plt.title("Soft DiceLoss vs. Network Size")
        elif eval_metric == "lossfunc":
            plt.ylabel("Training Objective Loss")
            plt.title("Training Objective Loss vs. Network Size")
    plt.grid()
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width * 0.5,box.height])
    plt.legend(bbox_to_anchor=(1, 0, 0.5, 1), fontsize = "small")
    plt.savefig(save_file)
    plt.close()

def plot_performance_by_groupsize(model_dfs, eval_metric, class_name, 
    colors = ['b', 'g', 'r', 'y', 'm', 'c', 'k', 'w'], linestyles = None, label_postfixes = None, force_range_off = False, save_file = ""):
    if class_name != "all":
        loss_field = f"loss_{eval_metric}_class{class_name}"
    else: 
        loss_field = f"loss_{eval_metric}"
    for i in range(len(model_dfs)):
        model_df = model_dfs[i]
        model_names = model_df["model_name"].unique()
        assert len(model_names) == 1, f"Plotting vs. group size. Expect only 1 model name per input dataframe. But got: {model_names}."
        model_name = model_names[0]
        if label_postfixes is not None:
            label_postfix = label_postfixes[i]
            if label_postfix[0] != "_":
                label_postfix = f"_{label_postfix}"
        else: 
            label_postfix = ""
        if linestyles is not None: 
            linestyle = linestyles[i]
        else: 
            linestyle = "-"
        
        if loss_field in model_df.columns: 
            plot_range = False 
        else: 
            assert f"{loss_field}_mean" in model_df.columns and f"{loss_field}_std" in model_df.columns, \
                f"Current DataFrame has columns {model_df.columns}, which does not include loss field of interest. Please double check!"
            plot_range = True

        for j in range(len(model_df["model_size"].unique())):
            model_size = sorted(model_df["model_size"].unique())[j]
            model_df2 = model_df[model_df["model_size"] == model_size]
            x = []
            y = []
            y_upper = [] 
            y_lower = []
            for test_gs in sorted(model_df2["test_gs"].unique()):
                x.append(int(test_gs)) 
                if plot_range:
                    loss_min = model_df2[model_df2[f"test_gs"] == test_gs][f"{loss_field}_mean"].min()
                    y.append(loss_min)
                    loss_std = model_df2[(model_df2[f"test_gs"] == test_gs) & (model_df2[f"{loss_field}_mean"] == loss_min)][f"{loss_field}_std"].tolist()[0]
                    y_upper.append(loss_min + loss_std)
                    y_lower.append(loss_min - loss_std)
                else:
                    y.append(model_df2[model_df2["test_gs"] == test_gs][loss_field].min())
            if model_name == "simpleunet":
                assert len(model_df["gs"].unique()) == 1 and int(model_df["gs"].unique()[0]) == 1, \
                    f"Current model: SimpleUNet. Expect records to have only training group size of 1. But got: {model_df['gs'].unique()}."
                if len(model_df["bs"].unique()) == 1: 
                    plt.plot(x, y, linestyle = linestyle, color = colors[j], marker = ".", 
                    label = str(model_size) + f"_bs{model_df['bs'].unique()[0]}_gs1" + label_postfix)
                else: 
                    plt.plot(x, y, linestyle = linestyle, color = colors[j], marker = ".", 
                        label = str(model_size) + f"_bsall_gs1" + label_postfix)
                # plt.plot(x, y, linestyle = linestyle, color = colors[j], marker = ".", 
                #     label = str(model_size) + re.sub(r"gs", "bs", label_postfix))
            else: 
                assert len(model_df["bs"].unique()) == 1, \
                    f"Current model: {model_name}. Expect records to have only 1 training batch size. But got: {model_df['bs'].unique()}."
                assert len(model_df["gs"].unique()) == 1, \
                    f"Current model: {model_name}. Expect records to have only 1 training group size. But got: {model_df['gs'].unique()}."
                plt.plot(x, y, linestyle = linestyle, color = colors[j], marker = ".", 
                    label = str(model_size) + f"_bs{model_df['bs'].unique()[0]}_gs{model_df['gs'].unique()[0]}" + label_postfix)
            if plot_range and not force_range_off: 
                plt.fill_between(x, y_upper, y_lower, alpha = 0.2, color = colors[j])
        plt.xlabel(f"Group Size")
        if eval_metric == "harddice":
            plt.ylabel("Hard DiceLoss")
            plt.title("Hard DiceLoss vs. Group Size")
        elif eval_metric == "softdice":
            plt.ylabel("Soft DiceLoss")
            plt.title("Soft DiceLoss vs. Group Size")
    plt.grid()
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width * 0.5,box.height])
    plt.legend(bbox_to_anchor=(1, 0, 0.5, 1), fontsize = "small")
    plt.savefig(save_file)
    plt.close()