import os, glob, toml 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from structDict import recurseCreateStructDict


def generate_FIRdesignMat_template_toml(output_dir): 
    """
    Config keys:
    This function generates a toml config file for the FIR design matrix 
    runs: A list of name of the tasks or conditions for each fmri run. 
          the order should be IDENTICAL to how they will be concatenated when running the GLMs. 
    regressors: The name of the regressors (e.g., A TR within a block is usually one of the 
                3 components: instruction, task, and IBI), the names are only for clarity.
                the regressors do not need to cover the entire block. 
                for example, you can only modle the first 36 TRs for each 40 TR block
    run_leadingin_tr: the number of leading in TR for each run/scan, 0 if none 
    run_leadingout_tr: the number of leading out TR for each run/scan, 0 if none
    epoch_per_run: the number of block/epcohs per run 

    additional configuration keys for creating FIR matrix accounting for motion spikes 
    fmriprep_dir: the path to fmriprep derivative directory
    spike_cutoff: the threshold to ignore a frame in the designmatrix 
    prop_spike_cutoff: between 0 and 100, if the percentage of spikes in a run is greater than this cutoff, 
                       drop this run. 

    Important: 
    The length of each run should be run_leadingin_tr + epoch_per_run*epoch_tr + run_leadingout_tr
    """
    data = {
        "runs": "" #list
        ,"fir_regressors": "" #list
        ,"epoch_tr": "" #int 
        ,"run_leadingin_tr": "" #int
        ,"run_leadingout_tr": "" #int 
        ,"epoch_per_run": "" #int

        ,"fmriprep_dir": "" #str, path 
        ,"spike_cutoff": "" #float
        ,"prop_spike_cutoff": "" #float
    }

    # Write the data to the TOML file
    with open(os.path.join(output_dir,"FIRdesignMat_conf.toml"), "w") as toml_file:
        toml.dump(data, toml_file)

def write_FIRdesginMat(cfg_dir, output_dir): 

    """
    
    """


    # load configuration file 
    with open(cfg_dir, "r") as toml_file:
        cfg = toml.load(toml_file)
    cfg = recurseCreateStructDict(cfg)

    # combine task and regressor names to get all regressor (task*regressor)
    all_regressors = []
    for this_task, this_reg in product(cfg.runs, cfg.fir_regressors):
        all_regressors.append(this_task + "_" + this_reg)

    # FIR design matrix for each condition (should be identical across conditions)
    per_block_fir = np.eye(cfg.epoch_tr,len(cfg.fir_regressors)) # for each block (should be identical across each block)
    leadingin_dummy = np.zeros([cfg.run_leadingin_tr,len(cfg.fir_regressors)]) # leading in TRs, 
    leadingout_dummy = np.zeros([cfg.run_leadingout_tr,len(cfg.fir_regressors)]) # leading out TRs, 
    run_struct = np.ones([cfg.epoch_per_run,1]) # the structure of a run, in terms of blocks 
    per_run_fir = np.kron(run_struct,per_block_fir)
    per_run_fir_full = np.concatenate((leadingin_dummy, per_run_fir, leadingout_dummy), axis = 0) # the shape is run_TR x nReg

    # Each run should have the same structure     
    task_struct = np.eye(len(cfg.runs),len(cfg.runs)) 
    full_fir = np.kron(task_struct, per_run_fir_full)

    df = pd.DataFrame(full_fir, columns = all_regressors)
    df.to_csv(os.path.join(output_dir,'fir_design_matrix.txt'), header = False, index = False, sep = ' ')

    fig = plt.figure(figsize = (40,40))
    sns.heatmap(df)
    fig.savefig(os.path.join(output_dir,'fir_design_heatmap.png'))

