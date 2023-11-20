import sys
import os, glob, toml 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from .structDict import recurseCreateStructDict


def generate_FIRdesignMat_template_toml(output_dir): 
    """
This is the configuration file for constructing FIR design matrix. You need to specify all the parameters listed below in order to build the FIR design matrix for your need. You can refer to the comments below for more information regarding each parameter. If you do not need to account for motion (i.e., running only 'write_vanilla_FIRdesginMat'), you won't need to fill out anything below 'epoch_per_run'.
PARAMETERS:
- "conditions": "A list of conditions; The order should be IDENTICAL to how they will be concatenated when running the GLMs." 
- "rep": "The number of runs each condition is repeated"
- "fir_regressors": "The name of the regressors (e.g., A TR within a block is usually one of the 3 components: instruction, task, and IBI), the names are only for clarity. The regressors do not need to cover the entire block. For example, you can only modle the first 36 TRs for each 40 TR block" 
- "epoch_tr": "The number of TR within each block/epoch" 
- "run_leadingin_tr": "The number of leading in TR" 
- "run_leadingout_tr": "The number of leading out TR" 
- "epoch_per_run": "The number of blocks/epochs within each run" 
- "fmriprep_dir": "Where fmriprep derivative is located"  
- "spike_cutoff": "the threshold to ignore a frame in the designmatrix" 
- "prop_spike_cutoff": "between 0 and 100, if the percentage of frames within a run has fd > fd_cutoff is greater than prop_spike_cutoff, then remove the run"
- "sub_id": "subject id, naming convention should follow how the subject folder is named in the fMRIprep derivative folder, do not need to include the 'sub-' prefix"
- "order": "this is a dictionary, with keys being the name of the run and values being its order. This is necessary to locate all the confound files in the derivative folder and to sort them in the same order as they will be concatenated and modeled. The name of a fMRIprep output confound file is sub-%s_task-%s_run-%s_desc-preproc_bold.nii, the keys here need to be specified as'task-%s_run-%s'. For example, 'task-divPerFacePerTone_run-2'". 
Important: The length of each run should be run_leadingin_tr + epoch_per_run*epoch_tr + run_leadingout_tr
    """
    data = {
        
        "HOW TO USE":"This is the configuration file for constructing FIR design matrix. You need to specify all the parameters listed below in order to build the FIR design matrix for your need. You can refer to the comments below for more information regarding each parameter. If you do not need to account for motion (i.e., running only 'write_vanilla_FIRdesginMat'), you won't need to fill out anything below 'epoch_per_run'."

        ,"PARAMETERS":{
            "conditions": []
            ,"rep": ""
            ,"fir_regressors": [] 
            ,"epoch_tr": "" 
            ,"run_leadingin_tr": "" 
            ,"run_leadingout_tr": "" 
            ,"epoch_per_run": "" 
            ,"fmriprep_dir": "" 
            ,"fd_cutoff": ""
            ,"spike_cutoff": "" 
            ,"prop_spike_cutoff": "" 
            ,"sub_id": "" 
            ,"order": "" 
        }
        
        ,"COMMENTS":{
            "conditions": "A list of conditions; The order should be IDENTICAL to how they will be concatenated when running the GLMs." 
            ,"rep": "The number of runs each condition is repeated"
            ,"fir_regressors": "The name of the regressors (e.g., A TR within a block is usually one of the 3 components: instruction, task, and IBI), the names are only for clarity. The regressors do not need to cover the entire block. For example, you can only modle the first 36 TRs for each 40 TR block" 
            ,"epoch_tr": "The number of TR within each block/epoch" 
            ,"run_leadingin_tr": "The number of leading in TR" 
            ,"run_leadingout_tr": "The number of leading out TR" 
            ,"epoch_per_run": "The number of blocks/epochs within each run" 
            ,"fmriprep_dir": "Where fmriprep derivative folder is located"  
            ,"fd_cutoff": "if the percentage of frames within a run has fd > fd_cutoff is greater than prop_spike_cutoff, then remove the run"
            ,"spike_cutoff": "the threshold to ignore a frame in the designmatrix " 
            ,"prop_spike_cutoff": "between 0 and 100, if the percentage of frames within a run has fd > fd_cutoff is greater than prop_spike_cutoff, then remove the run"
            ,"sub_id": "subject id, naming convention should follow how the subject folder is named in the fMRIprep derivative folder, do not need to include the 'sub-' prefix"
            ,"order": "this is a dictionary, with keys being the name of the run and values being its order. This is necessary to locate all the confound files in the derivative folder and to sort them in the same order as they will be concatenated and modeled. The name of a fMRIprep output confound file is sub-%s_task-%s_run-%s_desc-preproc_bold.nii, the keys here need to be specified as'task-%s_run-%s'. For example, 'task-divPerFacePerTone_run-2'"
        }
    }

    # Write the data to the TOML file
    with open(os.path.join(output_dir,"FIRdesignMat_conf.toml"), "w") as toml_file:
        toml.dump(data, toml_file)
        
def _generate_vanilla_designMat(cfg_dir): 
    
    """
    This function generates the vanilla design matrix for all subjects, not accounting for motion or bad runs. This function is specifically aiming for block designed, FIR design matrix. This is for all subjects since it requires the fMRI runs to be concatenated in the same order across all subjects, although they were run in different order in the scanner. Moreover, the runs should be identical to each other and all blocks within in a run are of the same condition. Make sure to name the runs by the conditions during the R&D session, so later you will know the condition of each scan, which makes everything a lot easier. 
    Each FIR regressor models a specific "type" of TR. For example, a FIR regressor may model the first TR of every block/epoch of condition1. You may chose to model all TR within a block, or chose to model a subset of TRs. For example, if a block/epoch has 40 TRs in total, you can chose to model the first 36 TRs thats totally fine. As a result, you end up having (#_of_conditions * #_of TRs_modeled) number of regressors. 
    The final design matrix has the shape (total_#_of_TR_all_scans * #_of_regressors). 
    PARAMETER: cfg_dir: the path for the toml configuration file 
    OUTPUT:
    - full_fir: numpy array 
    - all_regressors: list of regressor names. 
    """
    # load configuration file 
    with open(cfg_dir, "r") as toml_file:
        cfg = toml.load(toml_file)
    cfg = recurseCreateStructDict(cfg)
    cfg = cfg.PARAMETERS

    # combine task and regressor names to get all regressor (#_of_conditions * #_of TRs_modeled)
    all_regressors = []
    for this_task, this_reg in product(cfg.conditions, cfg.fir_regressors):
        all_regressors.append(this_task + "_" + this_reg)

    # FIR design matrix for each condition (each condition may contain multiple runs)
    per_block_fir = np.eye(cfg.epoch_tr,len(cfg.fir_regressors)) # for each block (block structures should be all identical)
    leadingin_dummy = np.zeros([cfg.run_leadingin_tr,len(cfg.fir_regressors)]) # leading in TRs, 
    leadingout_dummy = np.zeros([cfg.run_leadingout_tr,len(cfg.fir_regressors)]) # leading out TRs, 
    run_struct = np.ones([cfg.epoch_per_run,1]) # the structure of a run, in terms of blocks 
    per_run_fir = np.kron(run_struct,per_block_fir)
    per_run_fir_full = np.concatenate((leadingin_dummy, per_run_fir, leadingout_dummy), axis = 0) # design matrix for a run
    per_run_fir_full = np.concatenate([per_run_fir_full]*cfg.rep) # design matrix for a condition
    
    # Each condition should have the same structure     
    task_struct = np.eye(len(cfg.conditions),len(cfg.conditions)) 
    full_fir = np.kron(task_struct, per_run_fir_full)
    
    return full_fir, all_regressors
    

def write_vanilla_FIRdesginMat(cfg_dir, output_dir): 

    """
    This function writes out the vanilla FIR design matrix to the output_dir. It also plots it out so you can eyeball the structure to make sure it looks correct. 
    """
    # generate vanilla fir matrix
    vanilla_fir, all_regressors = _generate_vanilla_designMat(cfg_dir)
    
    # write it out 
    df = pd.DataFrame(vanilla_fir, columns = all_regressors)
    df.to_csv(os.path.join(output_dir,'fir_design_matrix.txt'), header = False, index = False, sep = ' ')
    
    # plot it 
    fig = plt.figure(figsize = (40,40))
    sns.heatmap(df)
    fig.savefig(os.path.join(output_dir,'fir_design_heatmap.png'))


def write_personalized_FIRdesginMat(cfg_dir, output_dir): 
    
    """
    This function personlize FIR design matrix for each participant, aims to remove bad runs and bad frames from the FIR model. 
    Bad runs and bad frames were defined based on the framewise-displacement confound outputed by fMRIprep. The implementation here is that, 1) if more than prop_spike_cutoff (e.g., 5%) of frames with in a run has fd > fd_cutoff (e.g., 0.5), then all regressors will be 0 for all TRs within this run, 2) if the run is good, then look at each frame, if the frame has fd > spike_cutoff (e.g., 2), then all regressors of this frame and its preceding and following frames will be 0. 
    """
    # -----------------------
    # load configuration file and check required field from the configuration file 
    with open(cfg_dir, "r") as toml_file:
        cfg = toml.load(toml_file)
    required_fields = ["fmriprep_dir","fd_cutoff","spike_cutoff","prop_spike_cutoff","sub_id","order"]
    for f in required_fields: 
        if not cfg['PARAMETERS'][f]: 
            raise KeyError(f"{f} not specified in the configuration file")
    cfg = recurseCreateStructDict(cfg)
    cfg = cfg.PARAMETERS
    order = eval(cfg.order)
    
    # -----------------------
    # get confound files for the participant, order it as how 12 runs are being concatenated. 
    fir_design_matrix, all_regressors = _generate_vanilla_designMat(cfg_dir)
    sub_dir = os.path.join(cfg.fmriprep_dir, f'sub-{cfg.sub_id}/func')
    confound_files = glob.glob(os.path.join(sub_dir, "*_run-*_desc-confounds_timeseries.tsv")) # this is the standard format of fmriprep so should be generalizable
    # the standard format is sub-%s_task-%s_run-%s_desc-preproc_bold.nii, 
    # so by doing x.split("_")[1:3], I carve out 'task-%s_run-%s', 
    # so make sure you have the keys in this way when passing in the order. 
    # sort it as how the runs will be concatenated. 
    confound_files.sort(key=lambda x: order['_'.join(x.split("_")[1:3])]) 
    
    # -----------------------
    # Look for spikes or bad motion runs. 
    # len(ts)=total_#_of_TR_all_scans, if 0, then remove the corresponding column in the FIR matrix. 
    ts = []
    for f in confound_files: 
        df = pd.read_csv(f,sep = '\t')
        motion = list(df['framewise_displacement']) # get FD from the fMRIprep confound file. 
        outlier_prop = round(np.sum(np.array(motion)>cfg.fd_cutoff)/len(motion)*100,2) # %of frames that have FD>0.5
        
        if outlier_prop > cfg.prop_spike_cutoff: # remove the whole run (all frames are bad frames) 
            ts_run = [0 for _ in range(len(motion))]
        else: # otherwise look for spike 
            ts_run = [1 for _ in range(len(motion))] # at the beginning, all timepoints are good frames. 
            for idx, m in enumerate(motion): 
                if np.isnan(m) or m <= cfg.spike_cutoff: pass # if this is the frist frame (motion is np.nan for the first frame) or not a spike 
                if m > cfg.spike_cutoff: # if spike 
                    ts_run[idx-1], ts_run[idx] = 0, 0 # remove the previous and the current frame 
                    if idx + 1 < len(motion): # if not the last frame, remove the next frame 
                        ts_run[idx+1] = 0 
        ts += ts_run 
    
    # -----------------------
    # remove spikes or bad motion runs. 
    # if bad frame, remove the frame from FIR design matrix (all 0 for all regressors) 
    for idx, frame in enumerate(ts): 
        if frame == 1: 
            pass 
        else: 
            fir_design_matrix[idx,:] = 0 
    
    # write it out 
    df = pd.DataFrame(fir_design_matrix, columns = all_regressors)
    df.to_csv(os.path.join(output_dir,f'sub-{cfg.sub_id}_fir_design_matrix.txt'), header = False, index = False, sep = ' ')
    
    # plot it 
    fig = plt.figure(figsize = (40,40))
    sns.heatmap(df)
    fig.savefig(os.path.join(output_dir,f'sub-{cfg.sub_id}_fir_design_heatmap.png'))
  