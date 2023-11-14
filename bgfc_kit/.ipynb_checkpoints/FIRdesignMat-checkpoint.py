"""
TODO: 
1. configuration file needs to take in another "rep" parameter. otherwise run*reg is wrong 
2. work on the motion correction part 
"""
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
    This function generates a toml config file for the FIR design matrix 
    Important: The length of each run should be run_leadingin_tr + epoch_per_run*epoch_tr + run_leadingout_tr
    """
    data = {

        "PARAMETERS":{
            "conditions": "" 
            ,"rep": ""
            ,"fir_regressors": "" 
            ,"epoch_tr": "" 
            ,"run_leadingin_tr": "" 
            ,"run_leadingout_tr": "" 
            ,"epoch_per_run": "" 
            ,"fmriprep_dir": "" 
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
            ,"fmriprep_dir": "Where fmriprep derivative is located"  
            ,"spike_cutoff": "the threshold to ignore a frame in the designmatrix " 
            ,"prop_spike_cutoff": "between 0 and 100, if the percentage of spikes in a run is greater than this cutoff drop this run. " 
            ,"sub_id": "" 
            ,"order": "" 
        }
    }

    # Write the data to the TOML file
    with open(os.path.join(output_dir,"FIRdesignMat_conf.toml"), "w") as toml_file:
        toml.dump(data, toml_file)

def write_FIRdesginMat(cfg_dir, output_dir): 

    """
    This is specific to block design, with all fMRI runs being identical, and only the task/condition
    being performed differ across runs. In this design, each RUN consists of numbers of EPOCHS (block)
    of the same condition. 
    FIR model tries to model specified TR within epochs for each task/condition. For example, if there 
    are 6 tasks, and I would like to model the first 36 TRs within each epoch, then i would have 36*6 regressor. 

    Parameters: 
    cfg_dir: the path for the toml configuration file 
    output_dir: where the design matrix is writing to 
    """

    # load configuration file 
    with open(cfg_dir, "r") as toml_file:
        cfg = toml.load(toml_file)
    cfg = recurseCreateStructDict(cfg)
    cfg = cfg.PARAMETERS

    # combine task and regressor names to get all regressor (task*regressor)
    all_regressors = []
    for this_task, this_reg in product(cfg.conditions, cfg.fir_regressors):
        all_regressors.append(this_task + "_" + this_reg)

    # FIR design matrix for each condition (should be identical across conditions)
    per_block_fir = np.eye(cfg.epoch_tr,len(cfg.fir_regressors)) # for each block (should be identical across each block)
    leadingin_dummy = np.zeros([cfg.run_leadingin_tr,len(cfg.fir_regressors)]) # leading in TRs, 
    leadingout_dummy = np.zeros([cfg.run_leadingout_tr,len(cfg.fir_regressors)]) # leading out TRs, 
    run_struct = np.ones([cfg.epoch_per_run,1]) # the structure of a run, in terms of blocks 
    per_run_fir = np.kron(run_struct,per_block_fir)
    per_run_fir_full = np.concatenate((leadingin_dummy, per_run_fir, leadingout_dummy), axis = 0) # the shape is run_TR x nReg
    per_run_fir_full = np.concatenate([per_run_fir_full]*cfg.rep)
    
    # Each run should have the same structure     
    task_struct = np.eye(len(cfg.conditions),len(cfg.conditions)) 
    full_fir = np.kron(task_struct, per_run_fir_full)

    df = pd.DataFrame(full_fir, columns = all_regressors)
    df.to_csv(os.path.join(output_dir,'fir_design_matrix.txt'), header = False, index = False, sep = ' ')

    fig = plt.figure(figsize = (40,40))
    sns.heatmap(df)
    fig.savefig(os.path.join(output_dir,'fir_design_heatmap.png'))


# def personalized_fir(cfg_dir, output_dir): 
    
#     """
#     This function personlize FIR design matrix for each participant, aims to remove bad runs and bad frames 
#     from the FIR model. 
#     Required parameters in the configuration file, in addition to all the parameters needed for the above function: 
#     - fmriprep_dir: directory to the fmriprep derivative folder 
#     - spike_cutoff: cutoff for a spike (float)
#     - prop_spike_cutoff: cutoff for the proportion of spikes within a run to drop that run (float)
#     - sub_id: subject id, should be your subject folder in the fmriprep derivative directory
#     order: the order that the 12 runs are being concatenated 
#     run_prop: the proportion of frames with FD > 0.5 within each run, exceeding which removes the run (5%)
#     spike: the spike cutoff (2mm)
#     fir_design_matrix: 3984 x 216 np array (3984 timepoints; 216 regressors) 
#     """
#     # load configuration file and check required field from the configuration file 
#     with open(cfg_dir, "r") as toml_file:
#         cfg = toml.load(toml_file)
#     required_fields = ["fmriprep_dir","spike_cutoff","prop_spike_cutoff","sub_id","order"]
#     for f in required_fields: 
#         if not cfg[f]: 
#             raise KeyError(f"{f} not specified in the configuration file")
#     cfg = recurseCreateStructDict(cfg)

#     # get confound files for the participant, order it as how 12 runs are being concatenated. 
#     fir_design_matrix = np.loadtxt(fir_design)
#     fmriprep_dir = '/projects/hulacon/peetal/divatten/derivative/'
#     sub_dir = os.path.join(fmriprep_dir, f'sub-{sub:03d}/func')
#     confound_files = glob.glob(os.path.join(sub_dir, "*_run-*_desc-confounds_timeseries.tsv"))
#     confound_files.sort(key=lambda x: order['_'.join(x.split("_")[1:3])])
    
#     # len(ts)=3984, if 0, then remove the corresponding column in the FIR matrix. 
#     ts = []
#     for f in confound_files: 
#         df = pd.read_csv(f,sep = '\t')
#         motion = list(df['framewise_displacement']) # get FD 
#         outlier_prop = round(np.sum(np.array(motion)>0.5)/len(motion)*100,2) # %of frames that have FD>0.5
        
#         if outlier_prop > run_prop: # remove the whole run (all frames are bad frames) 
#             ts_run = [0 for _ in range(len(motion))]
#         else:   
#         # otherwise check spike, but at the beginning, all timepoints are good frames. 
#             ts_run = [1 for _ in range(len(motion))]
#             for idx, m in enumerate(motion): 
#                 if np.isnan(m) or m <= spike: pass # if the frist frame or not a spike 
#                 if m > spike: # if spike 
#                     ts_run[idx-1], ts_run[idx] = 0, 0 # remove the previous and the current frame 
#                     if idx + 1 < len(motion): # if not the last frame, remove the next frame 
#                         ts_run[idx+1] = 0 
#         ts += ts_run 
    
#     # check length 
#     if len(ts) != 3984: 
#         print("Timeseries length is not correct")
#         return 
    
#     # if bad frame, remove the frame from FIR design matrix (all 0 for all regressors) 
#     for idx, frame in enumerate(ts): 
#         if frame == 1: 
#             pass 
#         else: 
#             fir_design_matrix[idx,:] = 0 
            
#     tasks = ['divPerFacePerTone', 'divPerFaceRetScene', 'divRetScenePerTone', 'singlePerFace', 'singlePerTone', 'singleRetScene']
#     reg = ['inst1', 'inst2', 'inst3', 'inst4', 'task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7', 'task8', 'task9',
#            'task10', 'task11', 'task12', 'task13', 'task14', 'task15', 'task16', 'task17', 'task18', 'task19', 'task20', 'task21',
#            'task22', 'task23', 'task24', 'IBI1', 'IBI2', 'IBI3', 'IBI4', 'IBI5', 'IBI6', 'IBI7', 'IBI8']
#     all_regressors = []
#     for this_task, this_reg in product(tasks, reg):
#         all_regressors.append(this_task + "_" + this_reg)
            
#     df = pd.DataFrame(fir_design_matrix, columns = all_regressors)
#     df.to_csv(f'/projects/hulacon/peetal/divatten/preprocess/personalized_FIR_matrix/sub-{sub:03d}_fir_design_matrix.txt', header = False, index = False, sep = ' ')

#     fig = plt.figure(figsize = (40,40))
#     sns.heatmap(df)
#     fig.savefig(f'/projects/hulacon/peetal/divatten/preprocess/personalized_FIR_matrix/sub-{sub:03d}_fir_design_heatmap.png')

# for i in range(32,38): 
#     if i == 29: 
#         continue 
#     personalized_fir(i, order, fir_design)
