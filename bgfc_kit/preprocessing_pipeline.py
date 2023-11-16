import os, glob, toml 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from .structDict import recurseCreateStructDict
from itertools import chain, product
import subprocess



def generate_postfMRIprep_pipeline_template_toml(output_dir): 
    
    data = {

        "PARAMETERS":{

            "sub_id": ''
            ,"task_id": ''
            ,"space": ''
            ,"base_dir": ''
            ,"output_dir": ''
            ,"designMat_dir": ''
            ,"runRest_tr": ''
            ,"fwhm": ''
            ,"hpcutoff": ''
            ,"nproc": ''
        }


        ,"COMMENTS":{

            "sub_id": "subject id; This is placed at %s following 'sub-'; be consistent with fMRIprep naming convention: sub-%s_task-%s_space-%s_desc-preproc_bold.nii.gz"
            ,"task_id": "A list of tasks; This is placed at %s following 'task-'. IMPORTANT: make sure the order you provide is consistent with the design matrix"
            ,"space": "This be placed at %s following 'space-' (e.g., MNI152NLin2009cAsym_res-2)"
            ,"base_dir": "where fMRIPrep derivative folder is at" 
            ,"output_dir": "output directory"
            ,"designMat_dir": "directory for the FIR design matrix" 
            ,"runRest_tr": "list of TRs that are 'rest TR', will serve as baseline activity level"
            ,"fwhm": "smoothing kernel size"
            ,"hpcutoff": "high pass filter cut off, by setting default value being 50, high pass filter is 100"
            ,"nproc": "multithreading"
        }
    }

    # Write the data to the TOML file
    with open(os.path.join(output_dir,"postfMRIprep_pipeline_config.toml"), "w") as toml_file:
        toml.dump(data, toml_file)


def run_postfMRIprep_pipeline(cfg_dir): 

    """
    This function takes in a configuration file and then run the post-fMRIprep preprocessing pipeline 
    """

    # Get the path of the current script
    current_script_path = os.path.realpath(__file__)

    # Construct the path to the script within the library
    library_path = os.path.dirname(current_script_path)
    script_path = os.path.join(library_path, "scripts", "post_fMRIPrep_pipeline.py")

    # load configuration file 
    with open(cfg_dir, "r") as toml_file:
        cfg = toml.load(toml_file)
    cfg = recurseCreateStructDict(cfg)
    cfg = cfg.PARAMETERS
    
    task_id=""
    for task in cfg.task_id: 
        task_id += task
        task_id += " "
    rest_tr=""
    for tr in eval(cfg.runRest_tr): 
        rest_tr += str(tr) 
        rest_tr += " "
    
    
    # run the python script 
    try:
        command = f"python3 {script_path} --sub-id {cfg.sub_id} --task-id {task_id} --space {cfg.space} --base-dir {cfg.base_dir} --output-dir {cfg.output_dir} --designMat-dir {cfg.designMat_dir} --run-restTR {rest_tr} --fwhm {cfg.fwhm} --hpcutoff {cfg.hpcutoff} --nproc {cfg.nproc}"
        print(f"running the command {command}")
        subprocess.run(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")

#def submit_postfMRIprep_pipeline_SLURM(config_dir, partition):
