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
    
    """
    This is the configuration file for post-fMRIprep preprocessing pipeline. This pipeline includes 1) smoothing and high-pass filterinng, 2) nuisance GLM using FSL FEAT implemented with nipype, 3) demean of each run (i.e., zscoring the whole run using the mean and sd of the 'resting TRs', which will be defined by your configuration file), 
    and concatenating all runs into one long timeseries (the order of concatenation is crucial, it should be identical to your FIR design matrix, which will be defined by your configuration file), and 4) FIR glm. You need to specify all the parameters listed below in order to run the pipeline. The important output files are 
    1) evoked timeseries after regressing out confounds, which can be found sub-xx/before_FIR folder, 2) residual timeseries after regressing out stimulus evoked activities, which can be found sub-xx/FIR_residual folder, and 3) FIR regressor beta values, which can be found sub-xx/FIR_betas.
    
    Function parameters
    --------------------
    output_dir: 
        Where the template configuration file will be created at. 

    Configuration parameters
    ------------------------
    sub_id: 
        subject id; This is placed at %s following 'sub-'; be consistent with fMRIprep naming convention: sub-%s_task-%s_space-%s_desc-preproc_bold.nii.gz
    task_id: 
        A list of tasks; This is placed at %s following 'task-'. IMPORTANT: make sure the order you provide is consistent with the design matrix
    space: 
        This be placed at %s following 'space-' (e.g., MNI152NLin2009cAsym_res-2)
    base_dir:  
        Where fMRIPrep derivative folder is at 
    output_dir: 
        Output directory
    designMat_dir: 
        Directory for the FIR design matrix
    runRest_tr: 
        List of TRs that are 'rest TR', will serve as baseline activity level
    fwhm: 
        Smoothing kernel size
    hpcutoff: 
        High pass filter cut off, by setting default value being 50, high pass filter is 100
    nproc: 
        Multithreading
    """
    
    data = {
        
         "HOW TO USE":"This is the configuration file for post-fMRIprep preprocessing pipeline. This pipeline includes 1) smoothing and high-pass filterinng, 2) nuisance GLM using FSL FEAT implemented with nipype, 3) demean of each run (i.e., zscoring the whole run using the mean and sd of the 'resting TRs', which will be defined by your configuration file), and concatenating all runs into one long timeseries (the order of concatenation is crucial, it should be identical to your FIR design matrix, which will be defined by your configuration file), and 4) FIR glm. You need to specify all the parameters listed below in order to run the pipeline. The important output files are 1) evoked timeseries after regressing out confounds, which can be found sub-xx/before_FIR folder, 2) residual timeseries after regressing out stimulus evoked activities, which can be found sub-xx/FIR_residual folder, and 3) FIR regressor beta values, which can be found sub-xx/FIR_betas."
        
        ,"PARAMETERS":{

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
        
def _generate_python_command(cfg_dir): 
    
    """
    This function takes in the configuration file and generate the corresponding python command 
    for post-fMRIprep preprocessing pipeline

    Parameters
    -----------
    cfg_dir: 
        The directory of the configuration file 
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
    
    # task_id and runRest_tr are lists, reformat them for feeding into argparse. 
    task_id=""
    for task in cfg.task_id: 
        task_id += task
        task_id += " "
    rest_tr=""
    for tr in eval(cfg.runRest_tr): 
        rest_tr += str(tr) 
        rest_tr += " "
    
    # generate and return command 
    command = f"python3 {script_path} --sub-id {cfg.sub_id} --task-id {task_id} --space {cfg.space} --base-dir {cfg.base_dir} --output-dir {cfg.output_dir} --designMat-dir {cfg.designMat_dir} --run-restTR {rest_tr} --fwhm {cfg.fwhm} --hpcutoff {cfg.hpcutoff} --nproc {cfg.nproc}"
    
    return command


def run_postfMRIprep_pipeline(cfg_dir): 

    """
    This function takes in the configuration file and generate the corresponding python command for post-fMRIprep preprocessing pipeline.
    Then run that command to submit a python job. 
    
    Parameters
    -----------
    cfg_dir: 
        The directory of the configuration file  
    """
    # get command 
    command = _generate_python_command(cfg_dir)
    
    # run the python script 
    try:
        subprocess.run(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")

def submit_postfMRIprep_pipeline_SLURM(cfg_dir, shell_dir, account, partition, jobname, memory, time="1-00:00:00", log="%x_%A_%a.log", env="jupyterlab-tf-pyt-20211020"):
    
    """
    This function first write out a shell script, then submit the python command to SLURM
    
    Parameters 
    -----------
    cfg_dir: 
        The directory of the configuration file 
    shell_dir: 
        Where to write the shell script, including script's name 
    account: 
        The lab account (e.g., hulacon) 
    partition: 
        The node partition (e.g., long, short, fat) 
    memory: 
        The amount of memory (e.g., 100GB)
    """
    # get command 
    command = _generate_python_command(cfg_dir)
    
    # write shell scritp 
    bash_script_content = \
f'''#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}  
#SBATCH --job-name={jobname}  
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --output={log}

module load fsl
module load ants
module load miniconda
module load singularity
conda activate {env}

{command}
'''

    # Write the Bash script content to a file
    with open(shell_dir, 'w') as file:
        file.write(bash_script_content)

    # Submit the Bash script using sbatch
    sbatch_command = ['sbatch', shell_dir]
    result = subprocess.run(sbatch_command, capture_output=True, text=True)

    # Check the result
    if result.returncode == 0:
        print("Bash script submitted successfully.")
        print("Job ID:", result.stdout.strip())
    else:
        print("Error submitting Bash script.")
        print("Error message:", result.stderr)


