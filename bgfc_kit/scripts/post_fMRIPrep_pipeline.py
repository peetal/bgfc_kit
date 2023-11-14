import pandas as pd
import numpy as np
import os, glob, argparse
from subprocess import call
from nipype.interfaces.base import Bunch
from nipype.pipeline import engine as pe
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.algorithms.modelgen import SpecifyModel
import niflow.nipype1.workflows.fmri.fsl as fsl_wf

# -----------------
# Basic information
# -----------------
parser = argparse.ArgumentParser(description='Post-fmriprep pipeline for DIVATTEN')
parser.add_argument(
    '--sub-id',
    required=True,
    help='subject id, prefix do not include sub- , this would be placed at %s following sub- :\
          sub-%s_task-%s_space-%s_desc-preproc_bold.nii.gz')
parser.add_argument(
    '--task-id',
    required=True,
    help="A list of tasks, this be placed at %s following task- :\
          sub-%s_task-%s_space-%s_desc-preproc_bold.nii.gz\
          IMPORTANT: make sure the order you provide is consistent with the design matrix")
parser.add_argument(
    '--space',
    required=True,
    help="This be placed at %s following space- :\
          sub-%s_task-%s_space-%s_desc-preproc_bold.nii.gz\
          e.g., MNI152NLin2009cAsym_res-2") 
parser.add_argument(
    '--base-dir',
    required=True,
    help="where fMRIPrep derivative folder is at")
parser.add_argument(
    '--output-dir',
    required=True,
    help="output directory")
parser.add_argument(
    '--designMat-dir',
    required=True,
    help="directory for the FIR design matrix")
parser.add_argument(
    '--run-restTR',
    required=True,
    help="resting TRs in a run")
parser.add_argument(
    '--fwhm',
    required=False,
    default=5,
    help="smoothing kernel size")
parser.add_argument(
    '--hpcutoff',
    required=False,
    default=50,
    help="high pass filter cut off, by setting default value being 50, high pass filter is 100")
parser.add_argument(
    '--nproc',
    required=False,
    default=16,
    help="for multithreading")

# readin all the parameters
args = parser.parse_args()

subj_id = args.sub_id
task_id = args.task_id
space = args.space
resting_tr = args.run_restTR

# directory information
output_dir = args.output_dir
log_dir = args.output_dir
fir_design_matrix = args.designMat_dir
base_dir = args.base_dir

# importnat parameters
fwhm = args.fwhm
hpcutoff = args.hpcutoff
n_proc = args.nproc

call("module load fsl", shell = True)
call("module load ants", shell = True)
call("module load singularity", shell = True)

# ----------------
# Helper functions
# ----------------
def _perfix_id(subj_id):
    
    return f'sub-{subj_id}'

def _extract_regressor(full_confound_file):
    
    import pandas as pd
    import os
    """
    Input is the confound files for each functional run (this should be a string), thus this is an iterfield 
    The function would read in the tsv, select wm, csf, and the 6 motion parameters,
    and output a csv file containing only the selected confounds. 
    """
    selected_nuisance = ['csf','white_matter','trans_x','trans_y','trans_z','rot_x','rot_y','rot_z']
    full_confound = pd.read_csv(full_confound_file,sep = '\t')
    selected_regressor = full_confound[selected_nuisance]
    
    output_name = full_confound_file.split('/')[-1].split('.')[0]
    full_output_name = os.path.join(os.getcwd(), f"{output_name}_selected.csv")
    selected_regressor.to_csv(full_output_name, index = False)    
    
    return full_output_name

def _creat_subject_info(regressors_file):
    from nipype.interfaces.base import Bunch
    import pandas as pd
    import numpy as np
    
    confounds = pd.read_csv(regressors_file)
    conf_name = confounds.columns
    conf_value = []
    for con in conf_name:
        value = confounds[f'{con}'].tolist()
        conf_value.append(value)
    
    subject_info = Bunch(
            conditions=['trash_ev'],
            onsets=[[0]],
            durations=[[0]],
            amplitudes=[[0]],
            regressor_names=conf_name,
            regressors=conf_value,
            tmod=None,
            pmod=None)
    
    return subject_info

# scale up the whole image intensity value by making the median = 10000
# This is to fix the potential problem of FEAT when dealing with close to 0 value 
def _get_inormscale(median_values):
    if isinstance(median_values, float):
        op_string = '-mul {:.10f}'.format(10000 / median_values)
    elif isinstance(median_values, list):
        op_string = ['-mul {:.10f}'.format(10000 / val) for val in median_values]
    return op_string

# Zscore each seperate image using the mean and sd of the resting period. 
def _zscore_withRestingPeriod(func, mask, subj_id, task_id):
    
    from nilearn import image, masking
    import pandas as pd
    import numpy as np
    import os
    
    np.seterr(divide='ignore', invalid='ignore')
    
    # get the volume index for resting period and task period
    resting = resting_tr
    
    # load func nifti 
    img = image.load_img(func)
    total_tr = img.get_fdata().shape[-1]
    # load the corresponding mask, with 0 and 1. 
    mask = image.load_img(mask)
    
    # mask the brain
    img_brain = masking.apply_mask(img,mask)
    # get resting volumes 
    img_brain_resting = img_brain[resting,:]
    
    # sd of the resting volumes
    img_brain_resting_sd = np.std(img_brain_resting, axis = 0, ddof = 1)
    
    # check dimension
    if img_brain_resting_sd.shape[0] == np.sum(mask.get_data()[:,:,:]): 
        # check if there is any voxel that have SD = 0 across all resting volumes
        if (img_brain_resting_sd == 0).any():
            index_tup = np.where(img_brain_resting_sd == 0)
            for i in range(len(index_tup)):
                img_brain_resting_sd[index_tup[i][0]] = np.mean(img_brain_resting_sd)
        
        # compute the mean and sd image for the resting period in te 4D structure
        img_resting = img.get_data()[:,:,:,resting]
        img_resting_mean = np.mean(img_resting, axis = 3)
        img_resting_sd = np.std(img_resting, axis = 3)

        # convert the one volume mean image into a 4d image
        resting_mean_4d_unstack = [img_resting_mean for _ in range(total_tr)]
        resting_mean_4d_stack = np.stack(resting_mean_4d_unstack, axis = 3)
        # convert the one volume sd image into a 4d image
        resting_sd_4d_unstack = [img_resting_sd for _ in range(total_tr)]
        resting_sd_4d_stack = np.stack(resting_sd_4d_unstack, axis = 3)

        # compute the z_score image (using resting period mean and sd)
        img_resting_centered = (img.get_data() - resting_mean_4d_stack)/resting_sd_4d_stack 

        # times the z_scored image with the mask, so non-brain voxels become 0.
        # first make the mask into 4d structure
        mask_4d_unstack = [mask.get_data() for _ in range(total_tr)]
        mask_4d_stack = np.stack(mask_4d_unstack, axis = 3)
        # then element-wise multiply mask and time-series arrays
        zscore_brain_array = np.multiply(img_resting_centered, mask_4d_stack)
        # change all NAs to 0
        zscore_brain_array[np.isnan(zscore_brain_array)] = 0

        # write the z score array into a nifti file with the same header 
        img2 = image.new_img_like(img,zscore_brain_array, copy_header = True) 

        img2.to_filename(os.path.join(os.getcwd(),f'sub-{subj_id}_{task_id}_nuisanceRES_Zscore.nii.gz'))

        res_file_demean = os.path.join(os.getcwd(),f'sub-{subj_id}_{task_id}_nuisanceRES_Zscore.nii.gz')

    else: 
        res_file_demean = False
        
    return(res_file_demean)

# merge the 12 runs of a person after zscoring each image seperately, rename the resulted file
def _output_merge_filename(in_file, subj_id):
    
    import os
    from shutil import copyfile 
    
    out_fn = os.path.join(os.getcwd(),
                         (f'sub-{subj_id}_nuisanceRES_CONCAT.nii.gz'))
    
    copyfile(in_file, out_fn)
    return out_fn

# create a shared mask for 6 runs. Only the shared brain voxels = 1, others = 0
def _create_shared_mask(mask_list, subj_id):
    
    import os
    import numpy as np
    from nilearn import image
    
    # create a shared mask template with a mask dimension and all 1s
    shared_mask = np.ones(image.load_img(mask_list[0]).get_fdata().shape)
    # mask_list is a list of 12 brain masks for 12 runs from fmriprep
    for mask in mask_list: # mask = masks_list[0]
        # read in a single brain mask
        mask_img = image.load_img(mask)
        mask_array = mask_img.get_fdata()
        # multiply the shared mask template with the current brain mask, make non-brain voxels 0
        shared_mask = np.multiply(shared_mask, mask_array)
    
    # write the array to the niming like object
    shared_mask_img = image.new_img_like(mask_img,shared_mask) 
    # write out the shared brainmask
    shared_mask_img.to_filename(os.path.join(os.getcwd(),f'sub-{subj_id}_task-shared_brain-mask.nii.gz'))
    # output the path pointing out to the shared brainmask
    shared_mask_img_path = os.path.join(os.getcwd(),f'sub-{subj_id}_task-shared_brain-mask.nii.gz')
    
    return shared_mask_img_path


# for FIR glm output beta naming
def _beta_naming(subj_id):
     
    import os
    return os.path.join(os.getcwd(), (f'sub-{subj_id}_beta216.nii.gz'))

# for FIR glm output residual naming
def _res_naming(subj_id):
    
    import os
    return os.path.join(os.getcwd(), (f'sub-{subj_id}_res3984.nii.gz'))


# ----------------------
# Preprocess pipeline
# ----------------------

if __name__ == "__main__":

    # --------------------------------
    # Start NIPYPE pipelines
    # --------------------------------
    
    # start a wf for each subject
    if not os.path.isdir(os.path.join(output_dir,'working',subj_id)):
        os.makedirs(os.path.join(output_dir,'working',subj_id))
    wf = pe.Workflow(name = 'datainput')
    wf.base_dir = os.path.join(output_dir,'working',subj_id)

    # Feed the dynamic parameters into the node
    inputnode = pe.Node(
        interface = niu.IdentityInterface(
        fields = ['subj_id','task_id','space']), name = 'inputnode')
    # specified above
    inputnode.inputs.subj_id = subj_id
    inputnode.inputs.task_id = task_id
    inputnode.inputs.space = space

    # grab input data
    datasource = pe.MapNode(
        interface = nio.DataGrabber(
            infields = ['subj_id','task_id','space'],
            outfields = ['func','regressor','brainmask',"fir_design_matrix"]),
            iterfield = ['task_id'],
            name = 'datasource')
    # Location of the dataset folder
    datasource.inputs.base_directory = base_dir  
    # Necessary default parameters 
    datasource.inputs.template = '*' 
    datasource.inputs.sort_filelist = True
    # the string template to match
    # field_template and template_args are both dictionaries whose keys correspond to the outfields keyword
    datasource.inputs.template_args = dict(
        func=[['subj_id', 'subj_id', 'task_id','space']],
        regressor =[['subj_id', 'subj_id', 'task_id']],
        brainmask = [['subj_id', 'subj_id', 'task_id','space']])
    datasource.inputs.field_template = dict(
        func = 'derivative/sub-%s/func/sub-%s_task-%s_space-%s_desc-preproc_bold.nii.gz',
        regressor = 'derivative/sub-%s/func/sub-%s_task-%s_desc-confounds_timeseries.tsv',
        brainmask = 'derivative/sub-%s/func/sub-%s_task-%s_space-%s_desc-brain_mask.nii.gz')


    # Datasink for future use
    datasink = pe.Node(interface = nio.DataSink(), name = 'datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.parameterization = False


    # connect the input node with datagrabber, so the datagrabber will use the dynamic parameters to grab func and brain mask for each run
    wf.connect([(inputnode, datasource, [('subj_id', 'subj_id'),
                                         ('task_id', 'task_id'),
                                         ('space', 'space')]),
                (inputnode, datasink, [(('subj_id', _perfix_id), 'container')])
    ])

    # ------------------------------------
    # Smoothing and high-pass filtering
    # ------------------------------------

    # Use create_susan_smooth, a workflow from Niflow for smoothing
    smooth = fsl_wf.create_susan_smooth()
    smooth.inputs.inputnode.fwhm = fwhm
    select_smoothed = pe.MapNode(interface=niu.Select(), 
                                 iterfield=['inlist'],
                                 name='SelectSmoothed')
    select_smoothed.inputs.index = 0 # output a list of lots of files, only choose smoothing related file 

    # connect to the workflow
    wf.connect([(datasource, smooth, [('func','inputnode.in_files')]),
                (datasource, smooth, [('brainmask', 'inputnode.mask_file')]),
                (smooth, select_smoothed, [('outputnode.smoothed_files', 'inlist')])
                #(select_smoothed, datasink, [('out','smoothed')])
               ])


    # temporal mean and high pass node
    # fsl highpass will demean, here create a mean image for latter added back
    temporal_mean = pe.MapNode(interface = fsl.MeanImage(),
                               iterfield=['in_file'],
                               name = 'temporal_mean') 

    # bandpass-temporal filtering, two inputs: hp sigma and lp sigma, 
    # unit is sigma in volumn not in seconds 
    # set low pass sigma to -1 to skip this filter
    # hp sigma in volumn = FWHM / 2*TR \
    temporal_filter = pe.MapNode(interface = fsl.ImageMaths(suffix = '_tempfilt'),
                                 iterfield=['in_file', 'in_file2'],
                                 name = 'temporal_filter')
    temporal_filter.inputs.op_string = (f'-bptf {hpcutoff} -1 -add')

    # connect to the workflow
    wf.connect([(select_smoothed, temporal_mean,[('out','in_file')]),
                (select_smoothed, temporal_filter, [('out','in_file')]),
                (temporal_mean, temporal_filter, [('out_file','in_file2')])
                #(temporal_filter, datasink, [('out_file', 'tempfilt')])
               ])
    
    # ----------------------------------------------------------------
    # Select confound regressors from fmri output confound files 
    # ----------------------------------------------------------------
    
    select_confound = pe.MapNode(
        interface=niu.Function(input_names=['full_confound_file'],
                               output_names=['selected_confound_file'],
                               function=_extract_regressor),
        iterfield =['full_confound_file'], name='select_confound')
    
    wf.connect([
        (datasource, select_confound,[('regressor', 'full_confound_file')]),
        (select_confound, datasink, [('selected_confound_file','nuisance_regressor')])
    ])
    
    # -----------
    # Pre-FEAT
    # -----------

    # Get median value for each EPI data
    median_values = pe.MapNode(
        interface=fsl.ImageStats(op_string='-k %s -p 50'),
        iterfield = ['in_file','mask_file'],
        name='median_values')

    # Normalize each EPI data intensity to 10000
    intensity_norm = pe.MapNode(
        interface=fsl.ImageMaths(suffix='_intnorm'),
        iterfield = ['in_file','op_string'],
        name='intensity_norm')

    # Mask functional image
    masked_func = pe.MapNode(
        interface=fsl.ApplyMask(),
        iterfield = ['in_file','mask_file'],
        name='skullstrip_func')

    wf.connect([(datasource, median_values,[('brainmask', 'mask_file')]),
                (temporal_filter, median_values, [('out_file', 'in_file')]),
                (temporal_filter, intensity_norm, [('out_file','in_file')]),
                (median_values, intensity_norm, [(('out_stat', _get_inormscale), 'op_string')]),
                (intensity_norm, masked_func, [('out_file', 'in_file')]),
                (datasource, masked_func, [('brainmask', 'mask_file')])
    ])

    # ---------------
    # Set up FEAT
    # ---------------

    # Generate nuisance regressor and 1 trash EV (last TR)
    create_regressor = pe.MapNode(
        interface=niu.Function(
            input_names=['regressors_file'],
            output_names=['subject_info'],
            function=_creat_subject_info),
        iterfield =['regressors_file'], name='create_regressor')

    # feed in the nuisance regressor and EV
    model_spec = pe.MapNode(
        interface=SpecifyModel(),
        iterfield=['subject_info'],
        name='create_model_infos')
    model_spec.inputs.input_units = 'secs'
    model_spec.inputs.time_repetition = 1 # TR
    model_spec.inputs.high_pass_filter_cutoff = 100 # high-pass the design matrix with the same cutoff 

    # Generate fsf and ev files for the nuisance GLM
    design_nuisance = pe.MapNode(
        interface=fsl.Level1Design(),
        iterfield=['session_info'],
        name='create_nuisance_design')
    design_nuisance.inputs.interscan_interval = 1 # TR
    design_nuisance.inputs.bases = {'dgamma': {'derivs': False}} # convolution, ideally should be none, but idk how. For tash ev, should not matter much
    design_nuisance.inputs.model_serial_correlations = True # prewhitening

    # Generate the contrast and mat file for the nuisance GLM
    model_nuisance = pe.MapNode(
        interface=fsl.FEATModel(),
        iterfield=['fsf_file', 'ev_files'],
        name='create_nuisance_model')

    # Estimate nuisance GLM model
    model_estimate = pe.MapNode(
        interface=fsl.FILMGLS(),
        iterfield=['in_file','design_file', 'tcon_file'],
        name='estimate_nuisance_models')
    model_estimate.inputs.smooth_autocorr = True
    model_estimate.inputs.mask_size = 5
    model_estimate.inputs.threshold = 1000

    wf.connect([
        (select_confound, create_regressor, [('selected_confound_file','regressors_file')]),
        (create_regressor, model_spec, [('subject_info','subject_info')]),
        (masked_func, model_spec, [('out_file','functional_runs')]),
        (model_spec, design_nuisance, [('session_info','session_info')]),
        (design_nuisance, model_nuisance, [('ev_files', 'ev_files'),
                                           ('fsf_files', 'fsf_file')]),
        (model_nuisance, model_estimate, [('design_file', 'design_file'),
                                          ('con_file', 'tcon_file')]),
        (masked_func, model_estimate, [('out_file','in_file')])
    ])
    
    # ---------------
    # Post-FEAT
    # ---------------

    # Z score each image based on resting period mean and SD
    # THIS IS THE END OF MAPNODE!
    z_score = pe.MapNode(
    interface=niu.Function(
        input_names=['func','mask','subj_id','task_id'],
        output_names=['res_file_demean'],
        function=_zscore_withRestingPeriod),
        iterfield = ['func','mask','task_id'],
    name='zscore')

    # Concatenate 6 z_scored files
    concatenate = pe.Node(interface = fsl.Merge(), name = 'merge_res')
    concatenate.inputs.dimension = 't'
    #print(concatenate.cmdline)

    # rename the Nuisance_regressed_Zscored_Concatenated file
    rename_merge = pe.Node(
        interface = niu.Function(
            input_names = ['in_file','subj_id'],
            output_names = ['out_file'],
            function = _output_merge_filename),
        name = 'rename_merge')

    wf.connect([
        (model_estimate,z_score,[('residual4d', 'func')]),
        (datasource, z_score, [('brainmask','mask')]),
        (inputnode, z_score, [('subj_id', 'subj_id'),
                             ('task_id', 'task_id')]),
        (z_score, concatenate, [('res_file_demean','in_files')]),
        (z_score, datasink, [('res_file_demean','nuisance_res')]),
        (inputnode, rename_merge, [('subj_id','subj_id')]),
        (concatenate, rename_merge, [('merged_file', 'in_file')]),
        (rename_merge, datasink, [('out_file','before_FIR')])
    ])
    
    # -----------
    # FIR-GLM
    # -----------

    # create a shared brain mask for all 6 runs
    shared_mask = pe.Node(
        interface = niu.Function(
            input_names = ['mask_list','subj_id'],
            output_names = ['shared_mask_img_path'],
            function = _create_shared_mask),
        name = 'shared_mask')

    # fsl.GLM
    fir_glm = pe.Node(
        interface = fsl.GLM(),
        name = 'fir_glm'
    )
    fir_glm.inputs.output_type = 'NIFTI_GZ'
    fir_glm.inputs.out_file = os.path.join(os.getcwd(),'beta_param.nii.gz')
    fir_glm.inputs.out_res_name = os.path.join(os.getcwd(),'res4d.nii.gz')
    fir_glm.inputs.design = fir_design_matrix
    #print(fir_glm.cmdline)

    # beta image output file name
    beta_image = pe.Node(
        interface = niu.Function(
            input_names = ['subj_id'],
            output_names = ['beta_file_name'],
            function = _beta_naming),
        name = 'beta_image'

    )

    # beta image output file name
    residual_image = pe.Node(
        interface = niu.Function(
            input_names = ['subj_id'],
            output_names = ['residual_file_name'],
            function = _res_naming),
        name = 'residual_image'
    )

    wf.connect([
        (datasource, shared_mask, [('brainmask', 'mask_list')]),
        (inputnode, shared_mask, [('subj_id', 'subj_id')]),
        (rename_merge, fir_glm, [('out_file', 'in_file')]),
        (shared_mask, fir_glm, [('shared_mask_img_path','mask')]),
        (shared_mask, datasink, [('shared_mask_img_path','task_shared_mask')]),

        (inputnode, beta_image, [('subj_id', 'subj_id')]),
        (beta_image, fir_glm, [('beta_file_name', 'out_file')]),
        (inputnode, residual_image, [('subj_id','subj_id')]),
        (residual_image, fir_glm, [('residual_file_name', 'out_res_name')]),
        (fir_glm, datasink, [('out_file', 'FIR_betas')]),
        (fir_glm, datasink, [('out_res', 'FIR_residual')])
    ])
    
    # -----------------------
    # Atlas transformation
    # -----------------------
    
#     ants_transform = pe.Node(
#         interface = niu.Function(
#             input_names = ['subj_id', 't1_ref'],
#             output_names = ['schaefer_t1w', 'HOSPA_t1w'],
#             function = _MNIatlas_to_T1w),
#         name = 'ants_transform')
    
#     wf.connect([
#         (inputnode, ants_transform, [('subj_id', 'subj_id')]),
#         (shared_mask, ants_transform, [('shared_mask_img_path', 't1_ref')])
#     ])
    
    # ----------------
    # Run the workflow
    # ----------------

    # Run preproc workflow
    wf.write_graph(graph2use='colored')
    wf.config['logging'] = {'log_to_file': 'true', 'log_directory': output_dir}
    wf.config['execution'] = {
        'stop_on_first_crash': 'true',
        'crashfile_format': 'txt',
        'crashdump_dir': output_dir,
        'job_finished_timeout': '65'
    }
    wf.config['monitoring'] = {'enabled': 'true'}
    wf.run(plugin='MultiProc', plugin_args={'n_procs': n_proc})