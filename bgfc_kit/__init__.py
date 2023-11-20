from .structDict import recurseCreateStructDict
from .fir_design_matrix import (
    generate_FIRdesignMat_template_toml,
    write_vanilla_FIRdesginMat,
    write_personalized_FIRdesginMat
)
from .preprocessing_pipeline import (
    generate_postfMRIprep_pipeline_template_toml,
    run_postfMRIprep_pipeline,
    submit_postfMRIprep_pipeline_SLURM
)
from .bgfc_analyses import (
    load_sub_data,
    load_sub_evoked_data,
    detect_bad_frame,
    separate_epochs,
    separate_epochs_per_condition,
    separate_mvpa_epochs_per_condition,
    compute_sub_cond_connectome_ztrans,
    compute_sub_cond_connectome_ztrans_nobadframe,
    compute_epoch_cond_connectome_ztrans_nobadframe,
    compute_epoch_cond_connectome_ztrans,
    compute_epoch_cond_edgevec_ztrans_nobadframe,
    construct_graphs,
    compute_threshold,
    construct_threshold_binary_graphs,
    participation_coefficient
)