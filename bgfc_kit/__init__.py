#from .structDict import recurseCreateStructDict
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
    unpack_conf,
    load_sub_data,
    detect_bad_frame,
    parcellate_rmMotion_batch,
    define_epoch,
    separate_epochs,
    separate_epochs_per_condition,
    compute_sub_cond_connectome_ztrans_nobadframe,
    compute_epoch_cond_connectome_ztrans_nobadframe,
    vectorize_connectome,
    plot_parcel_FIR_estimates,
    construct_graphs,
    compute_threshold,
    construct_threshold_binary_graphs,
    participation_coefficient,
    compute_network_pc
)