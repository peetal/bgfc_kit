import numpy as np 
import pandas as pd
import os, glob
from nilearn import image
from nilearn.regions import img_to_signals_labels
import networkx as nx
import numpy.ma as ma
from plotnine import *
import random 
from collections import Counter, defaultdict 
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from subprocess import call 

def load_sub_data(base_dir:str, sub:int, subcortical:False) -> np.ndarray:
    
    """
    This function takes in the subject ID and then parcellate residual activity data. 
    
    Parameters
    ----------
    input_data: 
        A string pointing to the 4D timeseries niimg-like object (most likely to be the residual activity data). 
    atlas: 
        A string pointing to the 3D atlas (Schaefer or HO). 
    mask: 
        A string pointing to the 3D subject functional mask. 
    subcortical: 
        If true, then include subcortical parcellation. 
    
    Returns
    --------
    signal: a 2d numpy array, first dimension is Parcel and the second is TR.

    Notes
    -----
    The HOSPA atlas description is here: https://neurovault.org/images/1707/, But not all 21 parcels are worth looking at (e.g., 2 is cerebral cortex, 1 is while matter)
    4: left thalamus (201)
    5: left caudate (202)
    6: left putamen (203)
    7: left pallidum (204)
    9: left hippocampus (205)
    10: left amygdala (206)
    11: left accumbens (207)
    15: right thalamus (208)
    16: right caudate (209)
    17: right putamen (210)
    18: right pallidum (211)
    19: right hippocampus (212)
    20: right amygdala (213)
    21: right accumbens (214)
    """
    
    input_data = os.path.join(base_dir, f"sub-{sub:03d}", "FIR_residual", f"sub-{sub:03d}_res3984.nii.gz")
    label = os.path.join(base_dir, f"sub-{sub:03d}", "transformed_atlas", f"sub-{sub:03d}_schaefer200_T1W.nii.gz")
    label_sub = os.path.join(base_dir, f"sub-{sub:03d}", "transformed_atlas", f"sub-{sub:03d}_HOSPA_T1W.nii.gz")
    mask = os.path.join(base_dir, f"sub-{sub:03d}", "task_shared_mask", f"sub-{sub:03d}_task-shared_brain-mask.nii.gz")

    signal, _ = img_to_signals_labels(input_data, label, mask)
    if subcortical: 
        signal_sub, _ = img_to_signals_labels(input_data, label_sub, mask)
        signal_sub = signal_sub[:, [3,4,5,6,8,9,10,14,15,16,17,18,19,20]]
        signal = np.concatenate((signal, signal_sub), axis = 1)
    
    # make sure to transpose the signal, so that the first dimension is parcel (instead of TR). 
    return signal.T 

def detect_bad_frame(sub, signal, run_prop=5, spike=2): 
    
    """
    This function aims to detect bad runs and bad frames, then remove them from the timeseries. 
    
    Parameters 
    ----------
    sub: 
        Subject id, with out the 'sub-' prefix (e.g., 1, 2, 14).
    signal: 
        The output of load_sub_data, the shape is nparcel, ts (200,3984).
    run_prop: 
        The proportion of frames with FD > 0.5 within each run, exceeding which removes the run (5%).
    spike: 
        The spike cutoff, meaning that if a frame has FD greater than cutoff (2mm), then it would be treated as a spike. 
    
    Return 
    -------
    ts: signal without bad frames, the shape is still 200, 3984, but bad frames are np.nan across all 200 parcels 
    """
    # the order that the 12 runs were being concatenated 
    order = {'task-divPerFacePerTone_run-1':1,
             'task-divPerFacePerTone_run-2':2,
             'task-divPerFaceRetScene_run-1':3,
             'task-divPerFaceRetScene_run-2':4,
             'task-divRetScenePerTone_run-1':5,
             'task-divRetScenePerTone_run-2':6,
             'task-singlePerFace_run-1':7,
             'task-singlePerFace_run-2':8,
             'task-singlePerTone_run-1':9,
             'task-singlePerTone_run-2':10,
             'task-singleRetScene_run-1':11,
             'task-singleRetScene_run-2':12}
    
    # get confound files for the participant, order it as how 12 runs are being concatenated. 
    fmriprep_dir = '/projects/hulacon/peetal/divatten/derivative/'
    fir_design = "/projects/hulacon/peetal/divatten/preprocess/fir_design_matrix.txt"
    sub_dir = os.path.join(fmriprep_dir, f'sub-{sub:03d}/func')
    confound_files = glob.glob(os.path.join(sub_dir, "*_run-*_desc-confounds_timeseries.tsv"))
    confound_files.sort(key=lambda x: order['_'.join(x.split("_")[1:3])])
    
    # len(ts)=3984, if 0, then remove the corresponding column in the FIR matrix. 
    ts = []
    for f in confound_files: 
        df = pd.read_csv(f,sep = '\t')
        motion = list(df['framewise_displacement']) # get FD 
        outlier_prop = round(np.sum(np.array(motion)>0.5)/len(motion)*100,2) # %of frames that have FD>0.5
        
        if outlier_prop > run_prop: # remove the whole run (all frames are bad frames) 
            ts_run = [0 for _ in range(len(motion))]
        else:   
        # otherwise check spike, but at the beginning, all timepoints are good frames. 
            ts_run = [1 for _ in range(len(motion))]
            for idx, m in enumerate(motion): 
                if np.isnan(m) or m <= spike: pass # if the frist frame or not a spike 
                if m > spike: # if spike 
                    ts_run[idx-1], ts_run[idx] = 0, 0 # remove the previous and the current frame 
                    if idx + 1 < len(motion): # if not the last frame, remove the next frame 
                        ts_run[idx+1] = 0 
        ts += ts_run 
    
    # check length 
    if len(ts) != signal.shape[1]: 
        print("Timeseries length does not match")
        return 
    
    for idx, frame in enumerate(ts): # remove bad frames from timeseries 
        if frame == 1: 
            pass
        else: 
            signal[:, idx] = np.nan
    
    return signal
    

def separate_epochs(activity_data, epoch_list):
    """ 
    create data epoch by epoch;
    Separate data into conditions of interest specified in epoch_list

    Parameters
    ----------
    activity_data: 
        List of 2D array in shape [nVoxels, nTRs]
        The masked activity data organized in voxel*TR formats of all subjects.
    epoch_list: 
        List of 3D array in shape [condition, nEpochs, nTRs]
        Specification of epochs and conditions, assuming all subjects have the same number of epoch. len(epoch_list) equals the number of subjects.

    Returns
    -------
    raw_data: 
        List of 2D array in shape [nParcels, timepoints (36*16)]
        The data organized in epochs. len(raw_data) equals n subjects * 6 conditions for each subject)
    labels:
        List of 1D array, which is the condition labels of the epochs
        len(labels) labels equals the number of epochs 
    """
    raw_data = []
    labels = []
    for sid in range(len(epoch_list)): #for a given subject
        epoch = epoch_list[sid] # get their npy array
        for cond in range(epoch.shape[0]): # for a given condition
            # for each condition
            sub_epoch = epoch[cond, :, :]
            ts = np.zeros(3984)
            for eid in range(epoch.shape[1]):
                r = np.sum(sub_epoch[eid, :])
                if r > 0:   # there is an epoch in this condition
                    # collapse all epoch of a condition to a single time series
                    ts[sub_epoch[eid, :] == 1] = 1
            ts = ts.astype(bool)
            mat_cond = activity_data[sid][:,ts]
            mat_cond = np.ascontiguousarray(mat_cond)
            raw_data.append(mat_cond)
            labels.append(cond)

    return raw_data, labels

def _divide_into_epoch(ts, epoch_length): 
    
    """
    This function divide full timeseries into same length epochs
    
    Parameters
    ----------
    ts: a 2d array, first dimension is number of parcels and th second dimension is TR/epoch * epoch num
    epoch_length: TR/epoch 
    
    Return
    ------
    a generater for TRs within each epoch. 
    
    """
    ts_len = ts.shape[1]
    for i in range(0, ts_len, epoch_length):  
        yield ts[:,i:i + epoch_length] 
        
def separate_epochs_per_condition(raw_data, labels, condition_label, sub_num): 
    
    """
    This function goes from res3984 into epochs of each condition. 
    Automatically assume that each epoch has 36 TRs (36 is hard-coded)
    
    Parameter
    ----------
    raw_data: 
        Output from function `separate_epochs`
    labels: 
        Second otput from function `separate_epochs`
    condition_label: 
        This is arbitrarily defined generating epoch files: 
    sub_num: 
        The number of subject
    
    Return
    ------
    cond_epoch_ts: 
        a list of 3d array. 
        The length of the list is the number of subject. Each array is of the shape (16 epochs of the condition, 200 parcels, 36 TR)
    
    """
    cond_raw_data = [raw_data[cond] for cond in np.where(np.array(labels) == condition_label)[0].tolist()] 
    cond_epoch_ts = []
    for sub in range(sub_num):
        cond_epoch_ts.append(np.stack(list(_divide_into_epoch(cond_raw_data[sub], 36)), axis = 0))
    return cond_epoch_ts


def separate_mvpa_epochs_per_condition(raw_data, labels, condition_label, sub_num): 
    
    """
    This function goes from res3984 into epochs of each condition for activity-based analyese. 
    Each epoch includes 24 task TRs, shifted 3 TR accounting for hemodynamic delay. 
    The only difference between this function and the function above is 24 vs. 36, I make them 
    into 2 funtions to avoid having too many parameters to confuse myself in the future. 
    
    Parameters
    ----------
    raw_data: 
        Output from function `separate_epochs`
    labels: 
        Second otput from function `separate_epochs`
    condition_label: 
        This is arbitrarily defined generating epoch files: 
    sub_num: 
        The number of subject
    
    Return
    ------
    cond_epoch_ts: 
        a list of 3d array. 
        The length of the list is the number of subject. Each array is of the shape (16 epochs of the condition, 200 parcels, 36 TR)
    """
    
    cond_raw_data = [raw_data[cond] for cond in np.where(np.array(labels) == condition_label)[0].tolist()] 
    cond_epoch_ts = []
    for sub in range(sub_num):
        cond_epoch_ts.append(np.stack(list(_divide_into_epoch(cond_raw_data[sub], 24)), axis = 0))
    return cond_epoch_ts

def compute_sub_cond_connectome_ztrans(epoch_data:np.ndarray):
    
    """
    This function computes the connectome for each epoch, 
    and then AVERAGES across all peochs of the same condition 
    
    Parameters
    -----------
    epoch_data: A list of 2d array. The length equals the number of participant. 
                Each array is of the shape 16epoch, 200parcels, 36TR/epoch
                

    Yields
    ------
    sub_cond_connectome_ztrans: 
        A generator of 3d arrays. 200 by 200 corMat (averaged across all epochs). generator length is nsub
    """
    
    # basic information
    sub_num = len(epoch_data) # the number of partcipants
    epoch_num = epoch_data[0].shape[0] # the number of epochs 
    
    for sub in range(sub_num): 
        # grad the subject's data
        sub_epoch = epoch_data[sub]
        
        # compute connectome for each epoch [16, 200, 200]
        sub_epoch_connectome = [np.corrcoef(sub_epoch[e, :, :]) for e in range(epoch_num)]
        
        # averave CorMat across all epochs 
        sub_cond_connectome = np.mean(sub_epoch_connectome, axis = 0)
        np.fill_diagonal(sub_cond_connectome, 0) # fill the diagnal with 0 for graph construction later
        
        sub_cond_connectome_ztrans = np.arctanh(sub_cond_connectome)
        
        yield sub_cond_connectome_ztrans
        
def compute_sub_cond_connectome_ztrans_nobadframe(epoch_data:np.ndarray):
    
    """
    This function computes the connectome for each epoch, 
    and then AVERAGES across all peochs of the same condition. 
    Important: 
    1) If all frames within this epoch is np.nan, drop the epoch
    2) If some frames within this epoch is np.nan, drop the frame then compute the correlation matrix 
    Thus, per subject, per condition, not always 16 epochs! 
    
    Parameters: 
    -----------
    epoch_data: A list of 2d array. The length equals the number of participant. 
                Each array is of the shape 16epoch, 200parcels, 36TR/epoch

    Returns: 
    --------
    A generator of 2d arrays. 200 by 200 corMat (averaged across all epochs). generator length is nsub
    """
    
    # basic information
    sub_num = len(epoch_data) # the number of partcipants
    epoch_num = epoch_data[0].shape[0] # the number of epochs 
    
    for sub in range(sub_num): 

        # grab the subject's data, shape is [16,200,36]
        sub_epoch = epoch_data[sub]
        sub_epoch_connectome = []
        for cur_epoch in range(epoch_num): 
            if np.isnan(sub_epoch[cur_epoch,:,:]).all(): # if this epoch is full of na, meaning this epoch (run) has been dropped.
                pass 
            else: 
                # compute connectome for each nonempty epoch [8/16, 200, 200]
                sub_epoch_connectome.append(ma.corrcoef(ma.masked_invalid(sub_epoch[cur_epoch,:,:])).filled())

        #print(len(sub_epoch_connectome))

        # averave CorMat across all epochs 
        sub_cond_connectome = np.mean(sub_epoch_connectome, axis = 0)
        np.fill_diagonal(sub_cond_connectome, 0) # fill the diagnal with 0 for graph construction later

        sub_cond_connectome_ztrans = np.arctanh(sub_cond_connectome)

        yield sub_cond_connectome_ztrans
            
def compute_epoch_cond_connectome_ztrans_nobadframe(epoch_data:np.ndarray):
    
    """
    This function computes the connectome for each epoch for each condition
    
    Important: 
    1) If all frames within this epoch is np.nan, drop the epoch
    2) If some frames within this epoch is np.nan, drop the frame then compute the correlation matrix 
    Thus, per subject, per condition, not always 16 epochs! 
    
    Parameters: 
    -----------
    epoch_data: A list of 2d array. The length equals the number of participant. 
                Each array is of the shape 16epoch, 200parcels, 36TR/epoch
                
    Returns: 
    --------
    A generator of 2d arrays. 200 by 200 corMat (averaged across all epochs). generator length is the total number of epochs across all subjects. 
    """
    
    # basic information
    sub_num = len(epoch_data) # the number of partcipants
    epoch_num = epoch_data[0].shape[0] # the number of epochs 
    
    for sub in range(sub_num): 
        
        # grab the subject's data, shape is [16,200,36]
        sub_epoch = epoch_data[sub]
        sub_epoch_connectome = []
        for cur_epoch in range(epoch_num): 
            if np.isnan(sub_epoch[cur_epoch,:,:]).all(): # if this epoch is full of na, meaning this epoch (run) has been dropped.
                pass 
            else: 
                # compute connectome for each nonempty epoch [8/16, 200, 200]
                sub_epoch_connectome = ma.corrcoef(ma.masked_invalid(sub_epoch[cur_epoch,:,:])).filled()
                np.fill_diagonal(sub_epoch_connectome, 0)
                sub_epoch_connectome_ztrans = np.arctanh(sub_epoch_connectome)
                yield sub_epoch_connectome_ztrans

        
def compute_epoch_cond_connectome_ztrans(epoch_data:np.ndarray):
    
    """
    This function computes the connectome for each epoch, then vectorize each epoch 
    return a list of vectorized connectome
    
    Parameters: 
    -----------
    epoch_data: A list of 2d array. The length equals the number of participant. 
                Each array is of the shape 16epoch, 200parcels, 36TR/epoch
                

    Returns: 
    --------
    A list (len = subject) of list (len = epoch = 16) of vectorized connectom 
    """
    
    # basic information
    sub_num = len(epoch_data) # the number of partcipants
    epoch_num = epoch_data[0].shape[0] # the number of epochs 
    
    sub_epoch_vector = []
    for sub in range(sub_num): 
        # grad the subject's data
        sub_epoch = epoch_data[sub]
        
        # compute connectome for each epoch [16, 200, 200]
        sub_epoch_connectome = [np.corrcoef(sub_epoch[e, :, :]) for e in range(epoch_num)]
        sub_epoch_connectome_z = [np.arctanh(c) for c in sub_epoch_connectome]
        sub_epoch_connectome_vector = [list(c[np.triu_indices(200, k = 1)]) for c in sub_epoch_connectome_z]
        
        sub_epoch_vector.append(sub_epoch_connectome_vector)
    
    return sub_epoch_vector

def compute_epoch_cond_edgevec_ztrans_nobadframe(epoch_data:np.ndarray):
    
    """
    This function computes the connectome for each epoch, then vectorize each epoch 
    return a list of vectorized connectome
    
    Parameters: 
    -----------
    epoch_data: A list of 2d array. The length equals the number of participant. 
                Each array is of the shape 16epoch, 200parcels, 36TR/epoch
                

    Returns: 
    --------
    A list (len = subject) of list (len = epoch = 16 or 8) of vectorized connectom(19800)
    """
    
    # basic information
    sub_num = len(epoch_data) # the number of partcipants
    epoch_num = epoch_data[0].shape[0] # the number of epochs 
    
    sub_epoch_vector = []
    for sub in range(sub_num): 
        
        # grad the subject's data, shape is [16,200,36]
        sub_epoch = epoch_data[sub]
        sub_epoch_connectome = []
        for cur_epoch in range(epoch_num): 
            if np.isnan(sub_epoch[cur_epoch,:,:]).all(): # if this epoch is full of na, meaning this epoch (run) has been dropped.
                pass 
            else: 
                # compute connectome for each nonempty epoch 8 or 16 200x200 array (or 214x214). 
                sub_epoch_connectome.append(ma.corrcoef(ma.masked_invalid(sub_epoch[cur_epoch,:,:])).filled())

        # z_transform connectome for each epoch 8 or 16 200x200 array (or 214x214)
        sub_epoch_connectome_z = [np.arctanh(c) for c in sub_epoch_connectome]
        # vectorize connections: list (8 or 16 epoch) of 19900 vector
        nparcel = sub_epoch_connectome_z[0].shape[0]
        sub_epoch_connectome_vector = [list(c[np.triu_indices(nparcel, k = 1)]) for c in sub_epoch_connectome_z] 

        sub_epoch_vector.append(sub_epoch_connectome_vector)
    
    return sub_epoch_vector 
        
        
def construct_graphs(corMats, threshold=0):
    
    """
    This function construct a unthresholded, weighted, graph for each connectome
    
    Parameters: 
    -----------
    corMat_list: A list of connectome (i.e., correlation matrix)
    threshold: whether to include the edge, default is 0
    """
    
    def _do_single_graph(corMat, threshold):
        
        # Preset the graph
        G = nx.Graph()

        # Create the edge list
        nodelist = []
        edgelist = []
        
        for row_counter in range(corMat.shape[0]):
            nodelist.append(str(row_counter))  # Set up the node names (zero indexed)
            for col_counter in range(corMat.shape[1]):

                # Determine whether to include the edge based on whether it exceeds the threshold
                if abs(corMat[row_counter, col_counter]) > abs(threshold):
                    # Add a tuple specifying the voxel pairs being compared and the weight of the edge
                    edgelist.append((str(row_counter), str(col_counter), {'coupling_strength': corMat[row_counter, col_counter]}))
                    #edgelist.append((str(row_counter), str(col_counter), {'weight': 1}))

        # Create the nodes in the graph
        G.add_nodes_from(nodelist)

        # Add the edges
        G.add_edges_from(edgelist)
        
        nx.set_edge_attributes(G, {e: 1/d["coupling_strength"] for e, d in G.edges.items()}, "distance")
    
        return(G)
        
    if type(corMats) == list: # if input is a list of corMat, then return a list of graph 
        graph_list = [_do_single_graph(corMat, threshold) for corMat in corMats]
    else: # otherwise, just return a single graph. 
        graph_list = _do_single_graph(corMats, threshold)

    return graph_list

def compute_threshold(corMat, density):
    """
    This function computes threshold given edge density (i.e., the percentage of edges to keep). 
    
    Parameters: 
    -----------
    corMat: one correlation matrix 
    density: the percentage of edges to keep (e.g., 15 means to keep the top 15%)
    """
    
    def _upper_tri_indexing(A):
        m = A.shape[0]
        r,c = np.triu_indices(m,1)
        return A[r,c]
    threshold = np.percentile(_upper_tri_indexing(corMat),100-density)
    return threshold
    

def construct_threshold_binary_graphs(corMats, density):
    
    """
    This function construct thresholded, binary, graph for each connectome
    
    Parameters: 
    -----------
    corMat_list: A list of connectome (i.e., correlation matrix)
    density: the percentage of edges to keep (e.g., 15 means to keep the top 15%)
    """
    def _do_single_graph(corMat, density):
        
        threshold = compute_threshold(corMat, density)
        
        # Preset the graph
        G = nx.Graph()

        # Create the edge list
        nodelist = []
        edgelist = []
        
        for row_counter in range(corMat.shape[0]):
            nodelist.append(str(row_counter))  # Set up the node names (zero indexed)
            for col_counter in range(corMat.shape[1]):

                # Determine whether to include the edge based on whether it exceeds the threshold -> constructing binary graph
                if corMat[row_counter, col_counter] >= threshold:
                    edgelist.append((str(row_counter), str(col_counter), {'weight': 1}))

        # Create the nodes in the graph
        G.add_nodes_from(nodelist)

        # Add the edges
        G.add_edges_from(edgelist)
        
        return(G)
        
    if type(corMats) == list: # if input is a list of corMat, then return a list of graph 
        graph_list = [_do_single_graph(corMat, density) for corMat in corMats]
    else: # otherwise, just return a single graph. 
        graph_list = _do_single_graph(corMats, density)

    return graph_list


def participation_coefficient(G, module_partition):
    '''
    Computes the participation coefficient of nodes of G with partition
    defined by module_partition.
    (Guimera et al. 2005).

    Parameters
    ----------
    G : :class:`networkx.Graph`
    module_partition : dict
        a dictionary mapping each community name to a list of nodes in G

    Returns
    -------
    dict:
        A dictionary mapping the nodes of G to their participation coefficient
        under the participation specified by module_partition.
    '''
    # Initialise dictionary for the participation coefficients
    pc_dict = {}

    # Loop over modules to calculate participation coefficient for each node
    for m in module_partition.keys():
        # Create module subgraph
        M = set(module_partition[m])
        for v in M:
            # Calculate the degree of v in G
            degree = float(nx.degree(G=G, nbunch=v))

            # Calculate the number of intramodule degree of v
            wm_degree = float(sum([1 for u in M if (u, v) in G.edges()]))
            
            if degree == 0 and wm_degree == 0: 
                pc_dict[v] = 0
            else: 
            # The participation coeficient is 1 - the square of
            # the ratio of the within module degree and the total degree
                pc_dict[v] = 1 - ((float(wm_degree) / float(degree))**2)

    return pc_dict
