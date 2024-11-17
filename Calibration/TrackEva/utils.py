import os
import time

from evo.core import metrics
import pandas as pd
import copy

import matplotlib.pyplot as plt
#%matplotlib inline
#%matplotlib tk

import pprint
import numpy as np

import numpy as np

from evo.tools.file_interface import read_tum_trajectory_file
from evo.core import sync
import evo.core.lie_algebra as lie
from evo.tools import plot
from evo import core
from evo.core.trajectory import PosePath3D, PoseTrajectory3D

from evo.tools import log
log.configure_logging(verbose=False, debug=False, silent=True)



def check_monotionic_increaseing(traj, type="gt"):
    non_monotonic_rows = np.where((traj.timestamps[:-1] < traj.timestamps[1:]) == False)[0]
    total_non_monotonic_rows = 0
    # Remove non-monotonic rows
    while(len(non_monotonic_rows) > 0):
        traj.timestamps = np.delete(traj.timestamps, non_monotonic_rows, axis=0)
        traj._positions_xyz = np.delete(traj._positions_xyz, non_monotonic_rows, axis=0)
        traj._orientations_quat_wxyz = np.delete(traj._orientations_quat_wxyz, non_monotonic_rows, axis=0)

        total_non_monotonic_rows += len(non_monotonic_rows)
        # Check it again
        non_monotonic_rows = np.where((traj.timestamps[:-1] <= traj.timestamps[1:]) == False)[0]

    print(f"{type} - found {total_non_monotonic_rows} non-monotonic increasing rows")
    return traj



def check_gt_abnormal_traj(traj_gt, speed_threshold=6):
    abnormal_step_rows = np.where(traj_gt.speeds >= speed_threshold)[0]
    
    total_abnormal_step_rows = 0
    # Remove abnormal_step_rows
    while(len(abnormal_step_rows) > 0):
        to_delete_rows = np.unique(np.sort(np.concat([abnormal_step_rows, 
                                                    abnormal_step_rows-1, 
                                                    abnormal_step_rows+1])))[1:-1]

        traj_gt.timestamps = np.delete(traj_gt.timestamps, to_delete_rows, axis=0)
        traj_gt._positions_xyz = np.delete(traj_gt._positions_xyz, to_delete_rows, axis=0)
        traj_gt._orientations_quat_wxyz = np.delete(traj_gt._orientations_quat_wxyz, to_delete_rows, axis=0)

        total_abnormal_step_rows += len(abnormal_step_rows)
        # Check it again
        abnormal_step_rows = np.where(traj_gt.speeds >= speed_threshold)[0]

    print(f"gt - found {total_abnormal_step_rows} abnormal_step_rows")
    return traj_gt



# def get_Quest_traj_from_gt(traj_gt):
#     transform_matrix = np.eye(4, dtype=float)
#     # Quest
#     # transform_matrix[0,3] = - 0.11  # x
#     # transform_matrix[1,3] = 0 # - 0.017 # - 0.017  # y
#     # transform_matrix[2,3] = - 0.026  # z
#     transform_matrix[0,3] = -0.02  # x
#     transform_matrix[1,3] = -0.005 # -0.02 # y
#     transform_matrix[2,3] =  0.075  # z
#     traj_gt_xr = copy.deepcopy(traj_gt)
#     traj_gt_xr.transform(t=transform_matrix, right_mul=True, propagate=False)
#     return traj_gt_xr



def get_Quest_traj_from_gt(traj_gt):
    transform_matrix = np.eye(4, dtype=float)
    # Quest
    # transform_matrix = np.array(
    #   [[ 0.99327733, -0.02829756, -0.04932406,  0.017        ],
    #    [ 0.02733661,  0.9942952 , -0.02021545,  -0.061        ],
    #    [ 0.04980459,  0.01901585,  0.99705948,  0.102        ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
    # transform_matrix = np.array(
    #   [[ 0.99327733, -0.02829756, -0.04932406,  0.017     ],
    #    [ 0.02733661,  0.9942952 , -0.02021545, -0.042     ],
    #    [ 0.04980459,  0.01901585,  0.99705948,  0.104     ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
    # transform_matrix = np.array(
    #     [[ 0.99327733, -0.02829756, -0.04932406,  0.018     ],
    #    [ 0.02733661,  0.9942952 , -0.02021545, -0.033     ],
    #    [ 0.04980459,  0.01901585,  0.99705948,  0.106     ],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
    transform_matrix = np.array([[ 0.9875741 , -0.15088097, -0.02580624,  0.01      ],
       [ 0.15086376,  0.98791899, -0.00200797, -0.04      ],
       [ 0.0257956 , -0.00190042,  0.99963033,  0.104     ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
    transform_matrix = np.array([[ 0.96963494, -0.24172874, -0.02327466,  0.017     ],
       [ 0.24158216,  0.96991531, -0.01022584, -0.046     ],
       [ 0.02503697,  0.00431594,  0.99890661,  0.104     ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
    transform_matrix = np.array([[ 0.97928593, -0.19665039, -0.02414906,  0.021     ],
           [ 0.19662194,  0.97957354, -0.00376876, -0.042     ],
           [ 0.02463059, -0.00103345,  0.99968542,  0.104     ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])

    #transform_matrix = np.eye(4)
    traj_gt_xr = copy.deepcopy(traj_gt)
    traj_gt_xr.transform(t=transform_matrix, right_mul=True, propagate=False)
    return traj_gt_xr



def get_AVP_traj_from_gt(traj_gt):
    # transform_matrix = np.eye(4, dtype=float)
    # # AVP2
    # transform_matrix[0,3] = -0.017  # x
    # transform_matrix[1,3] = - 0.017 #- 0.017 # - 0.017  # y
    # transform_matrix[2,3] = 0.045 #+ 0.08 # z

    # transform_matrix[1,1] = 0.9848
    # transform_matrix[2,2] = 0.9848
    # transform_matrix[2,1] = 0.17365
    # transform_matrix[1,2] = -0.17365
    transform_matrix = np.array(
      [[ 0.99182384, -0.11747813, -0.04823677,  0.022     ],
       [ 0.11747405,  0.99300113, -0.00343895, -0.043    ],
       [ 0.04828045, -0.00226101,  0.99869436,  0.1       ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
    transform_matrix = np.array([[ 0.99625924, -0.05535767, -0.0371308 ,  0.022     ],
       [ 0.05575718,  0.99683364,  0.01344899, -0.026     ],
       [ 0.03697038, -0.01615976,  0.99905235,  0.104     ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
    
    traj_gt_xr = copy.deepcopy(traj_gt)
    traj_gt_xr.transform(t=transform_matrix, right_mul=True, propagate=False)
    return traj_gt_xr

# def get_AVP_traj_from_gt(traj_gt):
#     transform_matrix = np.eye(4, dtype=float)
#     # AVP2
#     transform_matrix[0,3] = -0.017  # x
#     transform_matrix[1,3] = - 0.017 #- 0.017 # - 0.017  # y
#     transform_matrix[2,3] = 0.045 #+ 0.08 # z

#     transform_matrix[1,1] = 0.9848
#     transform_matrix[2,2] = 0.9848
#     transform_matrix[2,1] = 0.17365
#     transform_matrix[1,2] = -0.17365
#     traj_gt_xr = copy.deepcopy(traj_gt)
#     traj_gt_xr.transform(t=transform_matrix, right_mul=True, propagate=False)
#     return traj_gt_xr


def check_orb_abnormal_traj(traj_est, traj_ref, speed_threshold=6):
    # 1. Find out regions that is lost tracking
    zero_indices = np.where(traj_est.speeds == 0.0)[0]
    lost_regions = []
    if len(zero_indices) > 0:
        start = zero_indices[0]
        end = zero_indices[0]

        for num in zero_indices[1:]:
            if num == end + 1 :
                end = num
            else:
                if(end - start > 1):
                    lost_regions.append([start,end])
                end = num
                start = num
    # 2. Find out moments that has map merge/relocalization
    shift_checkpoints = np.where(traj_est.speeds>speed_threshold)[0]
    # 3. Merge the first two regions
    checkpoints = []
    checkpoints.append(0)
    checkpoints.append(traj_est.num_poses-1)
    lost_checkpoints = []
    for lost_region in lost_regions:
        lost_checkpoints.append(lost_region[0])
        lost_checkpoints.append(lost_region[1])
    checkpoints = checkpoints + lost_checkpoints
    checkpoints = checkpoints + list(shift_checkpoints)
    checkpoints = list(set(checkpoints))
    checkpoints = sorted(checkpoints)
    # 4. Generate align regions
    align_regions = []
    start = checkpoints[0]
    for idx, checkpoint in enumerate(checkpoints[1:]):
        if(checkpoint - start <= 1):
            ## when there are consecutive checkpoints
            # When start in lost and end in shift, 
            # they need to be in seperate region
            if (start in lost_checkpoints and checkpoint in shift_checkpoints):
                start = checkpoint
            # When start in shift and end in shift, 
            # they need to be in the same region
            if (start in shift_checkpoints and checkpoint in shift_checkpoints):
                pass
            
        else:
            align_regions.append([start, checkpoint])
            start = checkpoint
    align_regions_dict = {}
    align_regions_dict['align_regions'] = align_regions
    align_regions_dict['lost_regions'] = lost_regions
    align_regions_dict['shift_checkpoints'] = shift_checkpoints

    if len(align_regions_dict['align_regions']) > 0:
        # print("Actual subtrajectories for alignment: {}".format(align_regions))
        xyz = traj_est._positions_xyz
        quat = traj_est._orientations_quat_wxyz
        time = traj_est.timestamps
        # split trajectory to subtrajectories
        subtrajectories = []
        for idx, region in enumerate(align_regions_dict['align_regions']):
            xyz_sub = xyz[region[0]:region[1], :]
            xyz_sub[0,:] = xyz_sub[1,:]
            xyz_sub[-1,:] = xyz_sub[-2,:]
            
            quat_sub = quat[region[0]:region[1], :]
            quat_sub[0,:] = quat_sub[1,:]
            quat_sub[-1,:] = quat_sub[-2,:]

            # Perform subtrajectory alignment
            time_sub = time[region[0]:region[1]]
            traj_sub = PoseTrajectory3D(xyz_sub, quat_sub, time_sub)
            #print("Original estimated subtrajectory length: {}".format(traj_sub.num_poses))
            try:
                traj_ref_copy = copy.deepcopy(traj_ref)
                traj_ref_copy, traj_sub = sync.associate_trajectories(traj_ref_copy, traj_sub, max_diff=0.05)
                
                #n = int(traj_sub.timestamps.shape[0]/2)
                #traj_sub.align(traj_ref_copy, correct_scale=True, correct_only_scale=False, n=n)
                traj_sub.align(traj_ref_copy, correct_scale=False, correct_only_scale=False)
                
                subtrajectories.append(traj_sub)
            except Exception as e:
                print("subtrajectory alignment failed: {}".format(e))
                traj_ref_copy = copy.deepcopy(traj_ref)
                traj_ref_sub, traj_sub = sync.associate_trajectories(traj_ref_copy, traj_sub, max_diff=0.05)
                subtrajectories.append(traj_ref_sub)

            #self.plot_trajectory(subtrajectories[-1], traj_ref, trial=idx)

        traj_est_aligned = core.trajectory.merge(subtrajectories)
        traj_ref, traj_est_aligned = sync.associate_trajectories(traj_ref, traj_est_aligned, max_diff=0.05)
        #n = int(traj_est_aligned.timestamps.shape[0]/2)
        #print("aligned length: {}".format(n))
        #traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False, n=n)
    else:
        # The redo find_align_region cannot find any subtrajectories to align
        # then directly synchronize the two trajectories
        traj_ref, traj_est_aligned = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.05)

    return traj_est_aligned, traj_ref



def align_trajectory(traj_ref, traj_est, offset):
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_ref_aligned = copy.deepcopy(traj_ref)
    traj_ref_aligned, traj_est_aligned = sync.associate_trajectories(traj_ref_aligned, traj_est_aligned, 
                                                                  max_diff=0.01, 
                                                                  offset_2=offset)
    traj_est_aligned.align(traj_ref_aligned, correct_scale=False, correct_only_scale=False)#, n=n)
    return traj_ref, traj_est
    
    

def plot_trajectory(traj_est, traj_ref, benchmark=None, trajectory=None, trial=None):
    fig = plt.figure(figsize=[10,10])

    traj_est_aligned = copy.deepcopy(traj_est)
    traj_ref_aligned = copy.deepcopy(traj_ref)
    
    traj_by_label = {
        "estimate": traj_est_aligned,
        "reference": traj_ref_aligned
    }

    plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)


    ax = plt.gca()
    
    traj_est_aligned.downsample(int(0.1*traj_est_aligned.num_poses))
    traj_ref_aligned.downsample(int(0.1*traj_ref_aligned.num_poses))
    
    plot.draw_correspondence_edges(ax=ax, traj_1=traj_est_aligned,
                                    traj_2=traj_ref_aligned,
                                    plot_mode=plot.PlotMode.xyz, style = '-',
                                    color="black", alpha = 0.2)
    
    # plot.draw_coordinate_axes(ax=ax, traj=traj_est_aligned,
    #                      plot_mode=plot.PlotMode.xyz, marker_scale = 0.1,
    #                      x_color="r", y_color="g", z_color="b")
    
    plot.draw_coordinate_axes(ax=ax, traj=traj_ref_aligned,
                         plot_mode=plot.PlotMode.xyz, marker_scale = 0.05,
                         x_color="r", y_color="g", z_color="b") 
    #fig.savefig('./figures/{}-{}-{}-trajectory.png'.format(benchmark, trajectory, trial))



def plot_aligned_trajectory(traj_est, traj_ref, benchmark=None, trajectory=None, trial=None):
    fig = plt.figure(figsize=[10,10])
    
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_ref_copy = copy.deepcopy(traj_ref)
    traj_ref_copy, traj_est_aligned = sync.associate_trajectories(traj_ref_copy, traj_est_aligned, 
                                                                  max_diff=0.01, 
                                                                  offset_2=0)

    #n = int(traj_est_aligned.timestamps.shape[0])#/2)
    #print("aligned length: {}".format(n))
    
    traj_by_label = {
        #"estimate (not aligned)": traj_est,
        "estimate (aligned)": traj_est_aligned,
        "reference": traj_ref_copy
    }
    
    plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
    
    ax = plt.gca()
    
    traj_est_aligned.downsample(int(0.05*traj_est_aligned.num_poses))
    traj_ref_copy.downsample(int(0.05*traj_ref_copy.num_poses))
    
    plot.draw_correspondence_edges(ax=ax, traj_1=traj_est_aligned,
                              traj_2=traj_ref_copy,
                              plot_mode=plot.PlotMode.xyz, style = '-',
                              color="black", alpha = 0.2)
    
    # plot.draw_coordinate_axes(ax=ax, traj=traj_est_aligned,
    #                      plot_mode=plot.PlotMode.xyz, marker_scale = 0.1,
    #                      x_color="r", y_color="g", z_color="b")
    
    plot.draw_coordinate_axes(ax=ax, traj=traj_ref_copy,
                         plot_mode=plot.PlotMode.xyz, marker_scale = 0.05,
                         x_color="r", y_color="g", z_color="b") 
    #fig.savefig('./figures/{}-{}-{}-trajectory.png'.format(benchmark, trajectory, trial))



def calculate_RE(traj_est, traj_gt, est_offset=0):

    traj_est_copy = copy.deepcopy(traj_est)
    traj_est_copy.timestamps = traj_est_copy.timestamps + est_offset
    # form the (reference, estimation) pair
    traj_ref = copy.deepcopy(traj_gt)

    # error metric settings
    pose_relation = metrics.PoseRelation.point_distance
    #pose_relation = metrics.PoseRelation.translation_part
    #pose_relation = metrics.PoseRelation.point_distance_error_ratio
    
    delta_unit = metrics.Unit.meters
    #delta_unit = metrics.Unit.frames

    delta = 0.1

    all_pairs = True

    #traj_est_copy, _ = check_orb_abnormal_traj(traj_est_copy, traj_ref, speed_threshold=3)

    # form the (reference, estimation) pair
    traj_ref = copy.deepcopy(traj_gt)
    
    traj_ref, traj_est_copy = sync.associate_trajectories(traj_ref, traj_est_copy, max_diff=0.05)
    data = (traj_ref, traj_est_copy)
    
    # load error metric setting
    rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta,
                            delta_unit=delta_unit, all_pairs=all_pairs)
    # calculate the error
    rpe_metric.process_data(data)
    # devided by the subjectory length --> invariant to the length
    #error = np.array(rpe_metric.error)
    return rpe_metric



def calculate_APE(traj_est, traj_gt, est_offset=0):

    traj_est_copy = copy.deepcopy(traj_est)
    traj_est_copy.timestamps = traj_est_copy.timestamps + est_offset
    # form the (reference, estimation) pair
    traj_ref = copy.deepcopy(traj_gt)

    # error metric settings
    #pose_relation = metrics.PoseRelation.point_distance
    pose_relation = metrics.PoseRelation.translation_part
    #pose_relation = metrics.PoseRelation.point_distance_error_ratio
    
    # form the (reference, estimation) pair
    traj_ref = copy.deepcopy(traj_gt)
    traj_ref, traj_est_copy = sync.associate_trajectories(traj_ref, traj_est_copy, max_diff=0.05)
    data = (traj_ref, traj_est_copy)
    
    # load error metric setting
    ape_metric = metrics.APE(pose_relation)
    # calculate the error
    ape_metric.process_data(data)
    return ape_metric



# def find_traj_est_offset(traj_est, traj_gt, iter=10, lower=-3, upper=3):
#     # We assume the RPE is a convex function to the offset
#     idx = 0
#     median = (lower+upper)/2
#     old_offset_list = [lower, median, upper]
#     new_offset_list = [lower, median, upper]
#     error_list = [None, None, None]
#     while(idx<iter):
#         for i, offset in enumerate(new_offset_list):
#             traj_est_cp = copy.deepcopy(traj_est)
#             traj_gt_cp = copy.deepcopy(traj_gt)
            
#             rpe_metric = calculate_RE(traj_est_cp, traj_gt_cp, est_offset=offset)
#             error_list[i] = np.mean(np.array(rpe_metric.error))
#         # conditions
#         old_offset_list = new_offset_list
#         if np.argmin(error_list) == 0:
#             #print("Case 1")
#             upper = median
#             lower = lower - 0.5*(median-lower)
#             median = (lower+median)/2
#         elif np.argmax(error_list) == 2:
#             #print("Case 2")
#             upper = median
#             #lower = lower
#             median = (lower+median)/2
#         elif np.argmax(error_list) == 0:
#             #print("Case 3")
#             lower = median
#             median = (from evo.tools.settings import SETTINGS
# # SETTINGS.plot_pose_correspondences = True
# # plot.apply_settings(SETTINGS)median+upper)/2
#             #upper = upper
#         elif np.argmin(error_list) == 2:
#             #print("Case 4")
#             upper = upper + 0.5*(upper-median)
#             lower = median
#             median = (upper+median)/2
#         new_offset_list = [lower, median, upper]
#         print(f"Iteration {idx} - Best offset = {old_offset_list[np.argmin(error_list)]} - checkpoint {old_offset_list}")
#         idx += 1
#     best_offset = old_offset_list[np.argmin(error_list)]
#     return best_offset



def find_traj_est_offset(traj_est, traj_gt, iter=10, lower=-3, upper=3):
    # We assume the RPE is a convex function to the offset
    idx = 0
    median = (lower+upper)/2
    old_offset_list = [lower, median, upper]
    new_offset_list = [lower, median, upper]
    error_list = [None, None, None]
    while(idx<iter):
        for i, offset in enumerate(new_offset_list):
            traj_est_cp = copy.deepcopy(traj_est)
            traj_gt_cp = copy.deepcopy(traj_gt)
            
            rpe_metric = calculate_RE(traj_est_cp, traj_gt_cp, est_offset=offset)
            error_list[i] = np.mean(np.array(rpe_metric.error))
            #ape_metric = calculate_APE(traj_est_cp, traj_gt_cp, est_offset=offset)
            #error_list[i] = ape_metric.get_statistic(metrics.StatisticsType.rmse)
        # conditions
        old_offset_list = new_offset_list
        if np.argmin(error_list) == 0:
            #print("Case 1")
            upper = median
            lower = lower - 0.5*(median-lower)
            median = (lower+median)/2
        elif np.argmax(error_list) == 2:
            #print("Case 2")
            upper = median
            #lower = lower
            median = (lower+median)/2
        elif np.argmax(error_list) == 0:
            #print("Case 3")
            lower = median
            median = (median+upper)/2
            #upper = upper
        elif np.argmin(error_list) == 2:
            #print("Case 4")
            upper = upper + 0.5*(upper-median)
            lower = median
            median = (upper+median)/2
        new_offset_list = [lower, median, upper]
        print(f"Iteration {idx} - Best offset = {old_offset_list[np.argmin(error_list)]} - checkpoint {old_offset_list}")
        idx += 1
    best_offset = old_offset_list[np.argmin(error_list)]
    return best_offset