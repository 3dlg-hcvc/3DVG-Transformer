# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

from utils.nn_distance import nn_distance, huber_loss
# from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou

from macro import *
def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one reference prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = box3d_iou(pred_bbox, gt_bbox)

    return iou


def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + center[0];
    corners_3d[1, :] = corners_3d[1, :] + center[1];
    corners_3d[2, :] = corners_3d[2, :] + center[2];
    corners_3d = np.transpose(corners_3d)

    return corners_3d


# def get_eval(data_dict, config, reference, use_lang_classifier=False, use_oracle=False, use_cat_rand=False,
#              use_best=False, post_processing=None, final_output=None, mem_hash=None):
#     """ Loss functions
#
#     Args:
#         data_dict: dict
#         config: dataset config instance
#         reference: flag (False/True)
#         post_processing: config dict
#     Returns:
#         loss: pytorch scalar tensor
#         data_dict: dict
#     """
#
#     #batch_size, num_words, _ = data_dict["lang_feat"].shape
#
#     objectness_preds_batch = torch.argmax(data_dict['objectness_scores'], 2).long()
#     objectness_labels_batch = data_dict['objectness_label'].long()
#
#     if post_processing:
#         _ = parse_predictions(data_dict, post_processing)
#         nms_masks = torch.cuda.LongTensor(data_dict['pred_mask'])
#
#         # construct valid mask
#         pred_masks = (nms_masks * objectness_preds_batch == 1).float()
#         label_masks = (objectness_labels_batch == 1).float()
#     else:
#         # construct valid mask
#         pred_masks = (objectness_preds_batch == 1).float()
#         label_masks = (objectness_labels_batch == 1).float()
#
#
#     #print("pred_masks", pred_masks.shape, label_masks.shape)
#
#
#     gt_center_list = torch.einsum("abc,adb->adc", data_dict["center_label"], data_dict["ref_box_label_list"].to(torch.float32))
#
#     batch_size, len_nun_max = gt_center_list.shape[:2]
#     cluster_preds = torch.argmax(data_dict["cluster_ref"], 1).long().unsqueeze(1).repeat(1, pred_masks.shape[1])
#
#
#
#     preds = torch.zeros(data_dict["cluster_ref"].shape, device="cuda")
#     preds = preds.scatter_(1, cluster_preds, 1)
#     cluster_preds = preds
#     cluster_labels = data_dict["cluster_labels"].reshape(batch_size*len_nun_max, -1).float()
#     #print(cluster_labels.shape, '<< cluster labels shape')
#     #cluster_labels *= label_masks
#
#     # compute classification scores
#     corrects = torch.sum((cluster_preds == 1) * (cluster_labels == 1), dim=1).float()
#     labels = torch.ones(corrects.shape[0], device="cuda")
#     ref_acc = corrects / (labels + 1e-8)
#
#     # store
#     data_dict["ref_acc"] = ref_acc.cpu().numpy().tolist()
#
#     # # scanrefer++ support, use threshold to filter predictions instead of argmax
#     if SCANREFER_ENHANCE:
#         pred_ref_mul_obj_mask = torch.logical_and((torch.sigmoid(data_dict["cluster_ref"]) >= SCANREFER_ENHANCE_EVAL_THRESHOLD), pred_masks.bool().repeat(1, len_nun_max).reshape(batch_size*len_nun_max, -1)).cpu().numpy()
#
#         # pred_ref_mul_obj_mask = torch.logical_and((torch.nn.functional.softmax(data_dict["cluster_ref"], dim=1) >= 0.3), pred_masks.bool())
#     # # end
#
#     # compute localization metricens
#     if use_best:
#         pred_ref = torch.argmax(data_dict["cluster_labels"], 1)  # (B,)
#         # store the calibrated predictions and masks
#         data_dict['cluster_ref'] = data_dict["cluster_labels"]
#     if use_cat_rand:
#         cluster_preds = torch.zeros(cluster_labels.shape, device="cuda")
#         for i in range(cluster_preds.shape[0]):
#             num_bbox = data_dict["num_bbox"][i]
#             sem_cls_label = data_dict["sem_cls_label"][i]
#             # sem_cls_label = torch.argmax(end_points["sem_cls_scores"], 2)[i]
#             sem_cls_label[num_bbox:] -= 1
#             candidate_masks = torch.gather(sem_cls_label == data_dict["object_cat"][i], 0, data_dict["object_assignment"][i])
#             candidates = torch.arange(cluster_labels.shape[1])[candidate_masks]
#             try:
#                 chosen_idx = torch.randperm(candidates.shape[0])[0]
#                 chosen_candidate = candidates[chosen_idx]
#                 cluster_preds[i, chosen_candidate] = 1
#             except IndexError:
#                 cluster_preds[i, candidates] = 1
#
#         pred_ref = torch.argmax(cluster_preds, -1)  # (B,)
#         # store the calibrated predictions and masks
#         data_dict['cluster_ref'] = cluster_preds
#     else:
#         #pred_ref = torch.argmax(data_dict['cluster_ref'], 1)  # (B,)
#         pred_mask1 = pred_masks[0].repeat(len_nun_max, 1)
#         for i in range(batch_size):
#             if i != 0:
#                 pred_mask = pred_masks[i].repeat(len_nun_max, 1)
#                 pred_mask1 = torch.cat([pred_mask1, pred_mask], dim=0)
#         pred_ref = torch.argmax(data_dict['cluster_ref'] * pred_mask1, 1)  # (B,)
#         # store the calibrated predictions and masks
#         #data_dict['cluster_ref'] = data_dict['cluster_ref'] * pred_masks
#
#     if use_oracle:
#         pred_center = data_dict['center_label']  # (B,MAX_NUM_OBJ,3)
#         pred_heading_class = data_dict['heading_class_label']  # B,K2
#         pred_heading_residual = data_dict['heading_residual_label']  # B,K2
#         pred_size_class = data_dict['size_class_label']  # B,K2
#         pred_size_residual = data_dict['size_residual_label']  # B,K2,3
#
#         # assign
#         pred_center = torch.gather(pred_center, 1, data_dict["object_assignment"].unsqueeze(2).repeat(1, 1, 3))
#         pred_heading_class = torch.gather(pred_heading_class, 1, data_dict["object_assignment"])
#         pred_heading_residual = torch.gather(pred_heading_residual, 1, data_dict["object_assignment"]).unsqueeze(-1)
#         pred_size_class = torch.gather(pred_size_class, 1, data_dict["object_assignment"])
#         pred_size_residual = torch.gather(pred_size_residual, 1, data_dict["object_assignment"].unsqueeze(2).repeat(1, 1, 3))
#     else:
#         pred_center = data_dict['center']  # (B,K,3)
#         pred_heading_class = torch.argmax(data_dict['heading_scores'], -1)  # B,num_proposal
#         pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2,
#                                              pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
#         pred_heading_class = pred_heading_class  # B,num_proposal
#         pred_heading_residual = pred_heading_residual.squeeze(2)  # B,num_proposal
#         pred_size_class = torch.argmax(data_dict['size_scores'], -1)  # B,num_proposal
#         pred_size_residual = torch.gather(data_dict['size_residuals'], 2,
#                                           pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
#                                                                                              3))  # B,num_proposal,1,3
#         pred_size_class = pred_size_class
#         pred_size_residual = pred_size_residual.squeeze(2)  # B,num_proposal,3
#
#     # store
#     data_dict["pred_mask"] = pred_masks
#     data_dict["label_mask"] = label_masks
#     data_dict['pred_center'] = pred_center
#     data_dict['pred_heading_class'] = pred_heading_class
#     data_dict['pred_heading_residual'] = pred_heading_residual
#     data_dict['pred_size_class'] = pred_size_class
#     data_dict['pred_size_residual'] = pred_size_residual
#
#     #print("ref_box_label", data_dict["ref_box_label"].shape, data_dict["ref_box_label_list"].shape)
#     #gt_ref = torch.argmax(data_dict["ref_box_label"], 1)
#     gt_ref = torch.argmax(data_dict["ref_box_label_list"], -1)
#     gt_center = data_dict['center_label']  # (B,MAX_NUM_OBJ,3)
#     gt_heading_class = data_dict['heading_class_label']  # B,K2
#     gt_heading_residual = data_dict['heading_residual_label']  # B,K2
#     gt_size_class = data_dict['size_class_label']  # B,K2
#     gt_size_residual = data_dict['size_residual_label']  # B,K2,3
#     lang_num = data_dict["lang_num"]
#
#     ious = []
#     multiple = []
#     others = []
#     pred_bboxes = []
#     gt_bboxes = []
#     #print("pred_ref", pred_ref.shape, gt_ref.shape)
#     pred_ref = pred_ref.reshape(batch_size, len_nun_max)
#     if SCANREFER_ENHANCE:
#         pred_ref_mul_obj_mask = pred_ref_mul_obj_mask.reshape(batch_size, len_nun_max, -1)
#
#     for i in range(batch_size):
#         # compute the iou
#         for j in range(len_nun_max):
#             if j < lang_num[i]:
#                 pred_ref_idx, gt_ref_idx = pred_ref[i][j], gt_ref[i][j]
#                 pred_obb = config.param2obb(
#                     pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy(),
#                     pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(),
#                     pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
#                     pred_size_class[i, pred_ref_idx].detach().cpu().numpy(),
#                     pred_size_residual[i, pred_ref_idx].detach().cpu().numpy()
#                 )
#                 gt_obb = config.param2obb(
#                     gt_center[i, gt_ref_idx, 0:3].detach().cpu().numpy(),
#                     gt_heading_class[i, gt_ref_idx].detach().cpu().numpy(),
#                     gt_heading_residual[i, gt_ref_idx].detach().cpu().numpy(),
#                     gt_size_class[i, gt_ref_idx].detach().cpu().numpy(),
#                     gt_size_residual[i, gt_ref_idx].detach().cpu().numpy()
#                 )
#                 pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
#                 gt_bbox = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3])
#                 iou = eval_ref_one_sample(pred_bbox, gt_bbox)
#                 ious.append(iou)
#
#                 # NOTE: get_3d_box() will return problematic bboxes
#                 pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
#                 gt_bbox = construct_bbox_corners(gt_obb[0:3], gt_obb[3:6])
#                 pred_bboxes.append(pred_bbox)
#                 gt_bboxes.append(gt_bbox)
#
#                 # construct the multiple mask
#                 multiple.append(data_dict["unique_multiple_list"][i][j].item())
#
#                 # construct the others mask
#                 flag = 1 if data_dict["object_cat_list"][i][j] == 17 else 0
#                 others.append(flag)
#
#                 # scanrefer++ support
#                 if SCANREFER_ENHANCE:
#                     multi_pred_bboxes = []
#                     multi_pred_ref_idxs = pred_ref_mul_obj_mask[i][j].nonzero()
#                     for idx in multi_pred_ref_idxs[0]:
#                         if USE_GT:
#                             pred_obb = config.param2obb(
#                                 data_dict["center_label"][i, idx,0:3].cpu().numpy(),
#                                 data_dict["heading_class_label"][i, idx].cpu().numpy(),
#                                 data_dict["heading_residual_label"][i, idx].cpu().numpy(),
#                                 data_dict["size_class_label"][i, idx].cpu().numpy(),
#                                 data_dict["size_residual_label"][i, idx].cpu().numpy()
#                             )
#                         else:
#                             pred_obb = config.param2obb(
#                                 pred_center[i, idx, 0:3].detach().cpu().numpy(),
#                                 pred_heading_class[i, idx].detach().cpu().numpy(),
#                                 pred_heading_residual[i, idx].detach().cpu().numpy(),
#                                 pred_size_class[i, idx].detach().cpu().numpy(),
#                                 pred_size_residual[i, idx].detach().cpu().numpy()
#                             )
#                         # pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
#                         pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
#                         multi_pred_bboxes.append(pred_bbox)
#                     output_info = {
#                         "object_id": data_dict["object_id"][i][j].item(),
#                         "ann_id": data_dict["ann_id"][i][j].item(),
#                         "aabbs": multi_pred_bboxes
#                     }
#                     scene_id = data_dict["scene_id"][i]
#                     key = (scene_id, output_info["object_id"], output_info["ann_id"])
#                     if final_output is not None and key not in mem_hash:
#                         final_output[scene_id].append(output_info)
#                     mem_hash[key] = True
#                 # end
#
#     # lang
#     if reference and use_lang_classifier:
#         object_cat = data_dict["object_cat_list"].reshape(batch_size*len_nun_max)
#         data_dict["lang_acc"] = (torch.argmax(data_dict['lang_scores'], 1) == object_cat).float().mean()
#     else:
#         data_dict["lang_acc"] = torch.zeros(1)[0].cuda()
#
#     # store
#     data_dict["ref_iou"] = ious
#     data_dict["ref_iou_rate_0.25"] = np.array(ious)[np.array(ious) >= 0.25].shape[0] / np.array(ious).shape[0]
#     data_dict["ref_iou_rate_0.5"] = np.array(ious)[np.array(ious) >= 0.5].shape[0] / np.array(ious).shape[0]
#     data_dict["ref_multiple_mask"] = multiple
#     data_dict["ref_others_mask"] = others
#     data_dict["pred_bboxes"] = pred_bboxes
#     data_dict["gt_bboxes"] = gt_bboxes
#
#
#     # --------------------------------------------
#     # Some other statistics
#     if not USE_GT:
#         obj_pred_val = torch.argmax(data_dict['objectness_scores'], 2)  # B,K
#         obj_acc = torch.sum(
#             (obj_pred_val == data_dict['objectness_label'].long()).float() * data_dict['objectness_mask']) / (
#                               torch.sum(data_dict['objectness_mask']) + 1e-6)
#         data_dict['obj_acc'] = obj_acc
#         # detection semantic classification
#         sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1,
#                                      data_dict['object_assignment'])  # select (B,K) from (B,K2)
#         sem_cls_pred = data_dict['sem_cls_scores'].argmax(-1)  # (B,K)
#         sem_match = (sem_cls_label == sem_cls_pred).float()
#         data_dict["sem_acc"] = (sem_match * data_dict["pred_mask"]).sum() / data_dict["pred_mask"].sum()
#
#     return data_dict


def get_eval_multi3drefer(data_dict, config):
    batch_size, lang_chunk_size = data_dict["ann_id"].shape

    pred_aabb_score_masks = (
            torch.sigmoid(data_dict["cluster_ref"]) >= 0.1
    ).reshape(shape=(batch_size, lang_chunk_size, -1)).cpu().numpy()

    pred_results = {}

    pred_center = data_dict['center'].detach().cpu().numpy()  # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3
    gts = {}
    new_box_mask = data_dict["multi_ref_box_label_list"]

    for i in range(batch_size):

        pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i],
                                                pred_heading_residual[i],
                                                pred_size_class[i], pred_size_residual[i])
        pred_bbox_corners = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
        min_max_bound = np.stack((pred_bbox_corners.min(1), pred_bbox_corners.max(1)), axis=1)

        for j in range(lang_chunk_size):
            if j < data_dict["lang_num"][i]:
                pred_aabbs = min_max_bound[pred_aabb_score_masks[i, j]]
                pred_results[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                     data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound": pred_aabbs
                }

                single_box_mask = new_box_mask[i][j]

                gt_bboxes_centers = data_dict["center_label"][i][single_box_mask].cpu().numpy()
                gt_heading_class_labels = data_dict["heading_class_label"][i][single_box_mask].cpu().numpy()
                gt_heading_residual_labels = data_dict["heading_residual_label"][i][single_box_mask].cpu().numpy()
                gt_size_class_labels = data_dict["size_class_label"][i][single_box_mask].cpu().numpy()
                gt_bboxes_residuals = data_dict["size_residual_label"][i][single_box_mask].cpu().numpy()

                gt_obb_batch_new = config.param2obb_batch(gt_bboxes_centers[:, 0:3], gt_heading_class_labels,
                                                          gt_heading_residual_labels,
                                                          gt_size_class_labels, gt_bboxes_residuals)

                gt_bbox_batch_new = get_3d_box_batch(gt_obb_batch_new[:, 3:6], gt_obb_batch_new[:, 6],
                                                     gt_obb_batch_new[:, 0:3])

                # gt_bboxes = gt_bboxes_list[i][single_mask]
                gt_bboxes_bound = np.stack((gt_bbox_batch_new.min(1), gt_bbox_batch_new.max(1)), axis=1)

                gts[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                     data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound": gt_bboxes_bound,
                    "eval_type": data_dict["eval_type"][j][i]
                }


    # batch_size = len(data_dict["scene_id"])
    # chunk_size = data_dict["ann_id"].shape[1]
    # box_masks = data_dict["multi_ref_box_label_list"]

    # gt_bboxes_list = data_dict["gt_bbox"]



    # gt_target_obj_id_masks = data_dict["gt_target_obj_id_mask"].permute(1, 0)FF
    # for i in range(batch_size):
    #     # aabb_start_idx = data_dict["aabb_count_offsets"][i]
    #     # aabb_end_idx = data_dict["aabb_count_offsets"][i + 1]
    #
    #
    #     for j in range(chunk_size):
    #
    #         if j < data_dict["lang_num"][i]:

    return pred_results, gts

    # objectness_preds_batch = torch.argmax(data_dict['objectness_scores'], 2).long()
    # pred_masks = (objectness_preds_batch == 1).float()
    #
    # gt_center_list = torch.einsum("abc,adb->adc", data_dict["center_label"],
    #                               data_dict["ref_box_label_list"].to(torch.float32))
    #
    # batch_size, len_nun_max = gt_center_list.shape[:2]
    #
    # pred_ref_mul_obj_mask = torch.logical_and(
    #     (torch.sigmoid(data_dict["cluster_ref"]) >= SCANREFER_ENHANCE_EVAL_THRESHOLD),
    #     pred_masks.bool().repeat(1, len_nun_max).reshape(batch_size * len_nun_max, -1)).cpu().numpy()
    #
    # pred_ref_mul_obj_mask = pred_ref_mul_obj_mask.reshape(batch_size, len_nun_max, -1)
    #
    # lang_num = data_dict["lang_num"]
    #
    # pred_mask1 = pred_masks[0].repeat(len_nun_max, 1)
    # for i in range(batch_size):
    #     if i != 0:
    #         pred_mask = pred_masks[i].repeat(len_nun_max, 1)
    #         pred_mask1 = torch.cat([pred_mask1, pred_mask], dim=0)
    # pred_ref = torch.argmax(data_dict['cluster_ref'] * pred_mask1, 1)
    #
    # pred_ref = pred_ref.reshape(batch_size, len_nun_max)
    # pred_center = data_dict['center']  # (B,K,3)
    #
    # gt_ref = torch.argmax(data_dict["ref_box_label_list"], -1)
    #
    # pred_heading_class = torch.argmax(data_dict['heading_scores'], -1)  # B,num_proposal
    # pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2,
    #                                      pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    # pred_heading_class = pred_heading_class  # B,num_proposal
    # pred_heading_residual = pred_heading_residual.squeeze(2)  # B,num_proposal
    # pred_size_class = torch.argmax(data_dict['size_scores'], -1)  # B,num_proposal
    # pred_size_residual = torch.gather(data_dict['size_residuals'], 2,
    #                                   pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
    #                                                                                      3))  # B,num_proposal,1,3
    # pred_size_class = pred_size_class
    # pred_size_residual = pred_size_residual.squeeze(2)  # B,num_proposal,3
    # pred_dict = {}
    # for i in range(batch_size):
    #     # compute the iou
    #     for j in range(len_nun_max):
    #         if j < lang_num[i]:
    #             pred_ref_idx, gt_ref_idx = pred_ref[i][j], gt_ref[i][j]
    #             pred_obb = config.param2obb(
    #                 pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy(),
    #                 pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(),
    #                 pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
    #                 pred_size_class[i, pred_ref_idx].detach().cpu().numpy(),
    #                 pred_size_residual[i, pred_ref_idx].detach().cpu().numpy()
    #             )
    #
    #             multi_pred_bboxes = []
    #             multi_pred_ref_idxs = pred_ref_mul_obj_mask[i][j].nonzero()
    #             for idx in multi_pred_ref_idxs[0]:
    #                 if USE_GT:
    #                     pred_obb = config.param2obb(
    #                         data_dict["center_label"][i, idx, 0:3].cpu().numpy(),
    #                         data_dict["heading_class_label"][i, idx].cpu().numpy(),
    #                         data_dict["heading_residual_label"][i, idx].cpu().numpy(),
    #                         data_dict["size_class_label"][i, idx].cpu().numpy(),
    #                         data_dict["size_residual_label"][i, idx].cpu().numpy()
    #                     )
    #                 else:
    #                     pred_obb = config.param2obb(
    #                         pred_center[i, idx, 0:3].detach().cpu().numpy(),
    #                         pred_heading_class[i, idx].detach().cpu().numpy(),
    #                         pred_heading_residual[i, idx].detach().cpu().numpy(),
    #                         pred_size_class[i, idx].detach().cpu().numpy(),
    #                         pred_size_residual[i, idx].detach().cpu().numpy()
    #                     )
    #                 # pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
    #                 pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
    #                 multi_pred_bboxes.append(pred_bbox)
    #
    #             pred_dict[(data_dict["scene_id"][i].item(), data_dict["object_id"][i][j].item(), data_dict["ann_id"][i][j].item())] = {
    #
    #             }
    #             output_info = {
    #                 "scene_id": data_dict["scene_id"][i].item(),
    #                 "object_id": data_dict["object_id"][i][j].item(),
    #                 "ann_id": data_dict["ann_id"][i][j].item(),
    #                 "aabbs": multi_pred_bboxes
    #             }