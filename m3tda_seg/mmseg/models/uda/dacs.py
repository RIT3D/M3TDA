# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio

import pdb

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.source_only = cfg['source_only']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def predict_pseudo_label_and_weight(self, 
                                        target_img, target_img_metas, 
                                        valid_pseudo_mask):   
        ema_logits = self.get_ema_model().generate_pseudo_label(
            target_img, target_img_metas)
        debug_output = self.get_ema_model().debug_output
        pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
            ema_logits)
        pseudo_weight = self.filter_valid_pseudo_region(
            pseudo_weight, valid_pseudo_mask)
        
        del ema_logits
        return pseudo_label, pseudo_weight, debug_output
    
    def mix_source_target(self, 
                          img, img_metas, 
                          gt_semantic_seg, gt_pixel_weight,
                          trg_img, 
                          pseudo_label, pseudo_weight,
                          strong_parameters,
                          seg_debug_mode):
        
        batch_size = img.shape[0]
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mixed_seg_weight = pseudo_weight.clone()
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], trg_img[i])),
                target=torch.stack(
                    (gt_semantic_seg[i][0], pseudo_label[i])))
            _, mixed_seg_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        del gt_pixel_weight

        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        mix_losses = self.get_model().forward_train(
            mixed_img,
            img_metas,
            mixed_lbl,
            seg_weight=mixed_seg_weight,
            return_feat=False,
        )
        debug_output = self.get_model().debug_output

        if seg_debug_mode:
            return mix_losses, debug_output, \
                (mixed_img.detach(), mixed_lbl.detach(), mix_masks, mixed_seg_weight)
        else:
            return mix_losses, debug_output, \
                (None, None, None, None)
    
    def mask_training(self,
                      img, img_metas,
                      gt_semantic_seg,
                      trg_img, trg_img_metas,
                      valid_pseudo_mask,
                      pseudo_label, pseudo_weight):
        
        masked_loss = self.mic(self.get_model(), img, img_metas,
                               gt_semantic_seg, trg_img,
                               trg_img_metas, valid_pseudo_mask,
                               pseudo_label, pseudo_weight
        )
        
        mic_debug_output = self.mic.debug_output
        
        return masked_loss, mic_debug_output

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight
    


    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      target2_img,
                      target2_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug_mode = (self.local_iter % self.debug_img_interval == 0 and \
            not self.source_only)
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on Source Images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        pseudo_label, pseudo_weight = None, None
        pseudo2_label, pseudo2_weight = None, None
        # Train on Target Images
        if not self.source_only:
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            # Get target1 pseudo label
            pseudo_label, pseudo_weight, debug_output = self.predict_pseudo_label_and_weight(
                target_img, target_img_metas, 
                valid_pseudo_mask
            )
            # Debug target1 seg. output
            seg_debug['Target1'] = debug_output

            # Get target2 pseudo label
            pseudo2_label, pseudo2_weight, debug_output = self.predict_pseudo_label_and_weight(
                target2_img, target2_img_metas, 
                valid_pseudo_mask
            )
            # Debug target2 seg. output
            seg_debug['Target2'] = debug_output
            
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            # Train on the mixed target 1 and source (MixS1) image
            mixs1_losses, debug_output, mixs1_debug_content = self.mix_source_target(
                img, img_metas,
                gt_semantic_seg, gt_pixel_weight,
                target_img,
                pseudo_label, pseudo_weight,
                strong_parameters,
                seg_debug_mode,
            )
            # Debug the mixed (MixS1) image 
            seg_debug['MixS1'] = debug_output
            mixs1_losses = add_prefix(mixs1_losses, 'mixs1')
            mixs1_loss, mixs1_log_vars = self._parse_losses(mixs1_losses)
            log_vars.update(mixs1_log_vars)
            mixs1_loss.backward(retain_graph=True)

            # Train on the mixed target 2 and source (MixS2) image
            mixs2_losses, debug_output, mixs2_debug_content = self.mix_source_target(
                img, img_metas,
                gt_semantic_seg, gt_pixel_weight,
                target2_img,
                pseudo2_label, pseudo2_weight,
                strong_parameters,
                seg_debug_mode,
            )
            # Debug the mixed (MixS2) image
            seg_debug['MixS2'] = debug_output
            mixs2_losses = add_prefix(mixs1_losses, 'mixs2')
            mixs2_loss, mixs2_log_vars = self._parse_losses(mixs2_losses)
            log_vars.update(mixs2_log_vars)
            mixs2_loss.backward()
            
            if self.enable_masking and self.mask_mode.startswith('separate'):
                # Train on the masked image (here only mask on target1 image - Masked1)
                maskeds1_loss, mic_debug_output = self.mask_training(
                    img, img_metas,
                    gt_semantic_seg,
                    target_img, target_img_metas,
                    valid_pseudo_mask,
                    pseudo_label, pseudo_weight
                )
                # Masked1 debug & log record
                # pdb.set_trace()
                # mic_debug_output['Masked1'] = mic_debug_output.pop('Masked')
                # seg_debug.update(mic_debug_output)
                if seg_debug_mode:
                    seg_debug['Masked1'] = mic_debug_output['Masked']
                maskeds1_loss = add_prefix(maskeds1_loss, 'masked1')
                maskeds1_loss, masked1_log_vars = self._parse_losses(maskeds1_loss)
                log_vars.update(masked1_log_vars)
                maskeds1_loss.backward()
                # Train on the masked image (here only mask on target2 image - Masked2)
                maskeds2_loss, mic_debug_output = self.mask_training(
                    img, img_metas,
                    gt_semantic_seg,
                    target2_img, target2_img_metas,
                    valid_pseudo_mask,
                    pseudo2_label, pseudo2_weight,
                )
                # Masked2 debug & log record
                # mic_debug_output['MaskedS2'] = mic_debug_output.pop('Masked')
                # seg_debug.update(mic_debug_output)
                if seg_debug_mode:
                    seg_debug['Masked2'] = mic_debug_output['Masked']
                maskeds2_loss = add_prefix(maskeds2_loss, 'maskeds2')
                maskeds2_loss, masked2_log_vars = self._parse_losses(maskeds2_loss)
                log_vars.update(masked2_log_vars)
                maskeds2_loss.backward()

            if seg_debug_mode:
                out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
                os.makedirs(out_dir, exist_ok=True)
                vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
                vis_trg1_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
                vis_trg2_img = torch.clamp(denorm(target2_img, means, stds), 0, 1)
                vis_mixs1_img = torch.clamp(denorm(mixs1_debug_content[0], means, stds), 0, 1)
                vis_mixs2_img = torch.clamp(denorm(mixs2_debug_content[0], means, stds), 0, 1)
                for j in range(batch_size):
                    rows, cols = 4, 4
                    fig, axs = plt.subplots(
                        rows, 
                        cols,
                        figsize=(3*cols, 3*rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0        
                        },
                    )
                    subplotimg(axs[0][0], vis_img[j], 'source Image')
                    subplotimg(axs[1][0], vis_trg1_img[j], 'Target1 Image')
                    subplotimg(axs[2][0], vis_trg2_img[j], 'Target2 Image')
                    subplotimg(axs[0][1], gt_semantic_seg[j],
                              'Source Seg GT',cmap='cityscapes')
                    subplotimg(axs[1][1], pseudo_label[j],
                               'Target1 Seg (Pseudo) GT', cmap='cityscapes')
                    subplotimg(axs[2][1], pseudo2_label[j],
                               'Target2 Seg (Pseudo) GT', cmap='cityscapes')
                    # MixS1
                    subplotimg(axs[0][2], vis_mixs1_img[j], 
                               'MixS1 Image')
                    subplotimg(axs[1][2], mixs1_debug_content[2][j][0], 
                               'Domain Mask (MixS1)', cmap='gray')
                    subplotimg(axs[0][3], mixs1_debug_content[3][j],
                               'Pseudo W. (MixS1)', vmin=0, vmax=1)
                    if mixs1_debug_content[1] is not None:
                        subplotimg(axs[1][3], mixs1_debug_content[1][j], 
                                   'MixS1 Label', cmap='cityscapes')
                    # MixS2
                    subplotimg(axs[2][2], vis_mixs2_img[j], 
                               'MixS2 Image')
                    subplotimg(axs[3][2], mixs2_debug_content[2][j][0], 
                               'Domain Mask (MixS2)', cmap='gray')
                    subplotimg(axs[2][3], mixs2_debug_content[3][j],
                               'Pseudo W. (MixS2)', vmin=0, vmax=1)
                    if mixs2_debug_content[1] is not None:
                        subplotimg(axs[3][3], mixs2_debug_content[1][j], 
                                   'MixS2 Label', cmap='cityscapes')
                    for ax in axs.flat:
                            ax.axis('off')
                    plt.savefig(os.path.join(out_dir,
                                f'{(self.local_iter + 1):06d}_{j}.png'))
                    plt.close()

                out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
                os.makedirs(out_dir, exist_ok=True)
                if seg_debug['Source'] is not None and seg_debug:
                    if 'Target1' in seg_debug:
                        seg_debug['Target1']['Pseudo W.'] = mixs1_debug_content[3].cpu().numpy()
                    if 'Target2' in seg_debug:
                        seg_debug['Target2']['Pseudo W.'] = mixs2_debug_content[3].cpu().numpy()
                    for j in range(batch_size):
                        cols = len(seg_debug)
                        rows = max(len(seg_debug[k]) for k in seg_debug.keys())
                        fig, axs = plt.subplots(
                            rows,
                            cols,
                            figsize=(5 * cols, 5 * rows),
                            gridspec_kw={
                                'hspace': 0.1,
                                'wspace': 0,
                                'top': 0.95,
                                'bottom': 0,
                                'right': 1,
                                'left': 0
                            },
                            squeeze=False,
                        )
                        for k1, (n1, outs) in enumerate(seg_debug.items()):
                            for k2, (n2, out) in enumerate(outs.items()):
                                subplotimg(
                                    axs[k2][k1],
                                    **prepare_debug_out(f'{n1} {n2}', out[j],
                                                        means, stds))
                        for ax in axs.flat:
                            ax.axis('off')
                        plt.savefig(
                            os.path.join(out_dir,
                                        f'{(self.local_iter + 1):06d}_{j}_s.png'))
                        plt.close()
            del seg_debug
        self.local_iter += 1
        return log_vars
