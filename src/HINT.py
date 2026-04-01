import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel, LandmarkDetectorModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR
from cv2 import circle
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import wandb
import lpips
import torchvision
import time

'''
This repo is modified basing on Edge-Connect
https://github.com/knazeri/edge-connect
'''

class HINT():
    def __init__(self, config):
        self.config = config


        if config.MODEL == 2:
            model_name = 'inpaint'

        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.landmark_model = LandmarkDetectorModel(config).to(config.DEVICE) if config.USE_LANDMARKS else None

        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')

        #train mode
        if self.config.MODE == 1:

            if self.config.MODEL == 2:
                self.train_dataset = Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST, config.TRAIN_MASK_FLIST, 
                                             landmark_flist=config.TRAIN_LANDMARK_LANDMARK_FLIST if config.USE_LANDMARKS else None,
                                             augment=True, training=True)

        # test mode
        if self.config.MODE == 2:
            if self.config.MODEL == 2:
                print('model == 2')
                self.test_dataset = Dataset(config, config.TEST_INPAINT_IMAGE_FLIST, config.TEST_MASK_FLIST,
                                            landmark_flist=config.TEST_LANDMARK_LANDMARK_FLIST if config.USE_LANDMARKS else None,
                                            augment=False, training=False)


        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):

        if self.config.MODEL == 2:
            self.inpaint_model.load()
            if self.landmark_model is not None:
                self.landmark_model.load()


    def save(self):
        if self.config.MODEL == 2:
            self.inpaint_model.save()
            if self.landmark_model is not None:
                self.landmark_model.save()


    def train(self):
        wandb.watch(self.inpaint_model, self.psnr, log='all', log_freq=10)
        
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])


            for items in train_loader:

                self.inpaint_model.train()
                if self.landmark_model is not None:
                    self.landmark_model.train()

                if model == 2:
                    if self.config.USE_LANDMARKS:
                        images, landmarks_gt, masks = self.cuda(*items)
                        landmark_map = self.generate_landmark_map(landmarks_gt)
                    else:
                        images, masks = self.cuda(*items)
                        landmark_map = None

                    # inpaint model
                    outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss= self.inpaint_model.process(images, landmark_map, masks)
                    outputs_merged = (outputs_img * masks) + (images * (1-masks))

                    psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    self.inpaint_model.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                if iteration % 10 == 0:
                        wandb.log({'gen_loss': gen_loss, 'l1_loss': gen_l1_loss, 'style_loss': gen_style_loss,
                                   'perceptual loss': gen_content_loss, 'gen_gan_loss': gen_gan_loss,
                                   'dis_loss': dis_loss}, step=iteration)

                ###################### visialization
                if iteration % 40 == 0:
                    create_dir(self.results_path)
                    inputs = (images * (1 - masks))
                    images_joint = stitch_images(
                        self.postprocess(images),
                        self.postprocess(inputs),
                        self.postprocess(outputs_img),
                        self.postprocess(outputs_merged),
                        img_per_row=1
                    )


                    path_masked = os.path.join(self.results_path,self.model_name,'masked')
                    path_result = os.path.join(self.results_path, self.model_name,'result')
                    path_joint = os.path.join(self.results_path,self.model_name,'joint')
                    name = self.train_dataset.load_name(epoch-1)[:-4]+'.png'

                    create_dir(path_masked)
                    create_dir(path_result)
                    create_dir(path_joint)

                    masked_images = self.postprocess(images*(1-masks)+masks)[0]
                    images_result = self.postprocess(outputs_merged)[0]

                    print(os.path.join(path_joint,name[:-4]+'.png'))

                    images_joint.save(os.path.join(path_joint,name[:-4]+'.png'))
                    imsave(masked_images,os.path.join(path_masked,name))
                    imsave(images_result,os.path.join(path_result,name))

                    print(name + ' complete!')
                    
                ##############

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
        print('\nEnd training....')


    def test(self):

        self.inpaint_model.eval()
        model = self.config.MODEL
        create_dir(self.results_path)
        cal_mean_nme = self.cal_mean_nme()

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            shuffle=False
        )
        
        psnr_list = []
        ssim_list = []
        l1_list = []
        lpips_list = []
        nme_list = []

        cal_mean_nme = self.cal_mean_nme_tracker()
        
        print('here')
        index = 0
        for items in test_loader:
            if self.config.USE_LANDMARKS:
                images, landmarks_gt, masks = self.cuda(*items)
            else:
                images, masks = self.cuda(*items)
            
            # inpaint model
            if model == 2:
                if self.config.USE_LANDMARKS:
                    # Predict landmarks if in joint mode (Stage 3 logic basically)
                    if self.landmark_model is not None:
                        landmark_pred = self.landmark_model(images, masks)
                        landmark_pred = landmark_pred.reshape(-1, self.config.LANDMARK_POINTS, 2).long()
                        landmark_map = self.generate_landmark_map(landmark_pred)
                    else:
                        landmark_map = self.generate_landmark_map(landmarks_gt)
                else:
                    landmark_map = None

                inputs = (images * (1 - masks))
                with torch.no_grad():
                    tsince = int(round(time.time()*1000))
                    outputs_img = self.inpaint_model(images, landmark_map, masks)
                    ttime_elapsed = int(round(time.time()*1000))-tsince
                    print('test batch time elapsed {}ms'.format(ttime_elapsed))
                
                outputs_merged = (outputs_img * masks) + (images * (1 - masks))
                
                # Loop through each sample in the batch
                batch_size = images.shape[0]
                for i in range(batch_size):
                    sample_psnr, sample_ssim = self.metric(images[i], outputs_merged[i])
                    psnr_list.append(sample_psnr)
                    ssim_list.append(sample_ssim)
                    
                    if torch.cuda.is_available():
                        pl = self.loss_fn_vgg(self.transf(outputs_merged[i].cpu().unsqueeze(0)).cuda(), 
                                             self.transf(images[i].cpu().unsqueeze(0)).cuda()).item()
                    else:
                        pl = self.loss_fn_vgg(self.transf(outputs_merged[i].cpu().unsqueeze(0)), 
                                             self.transf(images[i].cpu().unsqueeze(0))).item()
                    lpips_list.append(pl)                
                    
                    l1_loss = torch.nn.functional.l1_loss(outputs_merged[i], images[i], reduction='mean').item()
                    l1_list.append(l1_loss)

                    # NME Calculation
                    if self.config.USE_LANDMARKS:
                        # If we predicted landmarks (Mode 3 logic or if model 2 uses predictor)
                        if 'landmark_pred' in locals():
                            nme = self.cal_nme(landmarks_gt[i], landmark_pred[i])
                            nme_list.append(nme)
                            cal_mean_nme(nme)
                        else:
                            # If we are using ground truth for the map, NME is effectively for the GT input (0)
                            # or we can evaluate the detector if available
                            if self.landmark_model is not None:
                                landmark_eval = self.landmark_model(images[i].unsqueeze(0), masks[i].unsqueeze(0)).reshape(self.config.LANDMARK_POINTS, 2)
                                nme = self.cal_nme(landmarks_gt[i], landmark_eval)
                                nme_list.append(nme)
                                cal_mean_nme(nme)

                    sample_name = self.test_dataset.load_name(index * self.config.BATCH_SIZE + i)[:-4] + '.png'
                    metric_str = "Sample {}: psnr:{:.2f} ssim:{:.4f} l1:{:.4f} lpips:{:.4f}".format(
                        sample_name, sample_psnr, sample_ssim, l1_loss, pl)
                    if self.config.USE_LANDMARKS and nme_list:
                        metric_str += " nme:{:.4f} mean_nme:{:.4f}".format(nme_list[-1], cal_mean_nme.mean_nme)
                    print(metric_str + " {}".format(len(ssim_list)))

                    # Save individual results
                    path_masked = os.path.join(self.results_path, self.model_name, 'masked')
                    path_result = os.path.join(self.results_path, self.model_name, 'result')
                    create_dir(path_masked)
                    create_dir(path_result)

                    masked_img_i = self.postprocess((images[i] * (1 - masks[i]) + masks[i]).unsqueeze(0))[0]
                    result_img_i = self.postprocess(outputs_merged[i].unsqueeze(0))[0]
                    
                    imsave(masked_img_i, os.path.join(path_masked, sample_name))
                    imsave(result_img_i, os.path.join(path_result, sample_name))

                # Batch visualization logic if needed
                if self.debug or index % 10 == 0:
                    images_joint = stitch_images(
                        self.postprocess(images),
                        self.postprocess(inputs),
                        self.postprocess(outputs_img),
                        self.postprocess(outputs_merged),
                        img_per_row=1
                    )
                    path_joint = os.path.join(self.results_path, self.model_name, 'joint')
                    create_dir(path_joint)
                    joint_name = 'batch_{:04d}.png'.format(index)
                    images_joint.save(os.path.join(path_joint, joint_name))

            index += 1

        print('\nEnd Testing')
        final_str = 'Average PSNR: {:.4f}, SSIM: {:.4f}, L1: {:.4f}, LPIPS: {:.4f}'.format(
            np.average(psnr_list), np.average(ssim_list), np.average(l1_list), np.average(lpips_list))
        if nme_list:
            final_str += ', Average NME: {:.4f}'.format(np.average(nme_list))
        print(final_str)





    def log(self, logs):
        with open(self.log_file, 'a') as f:
            print('load the generator:')
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            print('finish load')

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def generate_landmark_map(self, landmark_cord):
        img_size = self.config.INPUT_SIZE
        if torch.is_tensor(landmark_cord):
            if landmark_cord.ndimension() == 3:
                landmark_img = torch.zeros(landmark_cord.shape[0], 1, img_size, img_size).to(landmark_cord.device)
                for i in range(landmark_cord.shape[0]):
                    # clamp coordinates to be within grid
                    l_y = landmark_cord[i, :, 1].clamp(0, img_size - 1)
                    l_x = landmark_cord[i, :, 0].clamp(0, img_size - 1)
                    landmark_img[i, 0, l_y, l_x] = 1
            elif landmark_cord.ndimension() == 2:
                landmark_img = torch.zeros(1, img_size, img_size).to(landmark_cord.device)
                l_y = landmark_cord[:, 1].clamp(0, img_size - 1)
                l_x = landmark_cord[:, 0].clamp(0, img_size - 1)
                landmark_img[0, l_y, l_x] = 1
        return landmark_img


    def metric(self, gt, pre):
        # gt and pre are [C, H, W] tensors
        pre = (pre.clamp(0, 1) * 255.0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        gt = (gt.clamp(0, 1) * 255.0).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, channel_axis=-1, data_range=255)

        return psnr, ssim
    
    def cal_nme(self, landmarks_gt, landmarks_pred):
        # landmarks are [points, 2] tensors
        landmarks_gt = landmarks_gt.float()
        landmarks_pred = landmarks_pred.float()
        
        if self.config.LANDMARK_POINTS == 68:
            # Inter-ocular distance normalization (inner corners of eyes)
            # Index 36: left eye outer corner, 45: right eye outer corner in 0-indexed CelebA-68
            point1 = landmarks_gt[36, :]
            point2 = landmarks_gt[45, :]
        elif self.config.LANDMARK_POINTS == 98:
            point1 = landmarks_gt[60, :]
            point2 = landmarks_gt[72, :]
        else:
            distance = 1.0
            
        distance = torch.norm(point1 - point2).item()
        if distance == 0: distance = 1.0
        
        # Calculate mean error across all points
        error = torch.mean(torch.norm(landmarks_pred - landmarks_gt, dim=1)).item()
        return error / distance
    
    class cal_mean_nme_tracker():
        sum = 0
        amount = 0
        mean_nme = 0

        def __call__(self, nme):
            self.sum += nme
            self.amount += 1
            self.mean_nme = self.sum / self.amount
            return self.mean_nme

        def get_mean_nme(self):
            return self.mean_nme

