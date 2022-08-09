import os
import logging
import warnings
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#from datasets.generic import Batch
from model.RAFTSceneFlow import RSF_DGCNN
from tools.loss import sequence_loss
from tools.metric import compute_epe_train, compute_epe2, compute_rmse_train, compute_rmse
from tools.utils import save_checkpoint
from deepptv.data import FluidflowDataset, FluidflowDataset3D

class Trainer(object):
    def __init__(self, args, mode='Train'):
        self.args = args
        self.root = args.root
        self.exp_path = args.exp_path
        self.dataset = args.dataset
        self.log_dir = None
        self.summary_writer = None

        self._log_init(mode)

        if self.dataset == 'PTVflow2D':
            folder = 'sample5000_particle50_plus' # 'data_sample'
            dataset_path = os.path.join(self.root, 'data', folder)
            self.train_dataset = FluidflowDataset(npoints=args.num_points, root = dataset_path, partition='train')
            self.val_dataset = FluidflowDataset(npoints=args.num_points, root = dataset_path, partition='test')
            self.test_dataset = self.val_dataset
        elif self.dataset == 'PTVflow3D':
            folder = 'PTVflow3D_norm' # 'data_sample'
            dataset_path = os.path.join('data/training_set', folder)
            self.train_dataset = FluidflowDataset3D(npoints=args.num_points, root = dataset_path, partition='train')
            self.val_dataset = FluidflowDataset3D(npoints=args.num_points, root = dataset_path, partition='test')
            self.test_dataset = self.val_dataset
        else:
            raise NotImplementedError

        self.train_dataloader = DataLoader(self.train_dataset, args.batch_size, shuffle=True,
                                           num_workers=4, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, args.test_batch_size, shuffle=False, num_workers=4,
                                         drop_last=False)
        self.test_dataloader = self.val_dataloader

        model = RSF_DGCNN(args)

        if torch.cuda.device_count() > 1:
            self.device = list(range(torch.cuda.device_count()))
        else:
            self.device = ['cuda'] if torch.cuda.is_available() else ['cpu']
        self.model = model.to(self.device[0])

        learning_rate = 1e-3
        # weight_decay = 1e-3
        # self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        # Optimizer with specific learning rate for ot
        deform_params = [v for k, v in self.model.named_parameters() if 'offset' in k]
        other_params = [v for k, v in self.model.named_parameters() if 'offset' not in k]
        deform_lr = 0.1 * learning_rate
        self.optimizer = Adam([{'params': other_params},
                            {'params': deform_params, 'lr': deform_lr}],
                            lr=learning_rate)
                            # weight_decay=weight_decay)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=args.num_epochs*len(self.train_dataset))

        self.begin_epoch = 1
        self._load_weights()

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.device)

        if self.begin_epoch > 0:
            for _ in range(self.begin_epoch):
                self.lr_scheduler.step()

        self.best_val_epe = 10

    def _log_init(self, mode='Train'):
        if self.exp_path is None:
            self.exp_path = datetime.now().strftime("exp-%y_%m_%d-%H_%M_%S_%f")
            self.args.exp_path = self.exp_path
        self.exp_path = os.path.join(self.root, 'experiments', self.exp_path)
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)

        log_dir = os.path.join(self.exp_path, 'logs')
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_name = mode + '_' + self.dataset + '.log'
        logging.basicConfig(
            filename=os.path.join(log_dir, log_name),
            filemode='w',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO
        )
        warnings.filterwarnings('ignore')

        ckpt_dir = os.path.join(self.exp_path, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        logging.info(self.args)
        logging.info('')

    def _load_weights(self, test_best=False):
        if self.args.weights is not None:
            if os.path.exists(self.args.weights):
                checkpoint = torch.load(self.args.weights)
                self.begin_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                print("Load checkpoint from {}".format(self.args.weight_path)) # *(self.args.weight_path) ==> (weight_path)
            else:
                raise RuntimeError(f"=> No checkpoint found at '{self.args.weights}")
        if test_best:
            weight_path = os.path.join(self.root, 'experiments', self.args.exp_path, 'checkpoints',
                                       'best_checkpoint.params')
            if os.path.exists(weight_path):
                checkpoint = torch.load(weight_path)
                if torch.cuda.device_count() > 1:
                    self.model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint['state_dict'])
                print("Load checkpoint from {}".format(weight_path))
            else:
                raise RuntimeError(f"=> No checkpoint found at '{weight_path}")


    def training(self, epoch):
        self.model.train()
        if self.summary_writer is None:
            self.summary_writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        loss_train = []
        epe_train = []
        rmse_train1 = [] ###### *modified* ######
        rmse_train2 = [] ###### *modified* ######

        train_progress = tqdm(self.train_dataloader, total=len(self.train_dataloader), smoothing=0.9) # ncols=150
        for i, batch_data in enumerate(train_progress):
            global_step = epoch * len(self.train_dataloader) + i
            # batch_data = batch_data.to(self.device[0])
            ###### *modified* ######
            # pc1, pc2, color1, color2, flow = batch_data   
            pc1, pc2, flow = batch_data  
            mask = torch.ones_like(flow)[:,:,0]
            batch_data = {"sequence": [pc1, pc2], "ground_truth": [mask, flow]}   
            for key in batch_data.keys():
                batch_data[key] = [d.to(self.device[0]) for d in batch_data[key]]
            ###### *modified* ######

            self.optimizer.zero_grad()
            
            est_flow = self.model(batch_data["sequence"], num_iters=self.args.iters)
            loss = sequence_loss(est_flow, batch_data, gamma=self.args.gamma)

            loss.backward()
            self.optimizer.step()

            epe = compute_epe_train(est_flow[-1], batch_data)
            loss_train.append(loss.detach().cpu())
            epe_train.append(epe.detach().cpu())

            ###### *modified* ######
            rmse1, rmse2 = compute_rmse_train(est_flow[-1], batch_data)
            rmse_train1.append(rmse1.detach().cpu())
            rmse_train2.append(rmse2.detach().cpu())

            self.summary_writer.add_scalar(
                tag='Train/Loss',
                scalar_value=np.array(loss_train).mean(),
                global_step=global_step
            )
            self.summary_writer.add_scalar(
                tag='Train/EPE',
                scalar_value=np.array(epe_train).mean(),
                global_step=global_step
            )
            ###### *modified* ######
            self.summary_writer.add_scalar(
                tag='Train/RMSE1',
                scalar_value=np.sqrt(np.array(rmse_train1).mean()),
                global_step=global_step
            )
            self.summary_writer.add_scalar(
                tag='Train/RMSE2',
                scalar_value=np.array(rmse_train2).mean(),
                global_step=global_step
            )


            train_progress.set_description(
                'Train Epoch {}: Loss: {:.5f} EPE: {:.5f} RMSE1: {:.5f} RMSE2: {:.5f}'.format(
                    epoch,
                    np.array(loss_train).mean(),
                    np.array(epe_train).mean(),
                    np.sqrt(np.array(rmse_train1).mean()), ###### *modified* ######
                    np.array(rmse_train2).mean() ###### *modified* ######
                )
            )

        self.lr_scheduler.step()
        save_checkpoint(self.model, self.args, epoch, 'train')
        
        logging.info('Train Epoch {}: Loss: {:.5f} EPE: {:.5f} RMSE1: {:.5f} RMSE2: {:.5f}'.format(
                    epoch,
                    np.array(loss_train).mean(),
                    np.array(epe_train).mean(),
                    np.sqrt(np.array(rmse_train1).mean()), ###### *modified* ######
                    np.array(rmse_train2).mean() ###### *modified* ######
                ))

    def val_test(self, epoch=0, mode='val'):
        self.model.eval()

        loss_run = []
        epe_run = []
        outlier_run = []
        acc3dRelax_run = []
        acc3dStrict_run = []
        rmse_run1 = [] ###### *modified* ######
        rmse_run2 = [] ###### *modified* ######

        if mode == 'val':
            run_dataloader = self.val_dataloader
            run_logstr = 'Val'
        else:
            run_dataloader = self.test_dataloader
            run_logstr = 'Test'
            self._load_weights(test_best=True)
        run_progress = tqdm(run_dataloader, ncols=150)
        for i, batch_data in enumerate(run_progress):
            global_step = epoch * len(run_dataloader) + i
            # batch_data = batch_data.to(self.device[0])
            ###### *modified* ######
            # pc1, pc2, color1, color2, flow = batch_data   
            pc1, pc2, flow = batch_data 
            mask = torch.ones_like(flow)[:,:,0]
            batch_data = {"sequence": [pc1, pc2], "ground_truth": [mask, flow]}   
            for key in batch_data.keys():
                batch_data[key] = [d.to(self.device[0]) for d in batch_data[key]]
            ###### *modified* ######

            with torch.no_grad():
                # est_flow, _ = self.model(batch_data["sequence"], num_iters=self.args.iters)  ###### *modified* ###### 32
                est_flow = self.model(batch_data["sequence"], num_iters=self.args.iters)  ###### *modified* ###### 32

            loss = sequence_loss(est_flow, batch_data, gamma=self.args.gamma)
            epe, acc3d_strict, acc3d_relax, outlier = compute_epe2(est_flow[-1], batch_data)
            rmse1, rmse2 = compute_rmse(est_flow[-1], batch_data) ###### *modified* ######
            loss_run.append(loss.cpu())
            epe_run.append(epe)
            outlier_run.append(outlier)
            acc3dRelax_run.append(acc3d_relax)
            acc3dStrict_run.append(acc3d_strict)
            rmse_run1.append(rmse1) ###### *modified* ######
            rmse_run2.append(rmse2) ###### *modified* ######

            if mode == 'val':
                self.summary_writer.add_scalar(
                    tag='Val/Loss',
                    scalar_value=np.array(loss_run).mean(),
                    global_step=global_step
                )
                self.summary_writer.add_scalar(
                    tag='Val/EPE',
                    scalar_value=np.array(epe_run).mean(),
                    global_step=global_step
                )
                self.summary_writer.add_scalar(
                    tag='Val/Outlier',
                    scalar_value=np.array(outlier_run).mean(),
                    global_step=global_step
                )
                self.summary_writer.add_scalar(
                    tag='Val/Acc3dRelax',
                    scalar_value=np.array(acc3dRelax_run).mean(),
                    global_step=global_step
                )
                self.summary_writer.add_scalar(
                    tag='Val/Acc3dStrict',
                    scalar_value=np.array(acc3dStrict_run).mean(),
                    global_step=global_step
                )
                ###### *modified* ######
                self.summary_writer.add_scalar(
                    tag='Val/RMSE1',
                    scalar_value=np.sqrt(np.array(rmse_run1).mean()),
                    global_step=global_step
                )
                self.summary_writer.add_scalar(
                    tag='Val/RMSE2',
                    scalar_value=np.array(rmse_run2).mean(),
                    global_step=global_step
                )

            run_progress.set_description(
                run_logstr +
                ' Epoch {}: Loss: {:.5f} EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f} RMSE1: {:.5f} RMSE2: {:.5f}'.format(
                    epoch,
                    np.array(loss_run).mean(),
                    np.array(epe_run).mean(),
                    np.array(outlier_run).mean(),
                    np.array(acc3dRelax_run).mean(),
                    np.array(acc3dStrict_run).mean(),
                    np.sqrt(np.array(rmse_run1).mean()),  ###### *modified* ######
                    np.array(rmse_run2).mean()  ###### *modified* ######
                )
            )

        if mode == 'val':
            if np.array(epe_run).mean() < self.best_val_epe:
                self.best_val_epe = np.array(epe_run).mean()
                save_checkpoint(self.model, self.args, epoch, 'val')
            logging.info(
                'Val Epoch {}: Loss: {:.5f} EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f} RMSE1: {:.5f} RMSE2: {:.5f}'.format(
                    epoch,
                    np.array(loss_run).mean(),
                    np.array(epe_run).mean(),
                    np.array(outlier_run).mean(),
                    np.array(acc3dRelax_run).mean(),
                    np.array(acc3dStrict_run).mean(),
                    np.sqrt(np.array(rmse_run1).mean()),  ###### *modified* ######
                    np.array(rmse_run2).mean()  ###### *modified* ######
                ))
            logging.info('Best EPE: {:.5f}'.format(self.best_val_epe))
        if mode == 'test':
            print('Test Result: EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f} RMSE1: {:.5f} RMSE2: {:.5f}'.format(
                np.array(epe_run).mean(),
                np.array(outlier_run).mean(),
                np.array(acc3dRelax_run).mean(),
                np.array(acc3dStrict_run).mean(),
                np.sqrt(np.array(rmse_run1).mean()),  ###### *modified* ######
                np.array(rmse_run2).mean()  ###### *modified* ######
            ))
            logging.info(
                'Test Result: EPE: {:.5f} Outlier: {:.5f} Acc3dRelax: {:.5f} Acc3dStrict: {:.5f} RMSE1: {:.5f} RMSE2: {:.5f}'.format(
                    np.array(epe_run).mean(),
                    np.array(outlier_run).mean(),
                    np.array(acc3dRelax_run).mean(),
                    np.array(acc3dStrict_run).mean(),
                    np.sqrt(np.array(rmse_run1).mean()),  ###### *modified* ######
                    np.array(rmse_run2).mean()  ###### *modified* ######
            ))


