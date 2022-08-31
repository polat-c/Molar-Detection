import torch
import torch.optim as optim
import os
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torch.sigmoid as sigmoid
import numpy as np
import sklearn.metrics as sk_metrics
import pandas as pd
import SimpleITK as sitk

import utils
from utils import cuda, visualize_image, load_images
import models
import dataset

# Vizualization
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

import segmentation_models_pytorch as smp


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    images_train=[],
                    images_eval=[], )

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, args):

        self.model = args.model
        self.n_class = args.n_class
        self.window_size = args.window_size
        self.res_size = args.res_size
        self.pretrain = args.pretrain

        self.save_step = args.save_step
        self.gather_step = args.gather_step
        self.viz_step = args.viz_step
        self.eval_step = args.eval_step
        self.evaluation = args.evaluation
        self.use_cuda = args.cuda and torch.cuda.is_available()

        self.max_iter = args.max_iter
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.num_workers = args.num_workers
        self.weight_decay = args.weight_decay

        self.exp_name = args.exp_name

        self.global_iter = 0
        self.global_iter_eval = 0

        self.train_loader, self.valid_loader = dataset.return_data(args)

        net = eval('models.' + self.model)

        if self.model == 'ResNetXIn':
            self.net = cuda(net(self.res_size, self.pretrain),
                        self.use_cuda)
        else:
            self.net = cuda(net(self.n_class, self.window_size),
                            self.use_cuda)

        # Optimizer
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(self.beta1, self.beta2))

        self.scheduler = ReduceLROnPlateau(optimizer=self.optim,
                                           factor=args.lrate_decay,
                                           patience=args.patience)

        # Load/prepare checkpoints
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.exp_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        # if self.ckpt_name is not None:
        #     self.load_checkpoint(self.ckpt_name)

        # Save Output
        self.output_dir = os.path.join(args.output_dir, self.exp_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Tensorboard
        self.gather = DataGather()
        self.writer = SummaryWriter(logdir="./logdir/" + self.exp_name)
        self.net.set_writer(self.writer)

    def train(self):
        self.net_mode(train=True)
        finish = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)

        while not finish:  # until max number of iterations is achieved
            # runs through all the batches in the dataset

            for idx, (x, target) in enumerate(self.train_loader):
                self.global_iter += 1
                pbar.update(1)

                # Forward pass
                # target = torch.flatten(target)
                # print(target.shape, target)
                x = Variable(cuda(x, self.use_cuda))
                target = Variable(cuda(target, self.use_cuda))

                out = self.net(x.float())

                loss = models.criterion(out, target)

                # Record training losses
                if self.global_iter % self.gather_step == 0:
                    # Accuracy
                    _, out_label = torch.max(out.data, 1)
                    metrics_train = self.report_metrics_binary(out_label.cpu().numpy(), target.cpu().numpy())


                    # Record training loss
                    self.writer.add_scalar("1/Loss - Cross Entropy [Training]",
                                           loss,
                                           self.global_iter)

                    # Record evaluation accuracy
                    self.writer.add_scalar("1/Loss - Accuracy [Training]",
                                           metrics_train['acc'],
                                           self.global_iter)
                    # Record evaluation balanced accuracy
                    self.writer.add_scalar("1/Loss - Bal. Accuracy [Training]",
                                           metrics_train['bal_acc'],
                                           self.global_iter)
                    # Record evaluation F1 Score
                    self.writer.add_scalar("1/Loss - F1-Score [Training]",
                                           metrics_train['f1_scr'],
                                           self.global_iter)
                    # Record evaluation recall
                    self.writer.add_scalar("1/Loss - Recall [Training]",
                                           metrics_train['recall'],
                                           self.global_iter)
                    # Record evaluation precision
                    self.writer.add_scalar("1/Loss - Precision [Training]",
                                           metrics_train['precision'],
                                           self.global_iter)

                # Evaluation
                if self.global_iter % self.eval_step == 0 and self.evaluation:

                    # Accuracy Full Test_set
                    metrics_test = self.evaluate_predictor()

                    # Record evaluation loss
                    self.writer.add_scalar("2/Loss - Cross Entropy [Evaluation]",
                                           metrics_test['loss'],
                                           self.global_iter)
                    # Record evaluation accuracy
                    self.writer.add_scalar("2/Loss - Accuracy [Evaluation]",
                                           metrics_test['acc'],
                                           self.global_iter)
                    # Record evaluation balanced accuracy
                    self.writer.add_scalar("2/Loss - Bal. Accuracy [Evaluation]",
                                           metrics_test['bal_acc'],
                                           self.global_iter)
                    # Record evaluation F1 Score
                    self.writer.add_scalar("2/Loss - F1-Score [Evaluation]",
                                           metrics_test['f1_scr'],
                                           self.global_iter)
                    # Record evaluation recall
                    self.writer.add_scalar("2/Loss - Recall [Evaluation]",
                                           metrics_test['recall'],
                                           self.global_iter)
                    #Record evaluation precision
                    self.writer.add_scalar("2/Loss - Precision [Evaluation]",
                                           metrics_test['precision'],
                                           self.global_iter)

                # Vizualization
                if self.global_iter % self.viz_step == 0:
                    self.gather.insert(iter=self.global_iter)
                    self.gather.insert(images_train=x.data)
                    # self.gather.insert(images_train=torch.sigmoid(x_recon).data)
                    # self.gather.insert(images_eval=x_test.data)
                    # self.gather.insert(images_eval=F.sigmoid(x_test_recon).data)
                    self.viz_reconstruction()
                    self.gather.flush()

                # Grad descent and backward pass
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Save checkpoits
                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter % 50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                # Check if finished
                if self.global_iter >= self.max_iter:
                    finish = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    def evaluate_predictor(self):
        self.net_mode(train=False)

        total_test_size = 0
        correct_test = 0
        loss_test = 0
        out_test_label_np = np.zeros(len(self.valid_loader.dataset))
        target_test_np = np.zeros(len(self.valid_loader.dataset))

        counter_images = 0
        for i, (x_test, target_test) in enumerate(self.valid_loader):
            # Forward
            # target_test = torch.flatten(target_test)
            x_test = Variable(cuda(x_test, self.use_cuda))
            out_test = self.net(x_test.float())
            target_test = Variable(cuda(target_test, self.use_cuda))

            # Loss
            loss_test += models.criterion(out_test, target_test)

            #join all batches from valid loader
            n_images_batch = target_test.size(0)
            _, out_test_label = torch.max(out_test.data, 1)
            print('HERE!!!',counter_images, n_images_batch,len(self.valid_loader.dataset), out_test_label_np.shape, out_test_label.cpu().numpy().shape)
            out_test_label_np[counter_images:counter_images+n_images_batch] = out_test_label.cpu().numpy()
            target_test_np[counter_images:counter_images+n_images_batch] = target_test.cpu().numpy()
            counter_images += n_images_batch


            print('out:', out_test_label, 'target:', target_test)


        # Sklearn metrics
        metrics_test = self.report_metrics_binary(target_test_np, out_test_label_np)
        # Loss
        metrics_test['loss'] = loss_test
        self.net_mode(train=True)

        return metrics_test

    def report_metrics_binary(self,target,out_label):
        metrics = {}
        metrics['acc'] = sk_metrics.accuracy_score(target, out_label)
        metrics['bal_acc'] = sk_metrics.balanced_accuracy_score(target, out_label)
        metrics['f1_scr'] = sk_metrics.f1_score(target, out_label, average='micro')
        metrics['recall'] = sk_metrics.recall_score(target, out_label,average='micro')
        metrics['precision'] = sk_metrics.precision_score(target, out_label, average='micro')

        return metrics


    def viz_reconstruction(self, num_samples=6):
        self.net_mode(train=False)

        def _visualize_image(image):
            fig, ax = plt.subplots()
            img = np.reshape(image, (self.window_size, self.window_size))
            ax.imshow(img)

            return fig

        if self.gather.data['images_train'][0].shape[0] < num_samples:
            num_samples = self.gather.data['images_train'][0].shape[0]

        for i in range(num_samples):
            x = self.gather.data['images_train'][0][i]
            tb_string_x = "1/Original Image -Training -Set" + ' |' + str(i)

            self.writer.add_figure(tb_string_x,
                                   _visualize_image(x.cpu().numpy()),
                                   self.global_iter,
                                   True)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net': self.net.state_dict(), }
        optim_states = {'optim': self.optim.state_dict(), }
        # win_states = {'recon':self.win_recon,
        #               'kld':self.win_kld,
        #               'mu':self.win_mu,
        #               'var':self.win_var,}
        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

###########################################################

class DetectionSolver(object):
    def __init__(self, args):

        self.model = args.model
        self.window_size = args.window_size
        self.res_size = args.res_size
        self.pretrain = args.pretrain
        self.predictor = args.predictor

        self.save_step = args.save_step
        self.gather_step = args.gather_step
        self.viz_step = args.viz_step
        self.eval_step = args.eval_step
        self.evaluation = args.evaluation
        self.use_cuda = args.cuda and torch.cuda.is_available()

        self.max_iter = args.max_iter
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.num_workers = args.num_workers
        self.weight_decay = args.weight_decay

        self.exp_name = args.exp_name

        self.global_iter = 0
        self.global_iter_eval = 0

        self.train_loader, self.valid_loader = dataset.return_detection_data(args)

        net = eval('models.' + self.model)

        if self.model == 'DetectionUnet':
            self.predictor = 'Classifier'
            self.net = cuda(net(), self.use_cuda)
        elif self.model == 'DetectionResNetXIn':
            self.net = cuda(net(self.res_size, self.pretrain),
                            self.use_cuda)
        else:
            self.net = cuda(net(self.window_size),
                            self.use_cuda)

        # Optimizer
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay,
                                betas=(self.beta1, self.beta2))

        self.scheduler = ReduceLROnPlateau(optimizer=self.optim,
                                           factor=args.lrate_decay,
                                           patience=args.patience)

        # Load/prepare checkpoints
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.exp_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        # Save Output
        self.output_dir = os.path.join(args.output_dir, self.exp_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # Tensorboard
        self.gather = DataGather()
        self.logdir = args.log_dir
        self.writer = SummaryWriter(logdir= self.logdir + "/" + self.exp_name)
        self.net.set_writer(self.writer)

    def train(self):
        print(self.net)
        self.net_mode(train=True)
        finish = False
        best_loss = 1e4

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)

        while not finish:  # until max number of iterations is achieved
            # runs through all the batches in the dataset

            for idx, (x, target) in enumerate(self.train_loader):
                self.global_iter += 1
                pbar.update(1)

                # Forward pass
                # target = torch.flatten(target)
                # print(target.shape, target)
                x = Variable(cuda(x, self.use_cuda))
                target = Variable(cuda(target, self.use_cuda))
                target = target.float()

                out = self.net(x.float())

                # loss
                if self.predictor == 'Regressor': # ROI center estimation
                    loss = models.detection_criterion_regressor(out, target)
                elif self.predictor == 'Classifier': # pixel classification
                    loss = models.detection_criterion_classifier(out, target)

                # Record training losses
                if self.global_iter % self.gather_step == 0:

                    if self.predictor == 'Classifier':
                        #IoU = self.IoU_score(out.cpu().numpy(), target.cpu().numpy())
                        IoU = self.IoU_score(out, target) ## no need to complicate things

                    # Record training loss
                    if self.predictor == 'Regressor':
                        self.writer.add_scalar("1/Loss - Mean Squared Error [Training]",
                                            loss,
                                            self.global_iter)
                    elif self.predictor == 'Classifier':
                        self.writer.add_scalar("1/Loss - DICE loss [Training]",
                                               loss,
                                               self.global_iter)
                        # Record IoU score
                        self.writer.add_scalar("1/Loss - IoU score [Training]",
                                               IoU,
                                               self.global_iter)

                # Evaluation
                if self.global_iter % self.eval_step == 0 and self.evaluation:

                    # Accuracy Full Test_set
                    loss_test = self.evaluate_predictor()

                    # Record evaluation loss
                    if self.predictor == 'Regressor':
                        self.writer.add_scalar("2/Loss - Mean Squared Error [Evaluation]",
                                            loss_test,
                                            self.global_iter)
                    elif self.predictor == 'Classifier':
                        self.writer.add_scalar("1/Loss - DICE loss [Evaluation]",
                                               loss_test['dice'],
                                               self.global_iter)
                        self.writer.add_scalar("1/Loss - IoU score [Evaluation]",
                                               loss_test['IoU'],
                                               self.global_iter)

                # Vizualization
                if self.global_iter % self.viz_step == 0:
                    self.gather.insert(iter=self.global_iter)
                    self.gather.insert(images_train=x.data)
                    # self.gather.insert(images_train=torch.sigmoid(x_recon).data)
                    # self.gather.insert(images_eval=x_test.data)
                    # self.gather.insert(images_eval=F.sigmoid(x_test_recon).data)
                    self.viz_reconstruction()
                    self.gather.flush()

                # Grad descent and backward pass
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Save checkpoits
                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                # Best checkpoints
                if self.global_iter % self.eval_step == 0  and self.evaluation:
                    if self.predictor == 'Regressor':
                        if loss_test < best_loss:
                            best_loss = loss_test
                            self.save_checkpoint('best')
                    elif self.predictor == 'Classifier':
                        if loss_test['dice'] < best_loss:
                            best_loss = loss_test['dice']
                            self.save_checkpoint('best')

                if self.global_iter % 50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                # Check if finished
                if self.global_iter >= self.max_iter:
                    finish = True
                    break

                torch.cuda.empty_cache()

        pbar.write("[Training Finished]")
        pbar.close()

    def IoU_score(self, target, output): # for predictor=''Classifier
        metric = smp.utils.metrics.IoU(threshold=0.5)
        score = metric.forward(target, output)
        return float(score)

    def evaluate_predictor(self):
        self.net_mode(train=False)
        loss_test = 0
        IoU_score_test = 0 # in case of classifier (segmentation)

        for i, (x_test, target_test) in enumerate(self.valid_loader):
            # Forward
            # target_test = torch.flatten(target_test)
            x_test = Variable(cuda(x_test, self.use_cuda))
            out_test = self.net(x_test.float())
            target_test = Variable(cuda(target_test, self.use_cuda))

            # Loss
            if self.predictor == 'Regressor':
                loss_test += float(models.detection_criterion_regressor(out_test, target_test))
            elif self.predictor == 'Classifier':
                loss_test += float(models.detection_criterion_classifier(out_test, target_test))
                IoU_score_test += self.IoU_score(target_test, out_test)

        self.net_mode(train=True)

        if self.predictor == 'Regressor':
            return loss_test
        elif self.predictor == 'Classifier':
            return {'dice':loss_test, 'IoU':IoU_score_test}

    def viz_reconstruction(self, num_samples=6):
        self.net_mode(train=False)

        def _visualize_image(image):
            fig, ax = plt.subplots()
            img = np.reshape(image, (self.window_size, self.window_size))
            ax.imshow(img)

            return fig

        if self.gather.data['images_train'][0].shape[0] < num_samples:
            num_samples = self.gather.data['images_train'][0].shape[0]

        for i in range(num_samples):
            x = self.gather.data['images_train'][0][i]
            tb_string_x = "1/Original Image -Training -Set" + ' |' + str(i)

            self.writer.add_figure(tb_string_x,
                                   _visualize_image(x.cpu().numpy()),
                                   self.global_iter,
                                   True)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net': self.net.state_dict(), }
        optim_states = {'optim': self.optim.state_dict(), }
        # win_states = {'recon':self.win_recon,
        #               'kld':self.win_kld,
        #               'mu':self.win_mu,
        #               'var':self.win_var,}
        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            #checkpoint = torch.load(file_path)
            checkpoint = torch.load(file_path, map_location=torch.device('cpu')) # for CPU-only machine
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    ###############################################

    def detect(self, dir_images, csv_file):
        '''
        Args:
            dir_images: directory of images to detect
        Returns:
        '''
        annotations = pd.read_csv(csv_file)
        images, image_ids = [],[]
        for name in annotations['File_name']:
            image = sitk.ReadImage(dir_images+'/'+name+'.dcm')
            images.append(image)
            image_ids.append(name)
        image_arrays = [sitk.GetArrayFromImage(i)[0, :, :] for i in images] # image2ndarray

        # implement transformations --> preprocessed_images
        preprocessed_images = []
        print('Preprocessing images ...')
        for image in image_arrays:
            image = utils.to_tensor(image)
            image = utils.normalize(image)
            image = utils.interpolate_(image, self.window_size)
            preprocessed_images.append(image)
        print('Preprocessing finished.')
        print('-----------------------')

        preprocessed_images = torch.stack(preprocessed_images)
        if self.predictor == 'Regressor':
            ROI_centers = self.net(preprocessed_images.float())
            xs = [float(roi[1]) for roi in ROI_centers]
            ys = [float(roi[0]) for roi in ROI_centers]
            predictions = pd.DataFrame({'ids':image_ids, 'x':xs, 'y':ys})
            predictions.to_csv(self.output_dir+'/predictions.csv')
        elif self.predictor == 'Classifier':
            print('Extracting masks ...')
            masks = self.net(preprocessed_images.float())
            print('Masks extracted.')
            # TODO add postprocessing for U-Net
            print('-----------------------')
            if not os.path.exists(self.output_dir+'/predicted_masks'):
                os.mkdir(self.output_dir+'/predicted_masks')
            for id, mask in enumerate(masks):
                mask_arr = mask.detach().numpy()
                utils.save_image_new(mask_arr, image_ids[id],self.output_dir+'/predicted_masks/', extension = '.gipl')

