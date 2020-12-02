"""
This module contains a Trainer class for quickly and easily building
VAE or FeedForward property predictors from the extracted structural protein
features.
"""
import time
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import numpy as np
import importlib
from se3cnn.util.get_param_groups import get_param_groups
from se3cnn.util.optimizers_L1L2 import Adam
from se3cnn.util.lr_schedulers import lr_scheduler_exponential
from cnns4qspr.util.pred_blocks import VAE, FeedForward
from cnns4qspr.util.losses import vae_loss, classifier_loss, regressor_loss
from cnns4qspr.util.tools import list_parser
from torch.utils.data import IterableDataset
from torch.utils.data.sampler import SubsetRandomSampler
from bisect import bisect
import os, psutil # used to monitor memory usage
import fnmatch
import torch.distributed as dist
import torch.multiprocessing as mp

from multiprocessing import Manager


class BigDataset(torch.utils.data.Dataset):
    #def __init__(self, data_paths, target_paths):
    def __init__(self, data_paths):
        manager = Manager()
        self.data_memmaps = manager([np.load(path, mmap_mode='r') for path in data_paths])
        #self.target_memmaps = [np.load(path, mmap_mode='r') for path in target_paths]
        self.start_indices = [0] * len(data_paths)
        self.data_count = 0
        for index, memmap in enumerate(self.data_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        data = self.data_memmaps[memmap_index][index_in_memmap]
        #target = self.target_memmaps[memmap_index][index_in_memmap]
        #return index, torch.from_numpy(data), torch.from_numpy(target)
        return index, torch.from_numpy(data)


class CNNTrainer():
    def __init__(self,
                 input_size,
                 output_size,
                 args):
        self.args = args
        self.model = self.args.model
        self.network_module = importlib.import_module('cnns4qspr.se3cnn_v3.networks.{:s}.{:s}'.format(self.model, self.model))
        self.input_size = input_size
        self.output_size = output_size
        self.network = self.network_module.network(n_input=self.input_size,
                                                   args=self.args)
        self.predictor_type = self.args.predictor_type
        if self.predictor_type == 'regressor':
            self.output_size = 1
        self.build_predictor(predictor_class=self.args.predictor,
                             type=self.predictor_type,
                             input_size=self.network.output_size,
                             output_size=self.output_size,
                             latent_size=self.args.latent_size,
                             predictor=self.args.predictor_layers,
                             encoder=self.args.encoder_layers,
                             decoder=self.args.decoder_layers)
        self.blocks = [block for block in self.network.blocks if block is not None]
        self.blocks.append(self.predictor)
        self.network.blocks = nn.Sequential(*[block for block in self.blocks if block is not None])
        self.history = {'train_loss': [],
                        'val_loss': [],
                        'train_predictor_loss': [],
                        'val_predictor_loss': [],
                        'train_vae_loss': [],
                        'val_vae_loss': [],
                        'train_acc': [],
                        'val_acc': []}
        self.current_state = {'state_dict': None,
                              'optimizer': None,
                              'epoch': None,
                              'best_acc': np.nan,
                              'best_loss': np.nan,
                              'history': self.history,
                              'input_size': self.input_size,
                              'output_size': self.output_size,
                              'predictor_architecture': self.predictor_architecture,
                              'args': self.args}
        self.best_state = {'state_dict': None,
                           'optimizer': None,
                           'epoch': None,
                           'best_acc': 0,
                           'best_loss': np.inf,
                           'history': None,
                           'input_size': self.input_size,
                           'output_size': self.output_size,
                           'predictor_architecture': self.predictor_architecture,
                           'args': self.args}
        self.n_epochs = 0
        self.best_acc = 0
        self.best_loss = np.inf
        self.loaded = False

        # distributed training parameters
        self.disdtributed = self.args.multiprocessing_distributed
        self.gpu =self.args.gpu
        self.dist_backend = self.args.dist_backend
        self.dist_url = self.args.dist_url
        self.rank = self.args.rank
        self.world_size = self.args.world_size
        self.pretrained = self.args.pretrained
        self.worker = self.args.workers
        self.batch_size = self.args.batch_size
      

    def build_predictor(self,
                        predictor_class,
                        type,
                        input_size,
                        output_size,
                        latent_size,
                        predictor,
                        encoder,
                        decoder):
        predictor = list_parser(predictor)
        encoder = list_parser(encoder)
        decoder = list_parser(decoder)
        if predictor_class == 'feedforward':
            self.predictor = FeedForward(type=type,
                                         n_output=output_size,
                                         input_size=input_size,
                                         predictor=predictor)
            self.predictor_architecture = {'output_size': self.output_size,
                                           'input_size': input_size,
                                           'predictor': self.predictor.predictor}
        elif predictor_class == 'vae':
            self.predictor = VAE(latent_size=latent_size,
                                 type=type,
                                 n_output=output_size,
                                 input_size=input_size,
                                 encoder=encoder,
                                 decoder=decoder,
                                 predictor=predictor)
            self.predictor_architecture = {'output_size': self.output_size,
                                           'input_size': input_size,
                                           'latent_size': self.args.latent_size,
                                           'encoder': self.predictor.encoder,
                                           'decoder': self.predictor.decoder,
                                           'predictor': self.predictor.predictor}
            self.network.blocks[5].register_forward_hook(self.vae_hook)

    def vae_hook(self, module, input_, output):
        global cnn_out
        cnn_out = output

    def save(self, mode='best', save_fn='best_model.ckpt', save_path='checkpoints'):
        os.makedirs(save_path, exist_ok=True)
        if os.path.splitext(save_fn)[1] == '':
            save_fn += '.ckpt'
        ckpt_save = {}
        if mode == 'best':
            for key in self.best_state.keys():
                ckpt_save[key] = self.best_state[key]
            torch.save(ckpt_save, os.path.join(save_path, save_fn))
        elif mode == 'current':
            for key in self.current_state.keys():
                ckpt_save[key] = self.current_state[key]
            torch.save(ckpt_save, os.path.join(save_path, save_fn))

    def load(self, checkpoint_path):
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        for key in self.current_state.keys():
            self.current_state[key] = loaded_checkpoint[key]

        self.args = self.current_state['args']
        self.model = self.args.model
        self.network_module = importlib.import_module('cnns4qspr.se3cnn_v3.networks.{:s}.{:s}'.format(self.model, self.model))
        self.input_size = self.current_state['input_size']
        self.output_size = self.current_state['output_size']
        self.network = self.network_module.network(n_input=self.input_size,
                                                   args=self.args)
        self.predictor_type = self.args.predictor_type
        self.predictor_architecture = self.current_state['predictor_architecture']
        self.build_predictor(predictor_class=self.args.predictor,
                             type=self.predictor_type,
                             input_size=self.network.output_size,
                             output_size=self.output_size,
                             latent_size=self.args.latent_size,
                             predictor=self.args.predictor_layers,
                             encoder=self.args.encoder_layers,
                             decoder=self.args.decoder_layers)
        self.blocks = [block for block in self.network.blocks if block is not None]
        self.blocks.append(self.predictor)
        self.network.blocks = nn.Sequential(*[block for block in self.blocks if block is not None])

        self.history = self.current_state['history']
        self.n_epochs = self.current_state['epoch']
        self.best_acc = self.current_state['best_acc']
        self.best_loss = self.current_state['best_loss']
        self.loaded = True

    def main_worker(self,
              #data,
              #targets,
              gpu,
              ngpus_per_node, 
              file_path,
              epochs='args',
              val_split=0.2,
              store_best=True,
              save_best=False,
              save_fn='best_model.ckpt',
              save_path='checkpoints',
              verbose=True):
        
        self.gpu = gpu
        print('training')
        if self.gpu is not None:
            print("Use GPU: {} for training".format(self.gpu))
    
        if self.distributed:
            if self.dist_url == "env://" and self.rank == -1:
                self.rank = int(os.environ["RANK"])
            if args.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.rank = self.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
        # create model
       # if args.pretrained:
       #     print("=> using pre-trained model '{}'".format(args.arch))
       #     model = models.__dict__[args.arch](pretrained=True)
       # else:
       #     print("=> creating model '{}'".format(args.arch))
       #     model = models.__dict__[args.arch]()
    
        model = self.network
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        elif self.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.batch_size = int(self.args.batch_size / ngpus_per_node)
                self.workers = int((self.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif self.gpu is not None:
            torch.cuda.set_device(self.gpu)
            model = model.cuda(self.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
    
        # define loss function (criterion) and optimizer
    
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    
        # optionally resume from a checkpoint
        #if args.resume:
        #    if os.path.isfile(args.resume):
        #        print("=> loading checkpoint '{}'".format(args.resume))
        #        if args.gpu is None:
        #            checkpoint = torch.load(args.resume)
        #        else:
        #            # Map model to be loaded to specified single gpu.
        #            loc = 'cuda:{}'.format(args.gpu)
        #            checkpoint = torch.load(args.resume, map_location=loc)
        #        args.start_epoch = checkpoint['epoch']
        #        best_acc1 = checkpoint['best_acc1']
        #        if args.gpu is not None:
        #            # best_acc1 may be from a checkpoint from a different GPU
        #            best_acc1 = best_acc1.to(args.gpu)
        #        model.load_state_dict(checkpoint['state_dict'])
        #        optimizer.load_state_dict(checkpoint['optimizer'])
        #        print("=> loaded checkpoint '{}' (epoch {})"
        #              .format(args.resume, checkpoint['epoch']))
        #    else:
        #        print("=> no checkpoint found at '{}'".format(args.resume))
    
        torch.backends.cudnn.benchmark = True
        use_gpu = torch.cuda.is_available()
        if epochs == 'args':
            epochs = self.args.training_epochs

        ### Configure settings for model type
        if self.predictor_type == 'classifier':
            predictor_loss = nn.MSELoss()
            store_criterion = 'val_acc'
        elif self.predictor_type == 'regressor':
            predictor_loss = nn.CrossEntropyLoss()
            store_criterion = 'val_loss'
        
        train_criterion = predictor_loss.cuda(self.gpu)
        ### Load data and format for training
        get_file_names = []
        dir_name = os.path.join(file_path, 'data')
        for file_name in os.listdir(dir_name):
    	    if fnmatch.fnmatch(file_name, 'feature*'):
                get_file_names.append(file_name)
        num_files = len(get_file_names)
        #data_paths = [os.path.join(file_path, f'data/feature{index}.npy') 
        #              for index in range(num_files)]
        data_paths = [os.path.join(file_path, f'data/feature{index}.npy') 
                      for index in range(2)]

        dataset = BigDataset(data_paths)


        dataset_size = len(dataset)
        print("Dataset size:", dataset_size, flush=True)
        self.n_samples = dataset_size
        self.batch_size = self.args.batch_size
        self.n_val_samples = int(val_split*self.n_samples)
        self.n_train_samples = self.n_samples - self.n_val_samples

        
        if verbose:
            sys.stdout.write('\r'+'loading data...\n')
            sys.stdout.flush()
        train_indices = np.random.choice(np.arange(self.n_samples), \
                        size=self.n_train_samples, replace=False)
        val_indices = np.array(list(set(np.arange(self.n_samples, dtype=np.int64)) - set(train_indices)))
        #train_data = CustomIterableDataset(data[train_indices, :])
        #dataset = CustomIterableDataset(data)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(torch.utils.data.Subset(dataset, train_indices))
            val_sampler = torch.utils.data.distributed.DistributedSampler(torch.utils.data.Subset(dataset, val_indices))
        else:
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size,
                                                   num_workers=self.workers, sampler=train_sampler,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size,
                                                 num_workers=self.workers, sampler=val_sampler,
                                                 pin_memory=True)

        if verbose:
            sys.stdout.write('\r'+'Data loaded...\n')
            sys.stdout.flush()

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ### Set up network model and optimizer
        self.param_groups = get_param_groups(model, self.args)
        self.optimizer = Adam(self.param_groups, lr=self.args.initial_lr)
        self.optimizer.zero_grad()

        if self.loaded:
            self.network.load_state_dict(self.current_state['state_dict'])
            self.optimizer.load_state_dict(self.current_state['optimizer'])

        if verbose:
            sys.stdout.write('\r'+'Network built...\n')
       

        #if args.evaluate:
        #    validate(val_loader, model, criterion, args)
        #return

        #for epoch in range(args.start_epoch, args.epochs):
        fi = open("training.txt", "a")
        for epoch in range(epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            #adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
            train_loss, train_pred_loss, train_acc = train(train_loader, model, train_criterion, epoch)

        # evaluate on validation set
            val_loss, val_pred_loss, val_acc = validate(val_loader, model, train_criterion, epoch)
            fi.write("%i, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f\n" %(epoch, np.mean(train_loss), 
            np.mean(train_pred_loss), np.mean(train_acc),
            np.mean(val_losses), np.mean(val_pred_losses), np.mean(val_acc)))
            fi.flush()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1) 
        ### Train

    def train(train_loader, model, criterion, epoch):
       self.optimizer, _ = lr_scheduler_exponential(self.optimizer,
                                                    epoch+self.n_epochs,
                                                    self.args.initial_lr,
                                                    self.args.lr_decay_start,
                                                    self.args.lr_decay_base,
                                                    verbose=False)

       ### Batch Loop
       model.train()
       train_total_losses = []
       train_vae_losses = []
       train_predictor_losses = []
       train_accs = []
       batch_time = AverageMeter('Time', ':6.3f')
       data_time = AverageMeter('Data', ':6.3f')
       losses = AverageMeter('Loss', ':8.5f')
       accu = AverageMeter('Accuracy', ':6.2f')
       #top5 = AverageMeter('Acc@5', ':6.2f')
       progress = ProgressMeter(
           len(train_loader),
           [batch_time, data_time, losses, accu],
           prefix="Epoch: [{}]".format(epoch))

       end = time.time()
       for batch_idx, data_t in enumerate(train_loader):
           data_time.update(time.time() - end) 
           print('Train {}_{}'.format(epoch, batch_idx), flush=True)
           #f.write("Train %s_ %s\n" % (epoch, batch_idx))
           #print(data_t[1].size())
          
               #data_t = data_t[1].to(device)
           inputs = data_t[1][:,:,:,:,:,0]
           targets = data_t[1][:,0,0,0,0,0]

           if self.gpu is not None:
               inputs = inputs.cuda(self.gpu, non_blocking=True)
           if torch.cuda.is_available():
               targets = targets.cuda(self.gpu, non_blocking=True)
           
           x = torch.autograd.Variable(inputs)
           y = torch.autograd.Variable(targets)

           # Forward and backward propagation
           if self.args.predictor == 'feedforward':
               predictions = model(x)
               pred_l = criterion(y, predictions)
               vae_l = torch.Tensor([np.nan])
               loss = pred_l
           elif self.args.predictor == 'vae':
               vae_out, mu, logvar, predictions = model(x)
               vae_in = torch.autograd.Variable(cnn_out)
               pred_l = criterion(y, predictions)
               vae_l = vae_loss(vae_in, vae_out, mu, logvar)
               loss = vae_l + pred_l

           loss.backward()
           self.optimizer.step()
           self.optimizer.zero_grad()

           _, argmax = torch.max(predictions, 1)
           acc = (argmax.squeeze() == targets).float().mean()

           train_total_losses.append(loss.item())
           #print(train_total_losses, flush=True)
           #f.write("%s" % (repr(total_losses)))
           train_predictor_losses.append(pred_l.item())
           train_vae_losses.append(vae_l.item())
           train_accs.append(acc.item())
           batch_time.update(time.time() - end)
           losses.update(loss.item())
           accu.update(acc.item())
           end = time.time()
           progress.display(batch_idx)
           #print(end - start, flush=True)


       self.history['train_loss'].append(np.mean(train_total_losses))
       self.history['train_predictor_loss'].append(np.mean(train_predictor_losses))
       self.history['train_vae_loss'].append(np.mean(train_vae_losses))
       self.history['train_acc'].append(np.mean(train_accs))
       self.n_epochs += 1


    def validate(val_loader, model, criterion, epoch):
            ### Validation Loop
       model.eval()
       val_total_losses = []
       val_vae_losses = []
       val_predictor_losses = []
       val_accs = []

       for batch_idx, data_v in enumerate(val_loader):
           data_time.update(time.time() - end)
           print('Val {}_{}'.format(epoch, batch_idx), flush=True)
           #f.write("'Val %s_%s\n" % (epoch, batch_idx))
           inputs = data_v[1][:,:,:,:,:,0]
           targets = data_v[1][:,0,0,0,0,0]

           if self.gpu is not None:
               inputs = inputs.cuda(self.gpu, non_blocking=True)
           if torch.cuda.is_available():
               targets = targets.cuda(self.gpu, non_blocking=True)
           
           x = torch.autograd.Variable(inputs)
           y = torch.autograd.Variable(targets)

           # Forward prop
           if self.args.predictor == 'feedforward':
               predictions = model(x)
               pred_l = criterion(y, predictions)
               vae_l = torch.Tensor([np.nan])
               loss = pred_l
           elif self.args.predictor == 'vae':
               vae_out, mu, logvar, predictions = self.model(x)
               vae_in = torch.autograd.Variable(cnn_out)
               pred_l = criterion(y, predictions)
               vae_l = vae_loss(vae_in, vae_out, mu, logvar)
               loss = vae_l + pred_l

           _, argmax = torch.max(predictions, 1)
           acc = (argmax.squeeze() == targets).float().mean()

           val_total_losses.append(loss.item())
           #print(val_total_losses, flush=True)
           #f.write("%s" % (repr(total_losses)))
           val_predictor_losses.append(pred_l.item())
           val_vae_losses.append(vae_l.item())
           val_accs.append(acc.item())
           batch_time.update(time.time() - end)
           losses.update(loss.item())
           accu.update(acc.item())
           end = time.time()
           progress.display(batch_idx)
           #print(end - start, flush=True)



       self.history['val_loss'].append(np.mean(val_total_losses))
       self.history['val_predictor_loss'].append(np.mean(val_predictor_losses))
       self.history['val_vae_loss'].append(np.mean(val_vae_losses))
       self.history['val_acc'].append(np.mean(val_accs))
       self.current_state['state_dict'] = self.network.state_dict()
       self.current_state['optimizer'] = self.optimizer.state_dict()
       self.current_state['epoch'] = self.n_epochs

       if store_best:
           if store_criterion == 'val_acc':
               if np.mean(val_accs) > self.best_acc:
                   self.best_acc = np.mean(accs)
                   self.best_state['state_dict'] = copy.deepcopy(self.network.state_dict())
                   self.best_state['optimizer'] = copy.deepcopy(self.optimizer.state_dict())
                   self.best_state['epoch'] = self.n_epochs
                   self.best_state['best_acc'] = self.best_acc
                   self.best_state['history'] = copy.deepcopy(self.history)
               else:
                   pass
               if save_best:
                   self.save(mode='best', save_fn=save_fn, save_path=save_path)
           elif store_criterion == 'val_loss':
               if np.mean(val_total_losses) < self.best_loss:
                   self.best_loss = np.mean(val_total_losses)
                   self.best_state['state_dict'] = copy.deepcopy(self.network.state_dict())
                   self.best_state['optimizer'] = copy.deepcopy(self.optimizer.state_dict())
                   self.best_state['epoch'] = self.n_epochs
                   self.best_state['best_loss'] = self.best_loss
                   self.best_state['history'] = copy.deepcopy(self.history)
               else:
                   pass
               if save_best:
                   self.save(mode='best', save_fn=save_fn, save_path=save_path)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    


def run(cur_dir, n_channels, args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'

    ngpus_per_node = torch.cuda.device_count()
    cnn = CNNTrainer(input_size=n_channels, output_size=3, args=args)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        print('started')
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(cnn.main_worker, nprocs=ngpus_per_node, args=(cur_dir, ngpus_per_node))
    else:
        # Simply call main_worker function
        cnn.main_worker(args.gpu, cur_dir, ngpus_per_node, args)
    
