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
from multiprocessing import Manager

class BigDataset(torch.utils.data.Dataset):
    #def __init__(self, data_paths, target_paths):
    def __init__(self, data_paths):
        manager = Manager()
        self.data_memmaps = [np.load(path, mmap_mode='r') for path in data_paths]
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
        #return index, torch.from_numpy(data)


#def CNNTrainer(input_size, output_size):
def main_worker(gpu, ngpus_per_node,
          file_path,
          input_size,
          output_size,
          val_split,
#          store_best=True,
#          save_best=False,
#          save_fn='best_model.ckpt',
#          save_path='checkpoints',
          verbose,
          args):
    print('entered the function')
    model = args.model
    network_module = importlib.import_module('cnns4qspr.se3cnn_v3.networks.{:s}.{:s}'.format(model, model))
    #self.input_size = input_size
    #self.output_size = output_size
    network = network_module.network(n_input=input_size,
                                           args=args)
    predictor_type = args.predictor_type
    if predictor_type == 'regressor':
        output_size = 1
    predictor = list_parser(args.predictor_layers)
    encoder = list_parser(args.encoder_layers)
    decoder = list_parser(args.decoder_layers)
    

    predictor_class = args.predictor
    if predictor_class == 'feedforward':
        predictor_type = FeedForward(type=type,
                                n_output=output_size,
                                input_size=network.output_size,
                                predictor=predictor)
        predictor_architecture = {'output_size': output_size,
                                  'input_size': network.output_size,
                                   'predictor': predictor_type.predictor}
    elif predictor_class == 'vae':
        predictor_type = VAE(latent_size=latent_size,
                        type=type,
                        n_output=output_size,
                        input_size=network.output_size,
                        encoder=encoder,
                        decoder=decoder,
                        predictor=predictor)
        predictor_architecture = {'output_size': output_size,
                                  'input_size': network.output_size,
                                  'latent_size': args.latent_size,
                                  'encoder': predictor.encoder,
                                   'decoder': predictor.decoder,
                                  'predictor': predictor_type.predictor}
        self.network.blocks[5].register_forward_hook(vae_hook)
    

    blocks = [block for block in network.blocks if block is not None]
    blocks.append(predictor_type)
    network.blocks = nn.Sequential(*[block for block in blocks if block is not None])
    history = {'train_loss': [],
               'val_loss': [],
               'train_predictor_loss': [],
               'val_predictor_loss': [],
               'train_vae_loss': [],
               'val_vae_loss': [],
               'train_acc': [],
               'val_acc': []}
#    current_state = {'state_dict': None,
#                     'optimizer': None,
#                     'epoch': None,
#                     'best_acc': np.nan,
#                      'best_loss': np.nan,
#                      'history': history,
#                      'input_size': input_size,
#                      'output_size': output_size,
#                      'predictor_architecture': args.predictor_architecture,
#                      'args': args}
#    best_state = {'state_dict': None,
#                  'optimizer': None,
#                  'epoch': None,
#                  'best_acc': 0,
#                  'best_loss': np.inf,
#                  'history': None,
#                  'input_size': self.input_size,
#                  'output_size': self.output_size,
#                  'predictor_architecture': args.predictor_architecture,
#                  'args': args}
    n_epochs = 0
    best_acc = 0
    best_loss = np.inf
    loaded = False

    # distributed training parameters
   # disdtributed = self.args.multiprocessing_distributed
   # self.gpu =self.args.gpu
   # self.dist_backend = self.args.dist_backend
   # self.dist_url = self.args.dist_url
   # self.rank = self.args.rank
   # self.world_size = self.args.world_size
   # self.pretrained = self.args.pretrained
   # self.worker = self.args.workers
   # self.batch_size = self.args.batch_size
      



#def save(self, mode='best', save_fn='best_model.ckpt', save_path='checkpoints'):
#    os.makedirs(save_path, exist_ok=True)
#    if os.path.splitext(save_fn)[1] == '':
#        save_fn += '.ckpt'
#    ckpt_save = {}
#    if mode == 'best':
#        for key in self.best_state.keys():
#            ckpt_save[key] = self.best_state[key]
#        torch.save(ckpt_save, os.path.join(save_path, save_fn))
#    elif mode == 'current':
#        for key in self.current_state.keys():
#            ckpt_save[key] = self.current_state[key]
#        torch.save(ckpt_save, os.path.join(save_path, save_fn))
# 
#def load(self, checkpoint_path):
#    loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#    for key in self.current_state.keys():
#        self.current_state[key] = loaded_checkpoint[key]
# 
#    self.args = self.current_state['args']
#    self.model = self.args.model
#    self.network_module = importlib.import_module('cnns4qspr.se3cnn_v3.networks.{:s}.{:s}'.format(self.model, self.model))
#    self.input_size = self.current_state['input_size']
#    self.output_size = self.current_state['output_size']
#    self.network = self.network_module.network(n_input=self.input_size,
#                                               args=self.args)
#    self.predictor_type = self.args.predictor_type
#    self.predictor_architecture = self.current_state['predictor_architecture']
#    self.build_predictor(predictor_class=self.args.predictor,
#                         type=self.predictor_type,
#                         input_size=self.network.output_size,
#                         output_size=self.output_size,
#                         latent_size=self.args.latent_size,
#                         predictor=self.args.predictor_layers,
#                         encoder=self.args.encoder_layers,
#                         decoder=self.args.decoder_layers)
#    self.blocks = [block for block in self.network.blocks if block is not None]
#    self.blocks.append(self.predictor)
#    self.network.blocks = nn.Sequential(*[block for block in self.blocks if block is not None])
# 
#    self.history = self.current_state['history']
#    self.n_epochs = self.current_state['epoch']
#    self.best_acc = self.current_state['best_acc']
#    self.best_loss = self.current_state['best_loss']
#    self.loaded = True

#def main_worker(self,
#          #data,
#          #targets,
#          file_path,
#          gpu,
#          ngpus_per_node, 
#          epochs='args',
#          val_split=0.2,
#          store_best=True,
#          save_best=False,
#          save_fn='best_model.ckpt',
#          save_path='checkpoints',
#          verbose=True):
    
    args.gpu = gpu
    print('args.gpu')
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
   # if args.pretrained:
   #     print("=> using pre-trained model '{}'".format(args.arch))
   #     model = models.__dict__[args.arch](pretrained=True)
   # else:
   #     print("=> creating model '{}'".format(args.arch))
   #     model = models.__dict__[args.arch]()

    model_final = network
    #print(model_final)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_final.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            print("workers size:", args.workers, flush=True)
            model_final = torch.nn.parallel.DistributedDataParallel(model_final, device_ids=[gpu])
        else:
            model_final.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model_final = torch.nn.parallel.DistributedDataParallel(model_final)
    elif args.gpu is not None:
        print('gpu used')
        torch.cuda.set_device(args.gpu)
        model_final = model_final.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model_final = torch.nn.DataParallel(model_final).cuda()

    # define loss function (criterion) and optimizer

    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)

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
    #if epochs == 'args':
    epochs = args.training_epochs

    ### Configure settings for model type
    if args.predictor_type == 'regressor':
        predictor_loss = nn.MSELoss()
        store_criterion = 'val_acc'
    elif args.predictor_type == 'classifier':
        predictor_loss = nn.CrossEntropyLoss()
        store_criterion = 'val_loss'
    
    train_criterion = predictor_loss.cuda(args.gpu)
    ### Load data and format for training
    get_file_names = []
    dir_name = os.path.join(file_path, 'data')
    for file_name in os.listdir(dir_name):
	    if fnmatch.fnmatch(file_name, 'feature*'):
                get_file_names.append(file_name)
    num_files = len(get_file_names)
    #data_paths = [os.path.join(file_path, f'feature{index}.npy') 
    #              for index in range(num_files)]
    data_paths = [os.path.join(file_path, f'data_50/feature{index}.npy') 
                  for index in range(1)]

    dataset = BigDataset(data_paths)

    #dataset = np.load('./data/feature1.npy')
    #dataset = np.concatenate((dataset_1, dataset_2))
    dataset_size = len(dataset)
    print("Dataset size:", dataset_size, flush=True)
    n_samples = dataset_size
    batch_size = args.batch_size
    n_val_samples = int(val_split * n_samples)
    n_train_samples = n_samples - n_val_samples

    #print(batch_size, flush=True) 
    if verbose:
        sys.stdout.write('\r'+'loading data...\n')
        sys.stdout.flush()
    train_indices = np.random.choice(np.arange(n_samples), \
                    size=n_train_samples, replace=False)
    val_indices = np.array(list(set(np.arange(n_samples, dtype=np.int64)) - set(train_indices)))
    #train_data = CustomIterableDataset(data[train_indices, :])
    #dataset = CustomIterableDataset(data)
    if args.distributed:
        #train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_sampler = torch.utils.data.distributed.DistributedSampler(torch.utils.data.Subset(dataset, train_indices))
        val_sampler = torch.utils.data.distributed.DistributedSampler(torch.utils.data.Subset(dataset, val_indices))
    else:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                               num_workers=1, sampler=train_sampler,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             num_workers=args.workers, sampler=val_sampler,
                                             pin_memory=True)

    if verbose:
        sys.stdout.write('\r'+'Data loaded...\n')
        sys.stdout.flush()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ### Set up network model and optimizer
    param_groups = get_param_groups(model_final, args)
    optimizer = Adam(param_groups, lr=args.initial_lr)
    optimizer.zero_grad()

#    if self.loaded:
#        self.network.load_state_dict(self.current_state['state_dict'])
#        self.optimizer.load_state_dict(self.current_state['optimizer'])

    if verbose:
        sys.stdout.write('\r'+'Network built...\n')
   

    #if args.evaluate:
    #    validate(val_loader, model, criterion, args)
    #return
    layers = []
    print(model_final)
    for n, p in model_final.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
    print(layers)
    #for epoch in range(args.start_epoch, args.epochs):
    fi = open("training.txt", "a")
    for epoch in range(epochs):
        print('check')
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
        optimizer, _ = lr_scheduler_exponential(optimizer,
                                                    epoch,
                                                    args.initial_lr,
                                                    args.lr_decay_start,
                                                    args.lr_decay_base,
                                                    verbose=False)
        train_loss, train_pred_loss, train_acc = train(train_loader, model_final, train_criterion, epoch, 
                                                       optimizer, args)

    # evaluate on validation set
   #     val_loss, val_pred_loss, val_acc = validate(val_loader, model_final, train_criterion, epoch)
   #     fi.write("%i, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f\n" %(epoch, np.mean(train_loss), 
   #     np.mean(train_pred_loss), np.mean(train_acc),
   #     np.mean(val_losses), np.mean(val_pred_losses), np.mean(val_acc)))
        fi.write("%i, %5.2f, %5.2f, %5.2f\n" %(epoch, np.mean(train_loss), 
        np.mean(train_pred_loss), np.mean(train_acc)))
        fi.flush()

    # remember best acc@1 and save checkpoint
    #is_best = acc1 > best_acc1
    #best_acc1 = max(acc1, best_acc1) 
    ### Train

def train(train_loader, model, criterion, epoch, optimizer, args):

    print('train_started')

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
 
    print(end, flush=True)
    for batch_idx, data_t in enumerate(train_loader):
        #print(data_t)
        data_time.update(time.time() - end) 
        print('Train {}_{}'.format(epoch, batch_idx), flush=True)
        #f.write("Train %s_ %s\n" % (epoch, batch_idx))
        #print(data_t[1].size())
       
            #data_t = data_t[1].to(device)
        inputs = data_t[1][:,:,:,:,:,0]
        targets = data_t[1][:,0,0,0,0,0]
 
        if args.gpu is not None:
            inputs = inputs.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(args.gpu, non_blocking=True)
        
        x = torch.autograd.Variable(inputs)
        y = torch.autograd.Variable(targets)
 
        # Forward and backward propagation
        if args.predictor == 'feedforward':
            predictions = model(x)
            pred_l = criterion(y, predictions)
            vae_l = torch.Tensor([np.nan])
            loss = pred_l
        elif args.predictor == 'vae':
            vae_out, mu, logvar, predictions = model(x)
            vae_in = torch.autograd.Variable(cnn_out)
            pred_l = criterion(y, predictions)
            vae_l = vae_loss(vae_in, vae_out, mu, logvar)
            loss = vae_l + pred_l
 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
 
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
        #print(torch.cuda.memory_summary())
  #print(end - start, flush=True)


#   args.history['train_loss'].append(np.mean(train_total_losses))
#   args.history['train_predictor_loss'].append(np.mean(train_predictor_losses))
#   args..history['train_vae_loss'].append(np.mean(train_vae_losses))
#   args.history['train_acc'].append(np.mean(train_accs))
   #n_epochs += 1
    return train_total_losses, train_predictor_losses, train_accs


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

       if args.gpu is not None:
           inputs = inputs.cuda(args.gpu, non_blocking=True)
       if torch.cuda.is_available():
           targets = targets.cuda(args.gpu, non_blocking=True)
       
       x = torch.autograd.Variable(inputs)
       y = torch.autograd.Variable(targets)

       # Forward prop
       if args.predictor == 'feedforward':
           predictions = model(x)
           pred_l = criterion(y, predictions)
           vae_l = torch.Tensor([np.nan])
           loss = pred_l
       elif args.predictor == 'vae':
           vae_out, mu, logvar, predictions = model(x)
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
       #print(torch.cuda.memory_summary())


#   args.history['val_loss'].append(np.mean(val_total_losses))
#   args.history['val_predictor_loss'].append(np.mean(val_predictor_losses))
#   args.history['val_vae_loss'].append(np.mean(val_vae_losses))
#   args.history['val_acc'].append(np.mean(val_accs))
#   args.current_state['state_dict'] = self.network.state_dict()
#   args.current_state['optimizer'] = self.optimizer.state_dict()
#   args.current_state['epoch'] = self.n_epochs

   return val_total_losses, val_predictor_losses, val_accs
#   if store_best:
#       if store_criterion == 'val_acc':
#           if np.mean(val_accs) > self.best_acc:
#               self.best_acc = np.mean(accs)
#               self.best_state['state_dict'] = copy.deepcopy(self.network.state_dict())
#               self.best_state['optimizer'] = copy.deepcopy(self.optimizer.state_dict())
#               self.best_state['epoch'] = self.n_epochs
#               self.best_state['best_acc'] = self.best_acc
#               self.best_state['history'] = copy.deepcopy(self.history)
#           else:
#               pass
#           if save_best:
#               self.save(mode='best', save_fn=save_fn, save_path=save_path)
#       elif store_criterion == 'val_loss':
#           if np.mean(val_total_losses) < self.best_loss:
#               self.best_loss = np.mean(val_total_losses)
#               self.best_state['state_dict'] = copy.deepcopy(self.network.state_dict())
#               self.best_state['optimizer'] = copy.deepcopy(self.optimizer.state_dict())
#               self.best_state['epoch'] = self.n_epochs
#               self.best_state['best_loss'] = self.best_loss
#               self.best_state['history'] = copy.deepcopy(self.history)
#           else:
#               pass
#           if save_best:
#               self.save(mode='best', save_fn=save_fn, save_path=save_path)


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


def vae_hook(self, module, input_, output):
    global cnn_out
    cnn_out = output
