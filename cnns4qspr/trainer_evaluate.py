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


class BigDataset(torch.utils.data.Dataset):
    #def __init__(self, data_paths, target_paths):
    def __init__(self, data_paths):
        print(data_paths)
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
        #print(self.network)      
        #print(self.network.load_state_dict(self.current_state['state_dict']))
        self.history = self.current_state['history']
        self.n_epochs = self.current_state['epoch']
        self.best_acc = self.current_state['best_acc']
        self.best_loss = self.current_state['best_loss']
        self.loaded = True

    def load_data(self,
                  file_path,
                  verbose=True):
        
        get_file_names = []
        #dir_name = os.path.join(file_path, 'data')
        for file_name in os.listdir(file_path):
    	    if fnmatch.fnmatch(file_name, 'feature*'):
                get_file_names.append(file_name)
        num_files = len(get_file_names)
        #print(file_path, num_files)
        #target_paths = [f's{index}.npy' for index in range(10)]
        
        data_paths = [os.path.join(file_path, f'feature{index}.npy')
                      for index in range(num_files)]
        #data_paths = [os.path.join(file_path, f'feature{index}.npy') 
        #              for index in range(3)]
        #print(data_paths)
        #dataset = BigDataset(data_paths, target_paths)
        dataset_loaded = BigDataset(data_paths)
        if verbose:
            print('data loaded')
        return dataset_loaded

    def count_parameters(self):
 
         
        if self.loaded:
            self.network.load_state_dict(self.current_state['state_dict'])
  

        model_parameters = filter(lambda p: p.requires_grad, self.network.parameters())
        params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        #params = sum([np.prod(p.size()) for p in model_parameters])
        print('total parameters in the model = {}'.format(params))
        #for p in self.network.parameters():
        #    print(p.size())

    def train(self,
              #data,
              #targets,
              file_path, 
              epochs='args',
              val_split=0.2,
              store_best=True,
              save_best=False,
              save_fn='best_model.ckpt',
              save_path='checkpoints',
              verbose=True):
        torch.backends.cudnn.benchmark = True
        use_gpu = torch.cuda.is_available()
        if epochs == 'args':
            epochs = self.args.training_epochs

        ### Configure settings for model type
        if self.predictor_type == 'classifier':
            predictor_loss = classifier_loss
            store_criterion = 'val_acc'
        elif self.predictor_type == 'regressor':
            predictor_loss = regressor_loss
            store_criterion = 'val_loss'

        ### Load data and format for training
        train_dataset = self.load_data(file_path + 'train/')
        val_dataset = self.load_data(file_path + 'validation/')

        dataset_size = len(train_dataset)
        print("training dataset size:", dataset_size, flush=True)
        #data = np.load(file_name, mmap_mode = 'r')['arr_0']
        self.n_samples = dataset_size
        self.batch_size = self.args.batch_size
        #self.n_val_samples = int(val_split*self.n_samples)
        #self.n_train_samples = self.n_samples - self.n_val_samples

        #targets = targets.squeeze()
        #data = np.expand_dims(data, 5)
        #data[:,0,0,0,0,0] = targets
        
        if verbose:
            sys.stdout.write('\r'+'loading data...\n')
            sys.stdout.flush()
        train_indices = np.arange(dataset_size)
        #train_indices = np.random.choice(np.arange(self.n_samples), \
        #                size=self.n_train_samples, replace=False)
        #val_indices = np.array(list(set(np.arange(self.n_samples, dtype=np.int64)) - set(train_indices)))
        #train_data = CustomIterableDataset(data[train_indices, :])
        #dataset = CustomIterableDataset(data)
        
        #with open ("train_indices.txt", 'w') as train_file:
        #    for item in train_indices:
        #        train_file.write("%s\n" %item)
        train_sampler = SubsetRandomSampler(train_indices)
        #val_sampler = SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size,
                                                   num_workers=5, sampler=train_sampler,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size,
                                                 num_workers=1, 
                                                 pin_memory=True)

        if verbose:
            sys.stdout.write('\r'+'val Data deleted...\n')
            sys.stdout.flush()
        #if verbose:
            sys.stdout.write('\r'+'Data loaded...\n')
            sys.stdout.flush()

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ### Set up network model and optimizer
        if use_gpu:
            self.network.cuda()

            #if torch.cuda.device_count() > 1:
            #    print("Let's use", torch.cuda.device_count(), "GPUs!")
            #    self.network = nn.DataParallel(self.network)

        #self.network.to(device)
        self.param_groups = get_param_groups(self.network, self.args)
        self.optimizer = Adam(self.param_groups, lr=self.args.initial_lr)
        self.optimizer.zero_grad()

        if self.loaded:
            self.network.load_state_dict(self.current_state['state_dict'])
            self.optimizer.load_state_dict(self.current_state['optimizer'])

        if verbose:
            sys.stdout.write('\r'+'Network built...\n')
        fi = open("training.txt", "a")
        ### Train
        for epoch in range(epochs):
            self.optimizer, _ = lr_scheduler_exponential(self.optimizer,
                                                         epoch+self.n_epochs,
                                                         self.args.initial_lr,
                                                         self.args.lr_decay_start,
                                                         self.args.lr_decay_base,
                                                         verbose=False)

            ### Batch Loop
            self.network.train()
            train_total_losses = []
            train_vae_losses = []
            train_predictor_losses = []
            train_accs = []
            batch_time_t = AverageMeter('Time', ':6.3f')
            data_time_t = AverageMeter('Data', ':6.3f')
            losses_t = AverageMeter('Loss', ':8.5f')
            accu_t = AverageMeter('Accuracy', ':6.2f')
            #top5 = AverageMeter('Acc@5', ':6.2f')
            progress_train = ProgressMeter(
                len(train_loader),
                [batch_time_t, data_time_t, losses_t, accu_t],
                prefix="Epoch: [{}]".format(epoch))

            end = time.time()
            for batch_idx, data_t in enumerate(train_loader):
                data_time_t.update(time.time() - end) 
                print('Train {}_{}'.format(epoch, batch_idx), flush=True)
                #f.write("Train %s_ %s\n" % (epoch, batch_idx))
                #print(data_t[1].size())
               
                if use_gpu:
                    data_t = data_t[1].cuda()
                    #data_t = data_t[1].to(device)
                inputs = data_t[:,:,:,:,:,0]
                targets = data_t[:,0,0,0,0,0]

                x = torch.autograd.Variable(inputs)
                y = torch.autograd.Variable(targets)

                # Forward and backward propagation
                if self.args.predictor == 'feedforward':
                    predictions = self.network(x)
                    pred_l = predictor_loss(y, predictions)
                    vae_l = torch.Tensor([np.nan])
                    loss = torch.sqrt(pred_l)
                elif self.args.predictor == 'vae':
                    vae_out, mu, logvar, predictions = self.network(x)
                    vae_in = torch.autograd.Variable(cnn_out)
                    pred_l = predictor_loss(y, predictions)
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
                batch_time_t.update(time.time() - end)
                losses_t.update(loss.item())
                accu_t.update(acc.item())
                end = time.time()
                progress_train.display(batch_idx)
                #print(end - start, flush=True)


            self.history['train_loss'].append(np.mean(train_total_losses))
            self.history['train_predictor_loss'].append(np.mean(train_predictor_losses))
            self.history['train_vae_loss'].append(np.mean(train_vae_losses))
            self.history['train_acc'].append(np.mean(train_accs))
            self.n_epochs += 1

            ### Validation Loop
            self.network.eval()
            val_total_losses = []
            val_vae_losses = []
            val_predictor_losses = []
            val_accs = []
            batch_time_v = AverageMeter('Time', ':6.3f')
            data_time_v = AverageMeter('Data', ':6.3f')
            losses_v = AverageMeter('Loss', ':8.5f')
            accu_v = AverageMeter('Accuracy', ':6.2f')
            progress_val = ProgressMeter(
                len(val_loader),
                [batch_time_v, data_time_v, losses_v, accu_v],
                prefix="Epoch: [{}]".format(epoch))

            for batch_idx, data_v in enumerate(val_loader):
                data_time_v.update(time.time() - end)
                print('Val {}_{}'.format(epoch, batch_idx), flush=True)
                #f.write("'Val %s_%s\n" % (epoch, batch_idx))
                if use_gpu:
                    data_v = data_v[1].cuda()
                inputs = data_v[:,:,:,:,:,0]
                targets = data_v[:,0,0,0,0,0]

                x = torch.autograd.Variable(inputs)
                y = torch.autograd.Variable(targets)

                # Forward prop
                if self.args.predictor == 'feedforward':
                    predictions = self.network(x)
                    pred_l = predictor_loss(y, predictions)
                    vae_l = torch.Tensor([np.nan])
                    loss = torch.sqrt(pred_l)
                elif self.args.predictor == 'vae':
                    vae_out, mu, logvar, predictions = self.network(x)
                    vae_in = torch.autograd.Variable(cnn_out)
                    pred_l = predictor_loss(y, predictions)
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
                batch_time_v.update(time.time() - end)
                losses_v.update(loss.item())
                accu_v.update(acc.item())
                end = time.time()
                progress_val.display(batch_idx)
                #print(end - start, flush=True)



            self.history['val_loss'].append(np.mean(val_total_losses))
            self.history['val_predictor_loss'].append(np.mean(val_predictor_losses))
            self.history['val_vae_loss'].append(np.mean(val_vae_losses))
            self.history['val_acc'].append(np.mean(val_accs))
            fi.write("%i, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f\n" %(epoch, np.mean(train_total_losses), np.mean(train_predictor_losses), np.mean(train_accs),
                 np.mean(val_total_losses), np.mean(val_predictor_losses), np.mean(val_accs)))
            fi.flush()
            self.current_state['state_dict'] = self.network.state_dict()
            self.current_state['optimizer'] = self.optimizer.state_dict()
            self.current_state['epoch'] = self.n_epochs
        
            if self.n_epochs % 10 == 0:
                self.save(mode='current', save_fn='model_' + str(self.n_epochs) + '.ckpt', save_path='saved_models')
            

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



    def predict(self, file_path, predict_type=None,
                 train_indices_file=None):

        if self.loaded:
            self.network.load_state_dict(self.current_state['state_dict'])
            #self.optimizer.load_state_dict(self.current_state['optimizer'])
        
#        get_file_names = []
#        for file_name in os.listdir(file_path):
#            if fnmatch.fnmatch(file_name, 'feature*'):
#                get_file_names.append(file_name)
#        num_files = len(get_file_names)
#
#        data_paths = [os.path.join(file_path, f'feature{index}.npy')
#                      for index in range(num_files)]
#        dataset = BigDataset(data_paths)
        dataset = self.load_data(file_path)
        n_samples = len(dataset)
        
        all_indices = np.arange(n_samples)
        if train_indices_file is not None:
            train_indices = np.sort(np.loadtxt(train_indices_file).astype(int))
            val_indices = np.array(list(set(np.arange(n_samples, dtype=np.int64)) - set(train_indices)))
            
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)


            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size,
                                                   num_workers=1, shuffle=False, 
                                                   pin_memory=True)
        

            tar, pred = self.predict_dataset(train_loader)
            self.write_result('train.txt', tar, pred)

            
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.batch_size,
                                                   num_workers=1, shuffle=False, 
                                                   pin_memory=True)
        

            tar, pred = self.predict_dataset(val_loader)
            self.write_result('validation.txt', tar, pred)

        else:
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size,
                                                   num_workers=1, shuffle=False,
                                                   pin_memory=True)
        

            tar, pred = self.predict_dataset(test_loader)
            #print(tar, pred)
            self.write_result(predict_type +'.txt', tar, pred)

    def predict_dataset(self,
                        dataloader):

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.network.cuda()
        self.network.eval()
        targets_all = []
        predictions_all = []
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        progress = ProgressMeter(len(dataloader),
                [batch_time, data_time])
        end = time.time()
        for batch_idx, data in enumerate(dataloader):
            data_time.update(time.time() - end) 
            print("predict %s\n" % (batch_idx), flush=True)
            if use_gpu:
                data = data[1].cuda()
            else:
                data = data[1]
            inputs = data[:,:,:,:,:,0]
            targets = data[:,0,0,0,0,0]

            x = torch.autograd.Variable(inputs)
            y = torch.autograd.Variable(targets)

           #evaluate
            if self.args.predictor == 'feedforward':
                predictions = self.network(x)
            elif self.args.predictor == 'vae':
                pass
            batch_time.update(time.time() - end)
            end = time.time()
            progress.display(batch_idx)
            predictions = predictions.data.cpu().numpy().flatten()
            targets = targets.data.cpu().numpy().flatten()
            targets_all.append(targets)
            predictions_all.append(predictions)
            #if batch_idx > 5:
            #    break
        return targets_all, predictions_all

    def write_result(self, file_name, target, predict):
   
        target = np.hstack(np.array(target, dtype=object).flatten())
        predict = np.hstack(np.array(predict, dtype=object).flatten())
        #target = np.concatenate(np.array(target, dtype=object).flatten(), axis = 0)
        #predict = np.concatenate(np.array(predict, dtype=object).flatten(), axis=0) 
        with open(file_name, 'w') as f:
            for i in range(len(target)):
                s1 = str(target[i]) + '\t' + str(predict[i]) + '\n'
                f.write(s1)
        f.close()


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
