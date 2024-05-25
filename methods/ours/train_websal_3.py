"""Main entry point for doing all pruning-related stuff."""
from __future__ import division, print_function
import os
import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(srcFolder)
import re

from logging import logMultiprocessing
from torch.optim import SGD, Adam, lr_scheduler
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from torchvision import *
from models.erfnet_RA_parallel_1layer import Net as erfNet_RA_parallel_1layer
from models.erfnet_RA_parallel_2layer025 import Net as erfNet_RA_parallel_2layer025
from methods.ours.prune import SparsePruner
from methods.utils.losses import TVdist
from methods.utils.training_utils import *
from methods.utils.util import *
# import wandb
writer = SummaryWriter(log_dir='pruned_salicon/erfnet_0.5_art_finetune')
# wandb.init(project="pruned_salicon")
# To prevent PIL warnings.
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument('--arch', default = 'erfNet_RA_parallel_1layer', help='Architectures')
parser.add_argument('--mode', default='finetune', choices=['finetune', 'prune', 'check', 'eval'], help='Run mode')
parser.add_argument('--finetune_layers', choices=['all', 'fc', 'classifier'], default='all', help='Which layers to finetune, fc only works with vgg')
parser.add_argument('--num_outputs', type=int, default=1, help='Num outputs for dataset')
# Optimization options.
parser.add_argument('--lr', type=float,default=5e-4, help='Learning rate')
parser.add_argument('--lr_decay_every', type=int,default =2, help='Step decay every this many epochs')
parser.add_argument('--lr_decay_factor', type=float,default = 0.1, help='Multiply lr by this much every step of decay')
parser.add_argument('--finetune_epochs', type=int,default =20, help='Number of initial finetuning epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
# Paths.
parser.add_argument('--dataset', type=str, default='websal', help='Name of dataset')
parser.add_argument('--dataset_mask', type=str, default='art', help='Name of dataset')
parser.add_argument('--save_prefix', type=str, default="", help='Location to save model')
parser.add_argument('--loadname', type=str, default='', help='Location to save model')
# Pruning options.
parser.add_argument('--prune_method', type=str, default='sparse', choices=['sparse'], help='Pruning method to use')
parser.add_argument('--prune_perc_per_layer', type=float, default=0.5, help='% of neurons to prune per layer')
parser.add_argument('--prune_perc_name', type=str, default='0.5', help='% of neurons to prune per layer')
parser.add_argument('--post_prune_epochs', type=int, default=10, help='Number of epochs to finetune for after pruning')
parser.add_argument('--disable_pruning_mask', action='store_true', default=False, help='use masking or not')
parser.add_argument('--train_biases', action='store_true', default=False, help='use separate biases or not')
parser.add_argument('--train_bn', action='store_true', default=True, help='train batch norm or not')
# Other.
parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
parser.add_argument('--init_dump', action='store_true', default=False, help='Initial model dump.')
parser.add_argument('--pretrainedModel', default = '/home/yzfan3/continual_learning/packnet-master/checkpoints/salicon/erfnet_pruned_0.5_1_final.pt')
parser.add_argument('--image_size', default=(1024,512), type=int, help='resized image resolution for training: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--tr_fxt_size', default=(512,256), type=int, help='resized training fixation resolution: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--val_fxt_size', default=(512,256), type=int, help='resized validation fixation resolution: (600, 800) | (480, 640) | (320, 640)') 
parser.add_argument('--train_art_img_dir', default='', type=str, help='training images path')
parser.add_argument('--train_art_gt_dir', default='', type=str, help='training human fixation maps path')
parser.add_argument('--val_art_img_dir', default='', type=str, help='validation images path')
parser.add_argument('--val_art_gt_dir', default='', type=str, help='validation human fixation maps path')
parser.add_argument('--train_websal_img_dir', default='', type=str, help='training images path')
parser.add_argument('--train_websal_gt_dir', default='', type=str, help='training human fixation maps path')
parser.add_argument('--val_websal_img_dir', default='', type=str, help='validation images path')
parser.add_argument('--val_websal_gt_dir', default='', type=str, help='validation human fixation maps path')
parser.add_argument('--task', type=int, default=0, help='current_task')
parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--nb_tasks', type=int, default=3)
parser.add_argument('--current_task', type=int, default=2)
parser.add_argument('--num_classes', type=int, nargs="+", help='pass list with number of classes', default=[20, 20, 20])


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, model_old, previous_masks, masks_arch_current, dataset2idx, dataset2biases):
        self.args = args
        self.cuda = args.cuda
        self.model = model
        self.model_old = model_old
        self.dataset2idx = dataset2idx
        self.dataset2biases = dataset2biases
        self.task = args.current_task

               
        if args.mode != 'check':
            testImgs = load_allimages(args.val_art_img_dir)
            oneimage = testImgs[0][0]
            oneimage = datasets.folder.default_loader(oneimage)
            oneimage = transforms.Resize(args.image_size)(oneimage)
            oneimage = transforms.ToTensor()(oneimage)
            oneimage = oneimage.view([1]+list(oneimage.size()))
            oneimage = Variable(oneimage).cuda()
            out_put = model(oneimage,args.current_task)
            # Set up data loader, criterion, and pruner.
            self.train_loader, self.test_loader = create_websal_data_loaders(args,
                                                outSize=tuple(out_put.size()[2:]), 
                                                imgSize=args.image_size, 
                                                trFxtSize=args.tr_fxt_size,
                                                valFxtSize=args.val_fxt_size,
                                                flip=False)
            self.criterion = TVdist

            self.pruner = SparsePruner(
                self.model, self.args.prune_perc_per_layer, previous_masks, masks_arch_current,
                self.args.train_biases, self.args.train_bn,self.args.mode)

    def eval(self, dataset_idx, criterion,biases=None):
        """Performs evaluation."""
        epoch_val_loss_value = []
        if not self.args.disable_pruning_mask:
            self.pruner.apply_mask(dataset_idx)
        if biases is not None:
            self.pruner.restore_biases(biases)
        with torch.no_grad():
            self.model.eval()

            print('Performing eval...') 
            for step,(batch, label) in enumerate(self.test_loader):
                if self.cuda:
                    batch = batch.cuda()
                    label = label.cuda()
                batch = Variable(batch, volatile=True)
                output = self.model(batch,self.task)
                if output.shape[2] != label.shape[2] or output.shape[3] != label.shape[3]:
                    output = F.interpolate(output,size=label.size()[2:], mode='bilinear', align_corners=True)

                loss = criterion(output, label)
                epoch_val_loss_value.append(loss.item())
                average_epoch_val_loss_value = sum(epoch_val_loss_value)/len(epoch_val_loss_value)

        if self.args.train_bn:
            self.model.train()
        else:
            self.model.train_nobn()
        return average_epoch_val_loss_value

    def do_batch(self, optimizer, batch, label):
        """Runs model for one batch."""
        if self.cuda:
            batch = batch.cuda()
            label = label.cuda()
        batch = Variable(batch)
        label = Variable(label)

        # Set grads to 0.
        self.model.zero_grad()

        # Do forward-backward.
        output = self.model(batch,self.task)
        if output.shape[2] != label.shape[2] or output.shape[3] != label.shape[3]:
            output = F.interpolate(output,size=label.size()[2:], mode='bilinear', align_corners=True)
        loss = self.criterion(output, label)
        loss.backward()
        # Set fixed param grads to 0.
        if not self.args.disable_pruning_mask:
            self.pruner.make_grads_zero()

        # Update params.
        optimizer.step()

        # Set pruned weights to 0.
        if not self.args.disable_pruning_mask:
            self.pruner.make_pruned_zero()
        return loss
    def do_epoch(self, epoch_idx, optimizer):
        """Trains model for one epoch."""
        epoch_train_loss_value = []
        for step,(batch, label) in enumerate(self.train_loader):
            batch_loss = self.do_batch(optimizer, batch, label)
            epoch_train_loss_value.append(batch_loss.item())
        average_epoch_loss_train = sum(epoch_train_loss_value)/len(epoch_train_loss_value)    
        return average_epoch_loss_train

    def save_model(self, epoch, min_loss, savename):
        """Saves model to file."""
        base_model = self.model

        # Prepare the ckpt.
        self.dataset2idx[self.args.dataset] = self.pruner.current_dataset_idx
        self.dataset2biases[self.args.dataset] = self.pruner.get_biases()
        ckpt = {
            'args': self.args,
            'epoch': epoch,
            'loss': min_loss,
            'model_state_dict':base_model.state_dict(),
            'dataset2idx': self.dataset2idx,
            'previous_masks': self.pruner.current_masks,
            'masks_arch_current': self.pruner.masks_arch,
            'model': base_model,
            # 'optimizer': optimizer.state_dict(),
        }
        if self.args.train_biases:
            ckpt['dataset2biases'] = self.dataset2biases

        # Save to file.
        torch.save(ckpt, savename + '.pt')
    

    def train(self, epochs, optimizer, save=True, savename='', min_loss= np.inf):
        """Performs training."""
        for name, m in self.model.named_parameters():
            if 'decoder' in name:
                if 'decoder.{}'.format(self.args.current_task) not in name:
                    m.requires_grad = False

            elif 'encoder' in name:
                if 'bn' in name or 'parallel_conv' in name:
                    if '.{}.weight'.format(self.args.current_task) in name or '.{}.bias'.format(self.args.current_task) in name:
                        continue
                    else:
                        m.requires_grad = False
        min_loss = min_loss
        criterion = self.criterion
        if self.args.cuda:
            self.model = self.model.cuda()
        
        print("Let's use", torch.cuda.device_count(), "GPUs!")


        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))
            optimizer = step_lr(epoch_idx, self.args.lr, self.args.lr_decay_every,
                                      self.args.lr_decay_factor, optimizer)
            if self.args.train_bn:
                self.model.train()
            else:
                self.model.train_nobn()
            train_loss = self.do_epoch(epoch_idx, optimizer)
            writer.add_scalar('train_loss',train_loss,idx)
            val_loss = self.eval(self.pruner.current_dataset_idx,criterion)
            writer.add_scalar('val_loss',val_loss,idx)

            # Save best model, if required.
            if save and val_loss < min_loss:
                print('Best model so far, loss: %0.2f%% -> %0.2f%%' %
                      (min_loss, val_loss))
                min_loss = val_loss
                self.save_model(epoch_idx, min_loss, savename)

        print('Finished finetuning...')
        print('-' * 16)

    def prune(self):
        """Perform pruning."""
        print('Pre-prune eval:')
        self.eval(self.pruner.current_dataset_idx, self.criterion)

        self.pruner.prune()
        self.check(True)

        print('\nPost-prune eval:')
        loss = self.eval(self.pruner.current_dataset_idx, self.criterion)

        # Do final finetuning to improve results on pruned network.
        if self.args.post_prune_epochs:
            print('Doing some extra finetuning...')
            optimizer = Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=self.args.weight_decay)
            
            self.train(self.args.post_prune_epochs, optimizer, save=True,
                       savename=self.args.save_prefix + self.args.arch + '_' + self.args.dataset + '_pruned_' + str(
                           self.args.prune_perc_per_layer) + '_base_' + str(self.args.prune_perc_per_layer) + '_final',
                       min_loss=logMultiprocessing)
        print('-' * 16)
        print('Pruning summary:')
        self.check(True)
        print('-' * 16)

    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for layer_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params))
                          
                          

def init_dump(args):
    """Dumps pretrained model in required format."""
    if args.arch == 'erfNet_RA_parallel_1layer':
        model = erfNet_RA_parallel_1layer()
    elif args.arch == 'erfNet_RA_parallel_2layer025':
        model = erfNet_RA_parallel_2layer025()
    else:
        raise ValueError('Architecture type not supported.')

    previous_masks = {}
    new_dict_load = {}
    saved_model = torch.load(args.pretrainedModel)['state_dict']
    for k, v in saved_model.items():
        nkey = re.sub("module.", "", k)
        new_dict_load[nkey] = v

    model.load_state_dict(new_dict_load)
    for module_idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
            if 'cuda' in module.weight.data.type():
                mask = mask.cuda()
            previous_masks[module_idx] = mask
    torch.save({
        'dataset2idx': {'websal': 3},
        'previous_masks': previous_masks,
        'model': model,
    }, args.init_savedir + args.arch + '_' + args.dataset + '.pt')
    print('Finished init_dump!')
    
    
    
def main():
    """Do stuff."""
    args = parser.parse_args()
    if args.init_dump:
        init_dump(args)
        return
    if args.prune_perc_per_layer <= 0:
        return

    # Load the required model.
    if args.arch == 'erfNet_RA_parallel_1layer':
        model = erfNet_RA_parallel_1layer(args.num_classes, args.nb_tasks, args.current_task)
        model_old = erfNet_RA_parallel_1layer(args.num_classes, args.nb_tasks - 1, args.current_task - 1)
    elif args.arch == 'erfNet_RA_parallel_2layer025':
        model = erfNet_RA_parallel_2layer025(args.num_classes, args.nb_tasks, args.current_task)
        model_old = erfNet_RA_parallel_2layer025(args.num_classes, args.nb_tasks - 1, args.current_task - 1)
    ckpt = torch.load(args.loadname)
    previous_masks = ckpt['previous_masks']
    masks_arch_current = ckpt['masks_arch_current']
    dataset2idx = ckpt['dataset2idx']
    if 'dataset2biases' in ckpt:
        dataset2biases = ckpt['dataset2biases']
    else:
        dataset2biases = {}
        

    new_dict_load = {}
    for k, v in ckpt['model_state_dict'].items():
        if k in model.state_dict().keys():  # take all the common params as it is
            new_dict_load[k] = v

    # put all the previous task's DS params into current tasks DS params. being used as an init strategy
    for k, v in ckpt['model_state_dict'].items():
        if 'encoder' in k:
            if 'parallel_conv' in k or 'bn' in k:
                if '.{}.weight'.format(args.current_task - 1) in k:
                    nkey = re.sub('.{}.weight'.format(args.current_task - 1),
                                    '.{}.weight'.format(args.current_task), k)
                    new_dict_load[nkey] = v
                elif '.{}.bias'.format(args.current_task - 1) in k:
                    nkey = re.sub('.{}.bias'.format(args.current_task - 1),
                                    '.{}.bias'.format(args.current_task), k)
                    new_dict_load[nkey] = v

        elif 'decoder' in k and 'output_conv' not in k:
            # this is important so as to maintain uniformity among bdd and idd experiments.
            nkey = re.sub('decoder.{}'.format(args.current_task - 1),
                            'decoder.{}'.format(args.current_task), k)
            new_dict_load[nkey] = v

    model.load_state_dict(new_dict_load, strict=False)
    print('loaded model from checkpoint provided.')    

    if args.cuda:
        model = model.cuda()
        model_old = model_old.cuda()
    task = args.current_task
    
    # Create the manager object.
    manager = Manager(args, model,model_old, previous_masks, masks_arch_current, dataset2idx, dataset2biases)

    # Perform necessary mode operations.
    if args.mode == 'finetune':
        # Make pruned params available for new dataset.
        manager.pruner.make_finetuning_mask()

        # Get optimizer with correct params.
        if args.finetune_layers == 'all':
            params_to_optimize = model.parameters()
        elif args.finetune_layers == 'classifier':
            for param in model.net.parameters():
                param.requires_grad = False
            params_to_optimize = model.classifier.parameters()
        elif args.finetune_layers == 'fc':
            params_to_optimize = []
            # Add fc params.
            for param in model.net.parameters():
                if param.size(0) == 4096:
                    param.requires_grad = True
                    params_to_optimize.append(param)
                else:
                    param.requires_grad = False
            # Add classifier params.
            for param in model.classifier.parameters():
                params_to_optimize.append(param)
            params_to_optimize = iter(params_to_optimize)
        optimizer = Adam(params_to_optimize, lr=args.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=args.weight_decay)
        # Perform finetuning.
        manager.train(args.finetune_epochs, optimizer, save=True, savename=args.save_prefix + args.arch + '_' + args.dataset + '_finetune_base_' +  args.prune_perc_name+'_final')
        

    elif args.mode == 'prune':
        # Perform pruning.
        manager.prune()
    elif args.mode == 'check':
        # Load model and make sure everything is fine.
        manager.check(verbose=True)
    elif args.mode == 'eval':
        # Just run the model on the eval set.
        biases = None
        if 'dataset2biases' in ckpt:
            biases = ckpt['dataset2biases'][args.dataset]
        manager.eval(ckpt['dataset2idx'][args.dataset], biases)


if __name__ == '__main__':
    main()
