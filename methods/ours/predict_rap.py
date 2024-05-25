import os
import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(srcFolder)
import re
import argparse
from models.erfnet_RA_parallel_1layer import Net as erfNet_RA_parallel_1layer
from models.erfnet_RA_parallel_2layer025 import Net as erfNet_RA_parallel_2layer025
import collections
args = collections.namedtuple
from methods.utils.training_utils import *

parser = argparse.ArgumentParser(description='Saliency Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lr_decay_epoch', default=2, type=float, help='every n epochs to decay learning rate')
parser.add_argument('--lr_coef', default=.1, type=float, help='lr coefficient to change learning rates')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--epoch', default=30, type=int, help='number of epochs')
parser.add_argument('--batchsize', default=12, type=int, help='batch size for training')
parser.add_argument('--val_batchsize', default=1, type=int, help='batch size for validation')
parser.add_argument('--model', default='erfNet_RA_parallel_1layer', type=str, help='backbone network: erfNet')

parser.add_argument('--type_dispatcher', default=0, type=int, help='0: use baseline, 1: l2 norm, 2: angle')
parser.add_argument('--memsize', default=5, type=int, help='memory size')
parser.add_argument('--mem_window', default=0, type=int, help='reset window')
parser.add_argument('--mem_offset', default=0, type=int, help='memory offset')
parser.add_argument('--mem_strength', default=0.0, type=float, help='memory strength')
parser.add_argument('--mtype', default='dcl', type=str, help='method type: baseline, dcl, gem')

parser.add_argument('--train_img_dir', default='', type=str, help='training images path')
parser.add_argument('--train_gt_dir', default='', type=str, help='training human fixation maps path')
parser.add_argument('--val_img_dir', default='', type=str, help='validation images path')
parser.add_argument('--val_gt_dir', default='', type=str, help='validation human fixation maps path')
parser.add_argument('--image_size', default=(1024,512), type=int, help='image resolution for training: (600, 800) | (480, 640) | (320, 640)')
parser.add_argument('--out_dir', default='/home/yzfan/AudioVisualSaliency/Continual_learning/logs/baseline/rap_art_base_salicon', type=str, help='validation human fixation maps path')
parser.add_argument('--optype', default='adam', type=str, help='optimizer')
parser.add_argument('--random_seed', default=0, type=int, help='random seed')
parser.add_argument('--pretrainedModel', default='', type=str, help='pretrained saliency model path')
parser.add_argument('--gaussblur_sigma', default=-1.0, type=float, help='Gaussian blur sigma')
parser.add_argument('--gaussblur_truncate', default=4.0, type=float, help='Gaussian blur truncate')
parser.add_argument('--file_type', default='jpg', type=str, help='output file type')
parser.add_argument('--num_classes', default=[20,20,20], type=int, help='batch size for training')
parser.add_argument('--nb_tasks', type=int, default=3)
parser.add_argument('--current_task', type=int, default=2)
args = parser.parse_args()

args.start_epoch = 0
args.pretrained = True
args.useMultiGPU = False

args.experiment_name = '{}'.format(args.model)

out_folder = args.out_dir
ensure_dir(out_folder)
if args.image_size is None:
    args.image_size = (480,640)
else:
    args.image_size = (args.image_size[0], args.image_size[1])

n_output = 256


print(vars(args))

# create the model and optimizer
if args.model == 'erfNet_RA_parallel_1layer':
    model = erfNet_RA_parallel_1layer(args.num_classes, args.nb_tasks, args.current_task)
elif args.model == 'erfNet_RA_parallel_2layer025':
    model = erfNet_RA_parallel_2layer025(args.num_classes, args.nb_tasks, args.current_task)
else:
    raise ValueError('Architecture type not supported.')

    
if args.pretrainedModel != '':
    new_dict_load = {}
    saved_model = torch.load(args.pretrainedModel)['model_state_dict']
    for k, v in saved_model.items():
        nkey = re.sub("module.", "", k)
        new_dict_load[nkey] = v

    model.load_state_dict(new_dict_load)

if not args.useMultiGPU:
    model = model.cuda()
elif args.useMultiGPU:
    model = nn.DataParallel(model).cuda()


val_loader = create_test_data_loader(args.val_img_dir, 
                                    args.val_batchsize,
                                    args.image_size)

#print total number of parameters
number_of_params = sum(p.numel() for p in model.parameters())
print('===Total parameters number: {}'.format(number_of_params))

txtlogger = open('{}/{}_result.txt'.format(out_folder, args.model), 'w')
print(vars(args),file=txtlogger, flush=True)

val_batchtime, val_datatime = predict_rap(model, args.current_task, val_loader, os.path.join(out_folder, 'salmaps'), 
                                args.gaussblur_sigma, args.gaussblur_truncate, args.file_type)
print('avg batch time {}, avg data time {}, avg proc time {}'.format(val_batchtime, val_datatime, val_batchtime-val_datatime),file=txtlogger, flush=True)
print('avg batch time {}, avg data time {}, avg proc time {}'.format(val_batchtime, val_datatime, val_batchtime-val_datatime))

txtlogger.close()
