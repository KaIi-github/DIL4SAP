import os
import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(srcFolder)
import os.path
import time

from torchvision import *
from torch.autograd import Variable
from torch.optim import SGD, Adam, lr_scheduler
from tensorboardX import SummaryWriter
from argparse import ArgumentParser

writer = SummaryWriter(log_dir='salicon_rapft_1')
import collections
args = collections.namedtuple

from methods.utils.losses import *
from methods.utils.training_utils import *
from methods.utils.util import *
from models.erfnet_RA_parallel_1layer import Net as erfNet_RA_parallel_1layer
from models.erfnet_RA_parallel_2layer025 import Net as erfNet_RA_parallel_2layer025

NUM_CHANNELS = 3
NUM_CLASSES = 20

def is_shared(n):
    return 'encoder' in n and 'parallel_conv' not in n and 'bn' not in n


def is_DS_curr(n):
    if 'decoder.{}'.format(current_task) in n:
        return True
    elif 'encoder' in n:
        if 'bn' in n or 'parallel_conv' in n:
            if '.{}.weight'.format(current_task) in n or '.{}.bias'.format(current_task) in n:
                return True

best_model = None
def train(args,model):
    global NUM_CLASSES
    NUM_CLASSES = args.num_classes[args.current_task]
    testImgs = load_allimages(args.val_salicon_img_dir)
    oneimage = testImgs[0][0]
    oneimage = datasets.folder.default_loader(oneimage)
    oneimage = transforms.Resize(args.image_size)(oneimage)
    oneimage = transforms.ToTensor()(oneimage)
    oneimage = oneimage.view([1]+list(oneimage.size()))
    oneimage = Variable(oneimage).cuda()
    output = model(input=oneimage, task=0)

    if args.dataset == "MIT1003":
        train_loader, val_loader = create_MIT_data_loaders(args,
                                                        outSize=tuple(output.size()[2:]),
                                                        imgSize=args.image_size,
                                                        trFxtSize=args.tr_fxt_size,
                                                        valFxtSize=args.val_fxt_size,
                                                        flip=False)

    elif args.dataset == "salicon":
        train_loader, val_loader = create_salicon_data_loaders(args,
                                                        outSize=tuple(output.size()[2:]),
                                                        imgSize=args.image_size,
                                                        trFxtSize=args.tr_fxt_size,
                                                        valFxtSize=args.val_fxt_size,
                                                        flip=False)
    elif args.dataset == "art":
        train_loader, val_loader = create_art_data_loaders(args,
                                                    outSize=tuple(output.size()[2:]),
                                                    imgSize=args.image_size,
                                                    trFxtSize=args.tr_fxt_size,
                                                    valFxtSize=args.val_fxt_size,
                                                    flip=False)
    elif args.dataset == "websal":
        train_loader, val_loader = create_websal_data_loaders(args,
                                                    outSize=tuple(output.size()[2:]),
                                                    imgSize=args.image_size,
                                                    trFxtSize=args.tr_fxt_size,
                                                    valFxtSize=args.val_fxt_size,
                                                    flip=False)

    criterion = TVdist

    '''
    RAP-FT model: freeze only DS parameters of the previous domains. Shared params will be trained.
    Freeze: previous decoders + previous DS 'bn' and 'parallel conv' layers
    '''


    for name, m in model.named_parameters():
        if 'decoder' in name:
            if 'decoder.{}'.format(current_task) not in name:
                m.requires_grad = False

        elif 'encoder' in name:
            if 'bn' in name or 'parallel_conv' in name:
                if '.{}.weight'.format(current_task) in name or '.{}.bias'.format(current_task) in name:
                    continue
                else:
                    m.requires_grad = False

    savedir = args.savedir

    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    params = list(model.named_parameters())

    grouped_parameters = [
        # only the shared conv layers in the encoder will use this lr
        {"params": [p for n, p in params if is_shared(n)], 'lr': 5e-6},
        {"params": [p for n, p in params if is_DS_curr(n)]},  # is domain-specific to current domain
    ]

    optimizer = Adam(grouped_parameters, 5e-4, (0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    def lambda1(epoch): return pow((1-((epoch-1)/args.epochs)), 0.9)  # scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  # scheduler 2

    start_epoch = 1

    for epoch in range(start_epoch, args.epochs+1):
        # ensure its set to the correct #classes for training the current dataset
        NUM_CLASSES = args.num_classes[args.current_task]
        print("-----TRAINING - EPOCH---", epoch, "-----")
        scheduler.step(epoch)  # scheduler 2

        epoch_loss = []
        time_train = []

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()

        for step, (images, labels) in enumerate(train_loader):
            start_time = time.time()
            if args.cuda:
                inputs = images.cuda()
                if type(labels).__name__=='list':
                    targets = [Variable(labels[i], requires_grad=False).cuda() for i in range(len(labels))]
                else:
                    targets = Variable(labels, requires_grad=False).cuda()

            outputs = model(inputs,current_task)
            if outputs.shape[2] != targets.shape[2] or outputs.shape[3] != targets.shape[3]:
                outputs = F.interpolate(outputs,size=targets.size()[2:], mode='bilinear', align_corners=True)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()  # should backprop ce_loss in all new Ds and shared params.
            # should backprop the loss only in the shared encoder params - it will be passed through the DS_CS params but they will be freezed so not updated
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        print('epoch took: ', sum(time_train))

        if epoch % 10 == 0 or epoch % 1 == 0:
            print("----- VALIDATING - EPOCH", epoch, "-----")
            # validate current task
            average_loss_val = eval(model, val_loader, criterion, current_task, args.num_classes, epoch)

        # logging tensorboard plots - epoch wise loss and accuracy. Not calculating iouTrain as that will slow down training
        info = {'train_loss': average_epoch_loss_train, 'val_loss_{}'.format(args.dataset): average_loss_val}

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch)

        # remember best val_loss and save checkpoint
        if epoch == 1 :
            average_loss_val = np.inf
            best_loss = average_loss_val
        if average_loss_val <= best_loss:
            best_loss = average_loss_val
        is_best = average_loss_val <= best_loss

        filenameCheckpoint = savedir + '/checkpoint_{}_{}_{}_{}{}.pth'.format(
                args.dataset, args.model, args.epochs, args.batch_size, args.model_name_suffix)
        filenameBest = savedir + '/model_best_{}_{}_{}_{}{}.pth'.format(
                args.dataset, args.model, args.epochs, args.batch_size, args.model_name_suffix)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        if (is_best):
            with open(savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-Loss= %.4f" % (epoch, average_loss_val))

        # SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, LR)
        # Epoch		Train-loss		Test-loss	learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (
                epoch, average_epoch_loss_train, average_loss_val, usedLr))

    return(model)

def eval(model, dataset_loader, criterion, task, num_classes, epoch):
    # Validate on val images after each epoch of training
    global NUM_CLASSES
    model.eval()
    epoch_loss_val = []
    time_val = []
    num_cls = num_classes[task]
    NUM_CLASSES = num_cls
    print('number of classes in current task: ', num_cls)
    print('validating task: ', task)

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataset_loader):
            start_time = time.time()
            inputs = images.cuda()
            targets = labels.cuda()

            outputs = model(inputs, task)
            if step == 1:
                print('------------------', outputs.size(), targets.size())

            loss = criterion(outputs, targets)
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)

            if 50 > 0 and step % 50 == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / 6))

    average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

    return average_epoch_loss_val


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    print("Saving model: ", filenameCheckpoint)
    if is_best:
        print("Saving model as best: ", filenameBest)
        torch.save(state, filenameBest)


def main(args):
    global current_task
    current_task = args.current_task

    print('\ndataset: ', args.dataset)
    savedir = args.savedir

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # Load Model
    if args.model == 'erfNet_RA_parallel_1layer':
        model = erfNet_RA_parallel_1layer(args.num_classes, args.nb_tasks, args.current_task)
    elif args.model == 'erfNet_RA_parallel_2layer025':
        model = erfNet_RA_parallel_2layer025(args.num_classes, args.nb_tasks, args.current_task)
    else:
        assert ValueError('Architecture type not supported.')

    if args.cuda:
        model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.pretrained:
        saved_model = torch.load(args.pretrainedModel)
        model.load_state_dict(saved_model['state_dict'], strict=False)
        print('loaded model from checkpoint provided.')

    print('loaded\n')

    model = train(args, model)
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    # NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--lr_decay_epoch', default=5, type=float, help='every n epochs to decay learning rate')
    parser.add_argument('--lr_coef', default=.1, type=float, help='lr coefficient to change learning rates')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=2, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size for training')
    parser.add_argument('--val_batch_size', default=1, type=int, help='batch size for validation')
    parser.add_argument('--model', default='erfNet_RA_parallel_1layer', type=str, help='backbone network: refnet')
    parser.add_argument('--pretrained', default=True)
    parser.add_argument('--pretrainedModel', default='', type=str, help='pretrained saliency model')
    parser.add_argument('--train_MIT_img_dir', default='', type=str, help='training images path')
    parser.add_argument('--train_MIT_gt_dir', default='', type=str, help='training human fixation maps path')
    parser.add_argument('--val_MIT_img_dir', default='', type=str, help='validation images path')
    parser.add_argument('--val_MIT_gt_dir', default='', type=str, help='validation human fixation maps path')
    parser.add_argument('--train_salicon_img_dir', default='', type=str, help='training images path')
    parser.add_argument('--train_salicon_gt_dir', default='', type=str, help='training human fixation maps path')
    parser.add_argument('--val_salicon_img_dir', default='', type=str, help='validation images path')
    parser.add_argument('--val_salicon_gt_dir', default='', type=str, help='validation human fixation maps path')
    parser.add_argument('--train_art_img_dir', default='', type=str, help='training images path')
    parser.add_argument('--train_art_gt_dir', default='', type=str, help='training human fixation maps path')
    parser.add_argument('--val_art_img_dir', default='', type=str, help='validation images path')
    parser.add_argument('--val_art_gt_dir', default='', type=str, help='validation human fixation maps path')
    parser.add_argument('--train_websal_img_dir', default='', type=str, help='training images path')
    parser.add_argument('--train_websal_gt_dir', default='', type=str, help='training human fixation maps path')
    parser.add_argument('--val_websal_img_dir', default='', type=str, help='validation images path')
    parser.add_argument('--val_websal_gt_dir', default='', type=str, help='validation human fixation maps path')
    parser.add_argument('--image_size', default=(1024, 512), type=int,
                        help='resized image resolution for training: (600, 800) | (480, 640) | (320, 640)')
    parser.add_argument('--tr_fxt_size', default=(512, 256), type=int,
                        help='resized training fixation resolution: (600, 800) | (480, 640) | (320, 640)')
    parser.add_argument('--val_fxt_size', default=(512, 256), type=int,
                        help='resized validation fixation resolution: (600, 800) | (480, 640) | (320, 640)')
    parser.add_argument('--fxt_loc_name', type=str, default='fixationPts', help='fixationPts|fixLocs')
    parser.add_argument('--random_seed', default=0, type=int, help='random seed')
    parser.add_argument('--model_val_path', default="", type=str,  help='save_best_model_path')
    parser.add_argument('--num_classes', type=int, nargs="+", default=[20, 20], help='pass list with number of classes')#
    parser.add_argument('--num_classes_old', type=int, nargs="+", default=[20], help='pass list with number of classes in previous task model, t-1 model')
    parser.add_argument('--nb_tasks', type=int, default=1)
    parser.add_argument('--current_task', type=int, default=0)

    # to be tuned, for now based on ADVENT
    parser.add_argument('--lambdac', type=float, default=0.1)
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--steps_loss', type=int, default=50)
    parser.add_argument('--steps_plot', type=int, default=50)
    # You can use this value to save model every X epochs
    parser.add_argument('--epochs_save', type=int, default=0)
    parser.add_argument('--savedir', default='')
    parser.add_argument('--dataset', default="salicon")
    parser.add_argument('--dataset_old', default="salicon")
    parser.add_argument('--model_name_suffix', default="RAPFT")

    main(parser.parse_args())