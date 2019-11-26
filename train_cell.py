from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='CELL', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=CELL_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')

parser.add_argument('--trained_model', default='weights/SSD512VOC.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--freeze_layer', default='conv1_2',
                    help='Name of freezying layers')
parser.add_argument('--focal_loss', default=False,
                    help='Use focal loss instead of cross entropy')
parser.add_argument('--resnet101', default=False,
                    help='Change backbone to ResNet101')
parser.add_argument('--use_pretrained', default=False,
                    help='Change backbone to ResNet101')
args = parser.parse_args()

if args.visdom:
    import visdom
    viz = visdom.Visdom(server='pc70')


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
# print('CUDA available:', torch.cuda.is_available())
# print('freeze layer: ', args.freeze_layer)

def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'CELL':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = cell
        dataset = CELLDetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net
    # print(net)

    if args.use_pretrained:
        # load pretrained model
        trained_model = torch.load(HOME + '/MA/ssd_cell/' + args.trained_model)

        # fine tune
        # mdfmodel = modify_model(trained_model)
        mdfmodel = modify_model_4multibox(trained_model)

        # load modified model
        print('Loading model...')
        net.load_state_dict(mdfmodel)

        # freeze layers vgg until convx_3
        freezing_dict = {'none': 0, 'conv1_2': 3, 'conv2_1': 6, 'conv2_2': 8,
                         'conv3_1': 11, 'conv3_2': 13, 'conv3_3': 15,
                         'conv4_1': 18, 'conv4_2': 20, 'conv4_3': 22,
                         'conv4_3_ReLu': 23, 'conv4_3_MaxP': 24,
                         'conv5_1': 25, 'conv5_2': 27, 'conv5_3': 29}

        layer = freezing_dict[args.freeze_layer]
        if layer != 0:
            for para in net.vgg[:layer].parameters():
                para.requires_grad = False
        print('Finished freezing vgg layers...')

    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # if args.resume:
    #     print('Resuming training, loading {}...'.format(args.resume))
    #     ssd_net.load_weights(args.resume)
    # else:
    #     vgg_weights = torch.load(args.save_folder + args.basenet)
    #     print('Loading base network...')
    #     ssd_net.vgg.load_state_dict(vgg_weights)

    # if args.cuda:
    #     net = net.cuda()

    # if not args.resume:
    #     print('Initializing weights...')
    #     # initialize newly added layers' weights with xavier method
    #     ssd_net.extras.apply(weights_init)
    #     ssd_net.loc.apply(weights_init)
    #     ssd_net.conf.apply(weights_init)

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        cudnn.benchmark = True
        net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda, args.focal_loss)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # create batch iterator
    batch_iterator = iter(data_loader)
    best_prior = torch.Tensor([0])
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            # targets = [Variable(ann, volatile=True) for ann in targets]
            targets = [Variable(ann) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, best_prior_tmp = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        # loc_loss += loss_l.data[0]
        loc_loss += loss_l.data
        # conf_loss += loss_c.data[0]
        loc_loss += loss_l.data
        print('loss_l.data', loss_l.data)

        if iteration <= 10:
            best_prior = best_prior + best_prior_tmp
            # print(best_prior)

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % loss.data, end=' ')
            # print('Sum of all data overlap: ', best_prior)

        if args.visdom:
            update_vis_plot(iteration, loss_l.data, loss_c.data,
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 10000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), args.save_folder + args.freeze_layer +
                       '-' + repr(iteration) + '.pth')
            # /work/scratch/zhou/MA/ssd_cell/weights/CELLconv4_x_repr(iteration).pth

    torch.save(ssd_net.state_dict(), args.save_folder + args.freeze_layer + '.pth')


def modify_model(trained_model):
    length_model = len(trained_model)
    # define modeifed model
    mdfmodel = {}
    mbox = [4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4]
    j = 0
    for i, (k, v) in enumerate(trained_model.items()):
        if i < (length_model - 14):
            mdfmodel[k] = v
        else:
            tmp_size = list(trained_model[k].shape)
            tmp_size[0] = mbox[j] * cell['num_classes']
            mdfmodel[k] = torch.zeros(tmp_size)
            j += 1
    return mdfmodel


def modify_model_4multibox(trained_model):
    remove_layers = ['extras.4.weight', 'extras.4.bias', 'extras.5.weight',
                     'extras.5.bias', 'extras.6.weight', 'extras.6.bias',
                     'extras.7.weight', 'extras.7.bias', 'extras.8.weight',
                     'extras.8.bias', 'extras.9.weight', 'extras.9.bias',
                     'loc.4.weight', 'loc.4.bias', 'loc.5.weight', 'loc.5.bias',
                     'loc.6.weight', 'loc.6.bias',
                     'conf.4.weight', 'conf.4.bias', 'conf.5.weight', 'conf.5.bias',
                     'conf.6.weight', 'conf.6.bias']
    modify_layers = ['conf.0.weight', 'conf.0.bias', 'conf.1.weight', 'conf.1.bias',
                     'conf.2.weight', 'conf.2.bias', 'conf.3.weight', 'conf.3.bias']

    # define modeifed model
    mdfmodel = {}
    mbox = [4, 4, 6, 6, 6, 6, 6, 6]
    j = 0
    for _, (k, v) in enumerate(trained_model.items()):
        if k in remove_layers:
            continue

        elif k in modify_layers:
            tmp_size = list(trained_model[k].shape)
            tmp_size[0] = mbox[j] * cell['num_classes']
            mdfmodel[k] = torch.zeros(tmp_size)
            j += 1

        else:
            mdfmodel[k] = v

    return mdfmodel


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=None
        )


if __name__ == '__main__':
    train()
