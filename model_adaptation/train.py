"""
Author:
Hiroki Azuma, ornitho1027@gmail.com

Our codes heavily rely on https://github.com/Red-Fairy/ZeroShotDayNightDA by Rundong Luo, rundongluo2002@gmail.com.
We really appreciate the awesome codes.
"""

from utils.helpers import gen_train_dirs, plot_confusion_matrix, get_multi_train_trans, get_test_trans
from utils.routines import evaluate, train_epoch_BYOL_dual, train_epoch_SimSiam_dual, train_epoch_similarity_dual
from utils.scheduler import PolyLR
from datasets.cityscapes_ext import CityscapesExt
from datasets.acdc import ACDC
from datasets.gta5 import GTA5DataSet
from models.segmentation.modeling import deeplabv3plus_byol_resnet50, deeplabv3plus_simsiam_resnet50, deeplabv3plus_similarity_resnet50, deeplabv3plus_similarity_resnet101, refinenet_similarity_resnet50
from datasets.paired_city import CityTwoDomains

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

import shutil, time, random
import matplotlib.pyplot as plt
import numpy as np
import os

class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")

def main(args):
    
    # Configure dataset paths here, paths should look like this
    source_path = args.source_path
    synth_path = os.path.join(source_path, args.synth_dir, 'train')
    target_path = args.target_path

    # Fix seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    # Generate training directories
    gen_train_dirs(args.experiment)

    log = logger('./')
    log.info(str(args))
    log.info('--- Training args ---')

    # Generate log files
    with open('logs/log_batch.csv', 'a') as batch_log:
        batch_log.write('epoch, epoch step, train loss, avg train loss, avg train source loss, avg train sim loss, train acc, avg train acc\n')
    with open('logs/log_epoch.csv', 'a') as epoch_log:
        epoch_log.write('epoch, train loss, train source loss, train synth loss, train sim loss, val loss source, train acc, train acc synth, val acc source, test target acc, miou, miou target, learning rate, scale \n')

    # Initialize metrics
    best_miou = 0.0
    best_miou_target = 0.0
    metrics = {'train_loss': [],
               'train_loss_source': [],
                'train_loss_synth': [],
               'train_loss_sim': [],
               'train_acc': [],
                'train_synth_acc': [],
               'val_acc_cs': [],
               'val_loss_cs': [],
               'miou_cs': [],
               'test_acc_target': [],
               'test_loss_target': [],
               'miou_target': []}
    start_epoch = 0

    # Define data transformation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    target_size = (512,1024)
    crop_size = (384,768)

    train_trans = get_multi_train_trans(mean, std, target_size, crop_size, args.jitter, args.scale, True, args.blur, args.cutout)
    test_trans = get_test_trans(mean, std, target_size)

    # Load dataset
    trainset = CityTwoDomains(root=source_path, root_new_domain=synth_path, transforms=train_trans)
    valset = CityscapesExt(source_path, split='val', target_type='semantic', transforms=test_trans)

    if args.target_type == 'acdc_night':
        testset_target = ACDC(target_path, split='val', transforms=test_trans, ACDC_sub='night')
    elif args.target_type == 'acdc_snow':
        testset_target = ACDC(target_path, split='val', transforms=test_trans, ACDC_sub='snow')
    elif args.target_type == 'acdc_rain':
        testset_target = ACDC(target_path, split='val', transforms=test_trans, ACDC_sub='rain')
    elif args.target_type == 'acdc_fog':
        testset_target = ACDC(target_path, split='val', transforms=test_trans, ACDC_sub='fog')  
    elif args.target_type == 'gta5':
        list_path = "./datasets/gta5_list/gtav_split_val.txt"
        testset_target = GTA5DataSet(target_path, list_path=list_path, transforms=test_trans)
    else:
        raise ValueError

    # Use mini-dataset for debugging purposes
    if args.xs:
        trainset = Subset(trainset, list(range(5)))
        valset = Subset(valset, list(range(5)))
        testset_target = Subset(testset_target, list(range(5)))
        log.info('WARNING: XS_DATASET SET TRUE')

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)
    dataloaders['val'] = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_target'] = torch.utils.data.DataLoader(testset_target, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    num_classes = len(CityscapesExt.validClasses)

    # Define model, loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)

    if args.load is None and args.resume is None:
        if args.ssl == 'byol':
            if args.model == 'resnet-50':
                model = deeplabv3plus_byol_resnet50(num_classes=num_classes, backbone_pretrained=True)
            else:
                raise NotImplementedError
        elif args.ssl == 'simsiam':
            if args.model == 'resnet-50':
                model = deeplabv3plus_simsiam_resnet50(num_classes=num_classes, backbone_pretrained=True)
        elif args.ssl == 'sim':
            if args.model == 'resnet-50':
                model = deeplabv3plus_similarity_resnet50(num_classes=num_classes, backbone_pretrained=True)
            elif args.model == 'resnet-101':
                model = deeplabv3plus_similarity_resnet101(num_classes=num_classes, backbone_pretrained=True)
            elif args.model == 'refinenet':
                model = refinenet_similarity_resnet50(num_classes=num_classes, backbone_pretrained=True)
            else:
                raise ValueError
        else:
            raise ValueError
    else:
        if args.ssl == 'byol':
            if args.model == 'resnet-50':
                model = deeplabv3plus_byol_resnet50(num_classes=num_classes, backbone_pretrained=False)
            else:
                raise NotImplementedError
        elif args.ssl == 'simsiam':
            if args.model == 'resnet-50':
                model = deeplabv3plus_simsiam_resnet50(num_classes=num_classes, backbone_pretrained=False)
            else:
                raise NotImplementedError
        elif args.ssl == 'sim':
            if args.model == 'resnet-50':
                model = deeplabv3plus_similarity_resnet50(num_classes=num_classes, backbone_pretrained=False)
            elif args.model == 'resnet-101':
                model = deeplabv3plus_similarity_resnet101(num_classes=num_classes, backbone_pretrained=False)
            elif args.model == 'refinenet':
                model = refinenet_similarity_resnet50(num_classes=num_classes, backbone_pretrained=False)
            else:
                raise ValueError
        else:
            raise ValueError

    if torch.cuda.is_available():    # Push model to GPU
        model = torch.nn.DataParallel(model).cuda()
        log.info('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
    
    if args.load is not None:
        state_dict = torch.load(args.load)['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.module._init_encoder_k()
        log.info('Loaded pretrained model from {}'.format(args.load))

    params_1 = []
    for name, param in model.named_parameters():
        if param.requires_grad and ('head' in name or 'pred' in name):
            params_1.append(param)

    params_2 = []
    for name, param in model.named_parameters():
        if param.requires_grad and not ('head' in name or 'pred' in name):
            params_2.append(param)
    
    optimizer = torch.optim.SGD([
        {"params": params_1, "lr": args.lr_head},
        {"params": params_2, "lr": args.lr}
        ], lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    # Prepare schedular
    if args.poly:
        scheduler = PolyLR(optimizer, args.epochs, power=0.9)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step)


    # Resume training from checkpoint
    if args.resume:
        log.info('Resuming training from {}.'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_miou_target = checkpoint['best_miou_target']
            best_miou = checkpoint['best_miou']
        except:
            log.info('Could not load optimizer state dict. Initializing optimizer from scratch.')
            epoch = checkpoint['epoch']
            for _ in range(epoch):
                scheduler.step()
        log.info(f'Current LR: {optimizer.param_groups[0]["lr"]}')
        start_epoch = checkpoint['epoch']+1

    since = time.time()

    train_epoch = {
        'byol': train_epoch_BYOL_dual,
        'simsiam': train_epoch_SimSiam_dual,
        'sim': train_epoch_similarity_dual,
    }
    for epoch in range(start_epoch, args.epochs):

        # Train
        log.info('--- Training ---')
        train_loss, train_loss_source, train_loss_synth, train_loss_sim, train_acc, train_synth_acc =\
            train_epoch[args.ssl](dataloaders['train'], model, criterion, optimizer, epoch, log, 
                            CityscapesExt.voidClass, args.sim_weight)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        metrics['train_loss'].append(train_loss)
        metrics['train_loss_source'].append(train_loss_source)
        metrics['train_loss_synth'].append(train_loss_synth)
        metrics['train_loss_sim'].append(train_loss_sim)
        metrics['train_acc'].append(train_acc)
        metrics['train_synth_acc'].append(train_synth_acc)

        # Validate
        log.info('--- Validation - Source ---')
        val_acc_cs, val_loss_cs, miou_cs, confmat_cs, iousum_cs = evaluate(dataloaders['val'],
            model, criterion, epoch, CityscapesExt.classLabels, CityscapesExt.validClasses, log,
            void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std)
        log.info('--- Validation - Target ---')
        test_acc_target, test_loss_target, miou_target, confmat_target, iousum_target = evaluate(dataloaders['test_target'],
            model, criterion, epoch, CityscapesExt.classLabels, CityscapesExt.validClasses, log,
            void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std)
        metrics['val_acc_cs'].append(val_acc_cs)
        metrics['val_loss_cs'].append(val_loss_cs)
        metrics['miou_cs'].append(miou_cs)
        metrics['test_acc_target'].append(test_acc_target)
        metrics['test_loss_target'].append(test_loss_target)
        metrics['miou_target'].append(miou_target)

        # Write logs
        with open('logs/log_epoch.csv', 'a') as epoch_log:
            epoch_log.write('{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(
                    epoch, train_loss, train_loss_source, train_loss_synth, train_loss_sim, val_loss_cs, train_acc, train_synth_acc, val_acc_cs, test_acc_target, miou_cs, miou_target, lr))
        with open('logs/class_iou.txt', 'w') as ioufile:
            ioufile.write(iousum_cs)
            ioufile.write(iousum_target)
        # Plot confusion matrices
        cm_title = 'mIoU : {:.3f}, acc : {:.3f}'.format(miou_cs, val_acc_cs)
        plot_confusion_matrix(confmat_cs,CityscapesExt.classLabels,normalize=True,title=cm_title).savefig('logs/confmat_cs.pdf', bbox_inches='tight')
        cm_title = 'mIoU : {:.3f}, acc : {:.3f}'.format(miou_target, test_acc_target)
        plot_confusion_matrix(confmat_target,CityscapesExt.classLabels,normalize=True,title=cm_title).savefig('logs/confmat_target.pdf', bbox_inches='tight')

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou_target': best_miou_target,
            'best_miou': best_miou,
            'metrics': metrics,
            }, f'weights/checkpoint.pth.tar')

        if miou_target > best_miou_target:
            log.info('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou_target, miou_target))
            best_miou_target = miou_target
            best_acc_target = test_acc_target # acc corresponding to the best miou
            shutil.copy('logs/confmat_cs.pdf', 'logs/best_target_confmat_cs.pdf') # save confmat
            shutil.copy('logs/confmat_target.pdf', 'logs/best_target_confmat_target.pdf') # save confmat
            shutil.copy('logs/class_iou.txt', 'logs/best_target_class_iou.txt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, 'weights/best_weights_target.pth.tar')

        if miou_cs > best_miou:
            log.info('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, miou_cs))
            best_miou = miou_cs
            best_acc = val_acc_cs # acc corresponding to the best miou
            shutil.copy('logs/confmat_cs.pdf', 'logs/best_confmat_cs.pdf') # save confmat
            shutil.copy('logs/confmat_target.pdf', 'logs/best_confmat_target.pdf') # save confmat
            shutil.copy('logs/class_iou.txt', 'logs/best_class_iou.txt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, 'weights/best_weights.pth.tar')

    time_elapsed = time.time() - since
    log.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    log.info('Best mIoU target: {:4f}'.format(best_miou_target))

    # Plot learning curves
    x = np.arange(args.epochs)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('miou')
    ln1 = ax1.plot(x, metrics['miou_cs'], color='tab:red')
    ln2 = ax1.plot(x, metrics['miou_target'], color='tab:green')
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln4 = ax2.plot(x, metrics['val_acc_cs'], color='tab:red', linestyle='dashed')
    ln5 = ax2.plot(x, metrics['test_acc_target'], color='tab:green', linestyle='dashed')
    lns = ln1+ln2+ln4+ln5
    plt.legend(lns, ['CS mIoU','Target mIoU', 'CS Accuracy','Target Accuracy'])
    plt.tight_layout()
    plt.savefig('logs/learning_curve.pdf')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation training and evaluation')
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--synth_dir', type=str, required=True)
    parser.add_argument('--target_type', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)

    parser.add_argument('--init-scale', metavar='1.0', default=[1.0], type=float,
                        help='initial value for scale')
    parser.add_argument('--resume', type=str, default=None,
                        help='resume training from checkpoint')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--step', type=int, default=30)
    parser.add_argument('--lr_head', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--cutout', action='store_true')

    parser.add_argument('--jitter', type=float, default=0.5, metavar='J',
                        help='color jitter augmentation (default: 0.0)')
    parser.add_argument('--scale', type=float, default=0.0, metavar='J',
                        help='random scale augmentation (default: 0.0)')
    
    parser.add_argument('--xs', action='store_true', default=False,
                        help='use small dataset subset for debugging')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--workers', type=int, default=4, metavar='W',
                        help='number of data workers (default: 4)')
    parser.add_argument('--sim_weight', type=float, default=0.1)
    parser.add_argument('--gpu_ids', type=str, default='0,1')

    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--load',type=str,
                        default=None,
                        help='path to pre-trained source model'
                        ) # load model from checkpoint

    parser.add_argument('--ssl', type=str, default='sim')
    parser.add_argument('--poly', type=bool, default=True)

    parser.add_argument('--model', type=str, default='resnet-50')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    main(args)


