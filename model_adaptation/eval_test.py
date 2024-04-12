"""
Author:
Hiroki Azuma, ornitho1027@gmail.com

Our codes heavily rely on https://github.com/Red-Fairy/ZeroShotDayNightDA by Rundong Luo, rundongluo2002@gmail.com.
We really appreciate the awesome codes.
"""

import os
from utils.helpers import get_test_trans
from utils.routines import evaluate, eval_evaluate
from datasets.cityscapes_ext import CityscapesExt
from datasets.gta5 import GTA5DataSet
from datasets.acdc import ACDC
from models.segmentation.modeling import deeplabv3plus_resnet50, deeplabv3plus_resnet101, refinenet_resnet50

import torch
import torch.nn as nn

def main(args):

    print(args)

    # Define data transformation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    target_size = (512,1024)
    test_trans = get_test_trans(mean, std, target_size)

    # Load dataset
    cs_path = args.cs_path
    acdc_path = args.acdc_path
    gta5_path = args.gta5_path

    testset_day = CityscapesExt(cs_path, split='val', target_type='semantic', transforms=test_trans)
    testset_fog = ACDC(acdc_path, split='val', transforms=test_trans, ACDC_sub='fog')
    testset_rain = ACDC(acdc_path, split='val', transforms=test_trans, ACDC_sub='rain')
    testset_night = ACDC(acdc_path, split='val', transforms=test_trans, ACDC_sub='night')
    testset_snow = ACDC(acdc_path, split='val', transforms=test_trans, ACDC_sub='snow')
    testset_gta5 = GTA5DataSet(gta5_path, list_path="./datasets/gta5_list/gtav_split_val.txt", transforms=test_trans)

    dataloaders = {}
    dataloaders['test_day'] = torch.utils.data.DataLoader(testset_day, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_fog'] = torch.utils.data.DataLoader(testset_fog, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_rain'] = torch.utils.data.DataLoader(testset_rain, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_night'] = torch.utils.data.DataLoader(testset_night, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_snow'] = torch.utils.data.DataLoader(testset_snow, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    dataloaders['test_gta5'] = torch.utils.data.DataLoader(testset_gta5, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    num_classes = len(CityscapesExt.validClasses)

    # Define model, loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=CityscapesExt.voidClass)

    if args.arch == "deeplabv3+":
        if args.model == "resnet50":
            model = deeplabv3plus_resnet50(num_classes=num_classes, backbone_pretrained=False)
        elif args.model == "resnet101":
            model = deeplabv3plus_resnet101(num_classes=num_classes, backbone_pretrained=False)
        else:
            raise ValueError
    elif args.arch == "refinenet":
        if args.model == "resnet50":
            model = refinenet_resnet50(num_classes=num_classes, backbone_pretrained=False)
        elif args.model == "resnet101":
            raise NotImplementedError
        else:
            raise ValueError

    # Push model to GPU
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))

    # Load weights from checkpoint
    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Validate
    print('--- Validation - daytime ---')
    val_acc_cs, val_loss_cs, miou_cs, confmat_cs, iousum_cs = evaluate(dataloaders['test_day'],
        model, criterion, 0, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std)
    print(miou_cs)
    
    print('--- Validation - ACDC Night ---')
    test_acc_night, test_loss_night, miou_night, confmat_night, iousum_night = eval_evaluate(dataloaders['test_night'],
        model, criterion, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std, save_root=os.path.join(args.save_path, 'night'), save=args.save)
    print(miou_night)
    
    print('--- Validation - ACDC Snow ---')
    test_acc_snow, test_loss_snow, miou_snow, confmat_snow, iousum_snow = eval_evaluate(dataloaders['test_snow'],
        model, criterion, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std, save_root=os.path.join(args.save_path, 'snow'), save=args.save)
    print(miou_snow)
    
    print('--- Validation - ACDC Rain ---')
    test_acc_rain, test_loss_rain, miou_rain, confmat_rain, iousum_rain = eval_evaluate(dataloaders['test_rain'],
        model, criterion, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std, save_root=os.path.join(args.save_path, 'rain'), save=args.save)
    print(miou_rain)

    print('--- Validation - ACDC Fog ---')
    test_acc_fog, test_loss_fog, miou_fog, confmat_fog, iousum_fog = eval_evaluate(dataloaders['test_fog'],
        model, criterion, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std, save_root=os.path.join(args.save_path, 'fog'), save=args.save)
    print(miou_fog)

    print('--- Validation - GTA5 ---')
    test_acc_gta5, test_loss_gta5, miou_gta5, confmat_gta5, iousum_gta5 = eval_evaluate(dataloaders['test_gta5'],
        model, criterion, CityscapesExt.classLabels, CityscapesExt.validClasses,
        void=CityscapesExt.voidClass, maskColors=CityscapesExt.maskColors, mean=mean, std=std, save_root=os.path.join(args.save_path, 'gta5'), save=args.save)
    print(miou_gta5)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Segmentation training and evaluation')
    parser.add_argument('--cs_path', type=str, required=True)
    parser.add_argument('--acdc_path', type=str, required=True)
    parser.add_argument('--gta5_path', type=str, required=True)

    parser.add_argument('--weight', type=str, required=True,
                        help='load weight file')
    parser.add_argument('--model', type=str, default='refinenet',
                        help='model (refinenet or deeplabv3)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 3)')
    parser.add_argument('--workers', type=int, default=4, metavar='W',
                        help='number of data workers (default: 4)')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--save_path', default='',type=str)
    parser.add_argument('--save', action='store_true', help='save visual results')

    parser.add_argument('--model', type=str, default='resnet-50')
    parser.add_argument('--arch', type=str, default='deeplabv3+')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    main(args)
