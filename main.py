import random
import sys

from baseline import Baseline
from lib import get_epoch, visualize_topk_all, mine_cluster, visualize_topk_disk
from datasets import PartImageNetDataset, CUBDataset, CelebA, S_CAR, SYSUDataset, RandomIdentitySampler, \
    MPerClassSampler, ShapeNet_CAR, MarketDataset
from nets import PPCNet
import os
import argparse
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import Dataset
from torchvision.models import resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
import json
from torch.utils.tensorboard import SummaryWriter
from train import train, validation, train_baseline, validation_baseline
import torchvision.transforms.v2 as transforms

from util import mine_centers

torch.multiprocessing.set_sharing_strategy('file_system')
# torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser(description='PDiscoNet')
    parser.add_argument('--model_name', help='Name under which the model will be saved', required=True)
    parser.add_argument('--data_root',
                    help='directory that contains the celeba, cub, or partimagenet folder', required=True)
    parser.add_argument('--dataset', help='The dataset to use. Choose celeba, cub, or partimagenet.', required=True)
    parser.add_argument('--num_parts', help='number of parts to predict', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=448, type=int) # 256 for celeba, 448 for cub,  224 for partimagenet
    parser.add_argument('--p_size', default=8, type=int)
    parser.add_argument('--epochs', default=20, type=int) # 15 for celeba, 28 for cub, 20 for partimagenet
    parser.add_argument('--resume', '-r', default='', help='If you want to load a pretrained model,'
                        'specify the path to the model here.')
    parser.add_argument('--save_figures', default=False,
                        help='Whether to save the attention maps to png', action='store_true')
    parser.add_argument('--only_test', default=False, action='store_true', help='Whether to only test the model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(log_dir=f'{args.dataset}/{args.model_name}')
    writer.add_text('Dataset', args.dataset.lower())
    writer.add_text('Device', str(device))
    writer.add_text('Learning rate', str(args.lr))
    writer.add_text('Batch size', str(args.batch_size))
    writer.add_text('Epochs', str(args.epochs))
    writer.add_text('Number of parts', str(args.num_parts))

    with open(f'{args.dataset}/{args.model_name}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    np.random.seed(1)
    data_path = args.data_root + '/' + args.dataset
    train_loader = None
    if args.dataset.lower() == 'cub':
        dataset_train = CUBDataset(data_path + '/CUB_200_2011', split=1.0, mode='train', image_size=args.image_size)
        dataset_val = CUBDataset(data_path + '/CUB_200_2011', mode='test',
                                 train_samples=dataset_train.trainsamples, image_size=args.image_size)
        num_cls = 200
    elif args.dataset.lower() == 'stanfordcars':
        dataset_train = S_CAR(data_path, mode='train', image_size=args.image_size)
        dataset_val = S_CAR(data_path, mode='test', image_size=args.image_size)
        num_cls = 196

        p_size = args.p_size
        k_size = args.batch_size // p_size
        # sampler = MPerClassSampler(dataset_train, p_size * k_size, k_size)

        # train_loader = torch.utils.data.DataLoader(dataset_train, args.batch_size, sampler=sampler, drop_last=True, pin_memory=True,
        #                               num_workers=8)
        # next(train_loader.sampler)
    elif args.dataset.lower() == 'market':
        class RandomGrayscale:
            def __init__(self) -> None:
                pass
            def __call__(self, x):
                op = random.randint(0, 10)

                if op == 0:
                    x[1] = x[2] = x[0]
                elif op == 1:
                    x[0] = x[2] = x[1]
                elif op == 2:
                    x[0] = x[1] = x[2]
                return x

        image_size = (400,400//2)
        train_transforms = transforms.Compose([
            transforms.Resize(size=image_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(0.05),
            transforms.RandomAffine(degrees=5, translate=(0.01, 0.05), scale=(0.9, 1.)),
            transforms.RandomCrop(image_size),
            transforms.ToDtype(torch.float32, scale=True)
        ])
        dataset_train = MarketDataset(data_path, mode='train', transform=train_transforms)

        transform_test = transforms.Compose([
            transforms.Resize(image_size,antialias=True),
            transforms.ToDtype(torch.float32, scale=True)
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_val = MarketDataset(data_path, mode='train', transform=transform_test)
        num_cls = len(np.unique(dataset_train.ids))

        p_size = args.p_size
        k_size = args.batch_size // p_size
        sampler = MPerClassSampler(dataset_train, p_size * k_size, k_size)

        # train_loader = torch.utils.data.DataLoader(dataset_train, args.batch_size, sampler=sampler, drop_last=True, pin_memory=True,
        #                                num_workers=8)
        # next(train_loader.sampler)

    else:
        raise RuntimeError("Choose stanfordCars, cub, or market as dataset")

    if not train_loader :
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                               num_workers=8)

    project_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=False,
                                               num_workers=8)

    test_batch = args.batch_size
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=test_batch, shuffle=True, num_workers=4)

    weights = ResNet50_Weights.DEFAULT
    basenet = resnet50(weights=weights)

    net = PPCNet(basenet, args.num_parts, num_classes=num_cls)
    # net = Baseline(num_classes=num_cls)
    if not os.path.exists(f'./results/results_{args.model_name}'):
        os.makedirs(f'./results/results_{args.model_name}')

    if args.resume:
        chk = torch.load(args.resume)
        if 'C_centers' in chk:
            net.register_buffer('C_centers', chk['C_centers'])
            # net.register_buffer('C_labels', labels)
            net.register_buffer('W_c', chk['W_c'])
            net.register_buffer('C_ind', chk['C_ind'])
            net.register_buffer('C_sec', chk['C_sec'])
            net.compute_Clabel()
        net.load_state_dict(chk)
        print(f'model loade from {args.resume}')

    net.to(device)
    epoch_leftoff = 0

    if args.only_test:
        args.epochs = 1

    all_losses = []

    high_lr_layers = ["modulation", "centerLoss", "partCenter"]
    med_lr_layers = ["fc_class_landmarks", "proto_cls", "proto_enc", "proto_concept", "global_enc", "global_cls", "proto_cluster", "concept_cls"]

    # First entry contains parameters with high lr, second with medium lr, third with low lr
    param_dict = [{'params': [], 'lr': args.lr * 100},
                  {'params': [], 'lr': args.lr * 10},
                  {'params' : [], 'lr': args.lr}]
    for name, p in net.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name in high_lr_layers:
            param_dict[0]['params'].append(p)
        elif layer_name in med_lr_layers:
            param_dict[1]['params'].append(p)
        else:
            param_dict[2]['params'].append(p)
    optimizer = torch.optim.Adam(params=param_dict)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss_hyperparams = {'l_class': 2, 'l_pres': 2, 'l_equiv': 1, 'l_conc': 2000, 'l_orth': 1}

    if args.dataset.lower() == 'cub':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    elif args.dataset.lower() in ['stanfordcars', 'market' ]:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    train_loader.namesToConcept = None
    train_loader.cluster_labels = None
    epoch_leftoff = 0
    losses = {}
    net.epoch = (0)

    # validation(device, net, val_loader, 0, args.model_name, args.save_figures, writer)
    for epoch in range(epoch_leftoff, args.epochs):
        net.epoch = (epoch - epoch_leftoff)
        # if epoch >= 0:
        #     cluster_labels = mine_cluster(device, net, train_loader, epoch, args.model_name)
        #     train_loader.cluster_labels = cluster_labels
        if not args.only_test:
            # validation(device, net, val_loader, epoch, args.model_name, args.save_figures, writer)
            if epoch % 4 == 0 and epoch >= 10:
                # namesToConcept = mine_concept(device, net, train_loader, epoch, args.model_name)
                centers, sec, W_c, ind = mine_centers(device, net, project_loader)
                # centers, labels, W_c, ind = torch.load('centers.pth', device),None, torch.load('W_c.pth', device), torch.load('ind.pth', device)
                if "C_centers" in net._buffers:
                    net.C_centers = centers
                    net.W_c = W_c
                    net.C_ind = ind
                    net.C_sec = sec
                else:
                    net.register_buffer('C_centers', centers)
                    net.register_buffer('C_sec', sec)
                    net.register_buffer('W_c', W_c)
                    net.register_buffer('C_ind', ind)
                net.compute_Clabel()

                print(f"{net.C_centers.shape[0]} concepts are set!")
                torch.save(net.state_dict(), f'./{args.dataset}/{args.model_name}.pt')
                # net.C_centers = centers
                # net.C_labels = labels
                # net.W_c = W_c
                # net.C_ind = ind
            #     train_loader.namesToConcept = namesToConcept
            # train_baseline(net, optimizer, train_loader, device, epoch, loss_fn, writer)

            net, losses = train(net, optimizer, train_loader, device, epoch, 0, loss_fn,
                                loss_hyperparams, writer, losses)
            scheduler.step()
            validation(device, net, val_loader, epoch, args.model_name, args.save_figures, writer)
            # validation_baseline(device, net, val_loader, epoch, args.model_name, args.save_figures, writer)
            torch.cuda.empty_cache()
        # Validation
        else:
            print('Validation accuracy with saved network:')
            # validation(device, net, val_loader, epoch, args.model_name, args.save_figures, writer)
            # visualize_topk_all(val_loader, model_name=args.model_name, net=net, device=device)
            visualize_topk_disk(val_loader, model_name=args.model_name, net=net, device=device)
            sys.exit(0)
        torch.save(net.state_dict(), f'./{args.dataset}/{args.model_name}.pt')
    writer.close()

if __name__ == "__main__":
    main()
