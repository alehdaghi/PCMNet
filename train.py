"""
Contains functions used for training and testing
"""


# Import statements
import torch
import numpy as np
from kmeans_pytorch import kmeans

from lib import rigid_transform, landmark_coordinates, save_maps, vis_pred, vis_cluster, update_losses, \
    batch_rigid_transform
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import einops
from losses import center_loss, TripletLoss, equiv_loss, conc_loss, separation_loss
import os
import torchvision
# gray_transform = torchvision.transforms.ColorJitter(brightness=.4, hue=.2)
gray_transform = torchvision.transforms.RandomGrayscale(p=0.4)
triplet_loss = TripletLoss()
# Function definitions




def train(net: torch.nn.Module, optimizer: torch.optim, train_loader: torch.utils.data.DataLoader,
          device: torch.device, epoch: int, epoch_leftoff: int, loss_fn: torch.nn.Module, loss_hyperparams: dict,
          writer: torch.utils.tensorboard.SummaryWriter, losses: dict = None) -> (torch.nn.Module, [float]):
    """
    Model trainer, saves losses to file
    Parameters
    ----------
    net: torch.nn.Module
        The model to train
    optimizer: torch.optim
        Optimizer used for training
    train_loader: torch.utils.data.DataLoader
        Data loader for the training set
    device: torch.device
        The device on which the network is trained
    epoch: int
        Current epoch, used for the running loss
    epoch_leftoff: int
        Starting epoch of the training function, used if a training run was
        stopped at e.g. epoch 10 and then later continued from there
    loss_fn: torch.nn.Module
        Loss function
    loss_hyperparams: dict
        Indicates, per loss, its hyperparameter
    writer: torch.utils.tensorboard.SummaryWriter
        The object to write performance metrics to
    all_losses: [float]
        The list of all running losses, used to display (not backprop)
    Returns
    ----------
    net: torch.nn.Module
        The model with updated weights
    all_losses: [float]
        The list of all running losses, used to display (not backprop)
    """
    # Training
    # if all_losses:
    #     running_loss_conc, running_loss_pres, running_loss_class, running_loss_equiv, running_loss_orth, running_loss_sim, running_loss_class_g = all_losses
    # elif not all_losses and epoch != 0:
    #     print(
    #         'Please pass the losses of the previous epoch to the training function')
    net.train()

    pbar = tqdm(total=len(train_loader), position=0, leave=True, mininterval=1)
    top_class = []
    l_class = loss_hyperparams['l_class']
    l_pres = loss_hyperparams['l_pres']
    l_conc = loss_hyperparams['l_conc']
    l_orth = loss_hyperparams['l_orth']
    l_equiv = loss_hyperparams['l_equiv']
    for i, (X, lab, index) in enumerate(train_loader):
        lab = lab.to(device)
        landmark_features, maps, scores, p_score, proto_features, G = net(X.to(device), y=lab)
        # loss_sim = clustering_loss3(p_feats=landmark_features.permute(0,2,1)[:,:-1], cluster_assignments=G['cluster_assign'])#similarity_loss(proto_features, lab)
        loss_sim = torch.tensor([0.0], requires_grad=True, device=device)


        # Equivariance loss: calculate rotated landmarks distance
        loss_equiv = equiv_loss(X, maps, net, device, net.num_landmarks, G, epoch, landmark_features.permute(0,2,1)) * l_equiv

        # Classification loss
        loss_class_g = loss_fn(scores, lab).mean() * l_class * 0.5

        p_loss_class = loss_fn(p_score, lab).mean()
        loss_class = p_loss_class * l_class


        # Classification accuracy
        preds = scores.argmax(dim=1)
        top_class.append((preds == lab).float().mean().cpu())

        # Get landmark coordinates
        loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps, device)

        # Concentration loss
        loss_conc = conc_loss(loc_x, loc_y, grid_x, grid_y, maps) * l_conc
        # loss_conc = torch.tensor([0.0], requires_grad=True, device=device)

        # Presence loss
        loss_pres = torch.nn.functional.avg_pool2d(maps[:, :-1, 2:-2, 2:-2], 3, stride=1).max(-1)[0].max(-1)[0].max(0)[0].mean()
        loss_pres = (1 - loss_pres) * l_pres
        # loss_pres = torch.tensor([0.0], requires_grad=True, device=device)

        # Orthogonality loss
        # loss_orth = orth_loss(net.num_landmarks, landmark_features, device) * l_orth

        # Center Part luster loss
        loss_orth = center_loss(landmark_features, net) * l_orth

        # Explain loss
        loss_X = separation_loss(maps) * 0.5

        loss_class_badParts = torch.tensor([0.0], requires_grad=True, device=device)#loss_fn(G['c_score'], lab).mean() * l_class / 2
        W_p = net.proto_cls.weight
        A_p = torch.stack([F.linear(proto_features[:, i], W_p[:, i * 256:i * 256 + 256]) for i in range(net.num_landmarks)], dim=1)
        if len(net.badPart) >0:
            loss_class_badParts = F.cross_entropy(A_p[:,net.badPart].mean(dim=1), lab)

        if 'C_centers' in net._buffers and  net.C_centers != None and 'c_score' in G:
            # loss_class_concept = concept_mining_loss(net, part_pooled=proto_features, Feat=landmark_features.permute(0, 2, 1),  labels=lab)
            loss_class_concept = F.cross_entropy(G['c_score'], lab)
        else:
            loss_class_concept = torch.tensor([0.0], requires_grad=True, device=device)



        total_loss = (loss_equiv + (loss_conc + loss_pres + loss_orth  + loss_class)  +
                      loss_sim + loss_class_badParts + 2*loss_class_concept )

        if epoch <= 5 :
            total_loss += loss_X
        if epoch >= 12:
            total_loss+= loss_class_g

        update_losses(losses, 'loss_equiv', loss_equiv.item())
        update_losses(losses, 'loss_conc',  loss_conc.item())
        update_losses(losses, 'loss_pres',  loss_pres.item())
        update_losses(losses, 'loss_orth',  loss_orth.item())
        update_losses(losses, 'loss_X',   loss_X.item())
        update_losses(losses, 'loss_class', loss_class.item())
        update_losses(losses, 'loss_class_g', loss_class_g.item())
        update_losses(losses, 'loss_class_badParts', loss_class_badParts.item())
        update_losses(losses, 'loss_concept',   loss_class_concept.item())

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # torch.cuda.empty_cache()

        pbar.set_description(
            (
                f"e:{epoch + 1}; l: {total_loss.item():.2f}; "
                f'co:{loss_conc.item():.2f}({losses["running_loss_conc"]:.2f});'
                f'pr:{loss_pres.item():.2f}({losses["running_loss_pres"]:.2f});'
                f'ce:{loss_class.item():.2f}({losses["running_loss_class"]:.2f});'
                f'cg:{loss_class_g.item():.2f}({losses["running_loss_class_g"]:.2f});'
                f'eq:{loss_equiv.item():.2f}({losses["running_loss_equiv"]:.2f}); '
                f'or:{loss_orth.item():.2f}({losses["running_loss_orth"]:.2f});'
                f'cc:{loss_class_concept.item():.2f}({losses["running_loss_concept"]:.2f});'
                f'bc:{loss_class_badParts.item():.2f}({losses["running_loss_class_badParts"]:.2f}); '
                f'X:{loss_X.item():.2f}({losses["running_loss_X"]:.2f}); '


            )
        )
        pbar.update()

    top1acc = np.mean(np.array(top_class))
    # writer.add_scalar('Concentration loss', running_loss_conc, epoch)
    # writer.add_scalar('Presence loss', running_loss_pres, epoch)
    # writer.add_scalar('Classification loss', running_loss_class, epoch)
    # writer.add_scalar('Equivariance loss', running_loss_equiv, epoch)
    # writer.add_scalar('Orthogonality loss', running_loss_orth, epoch)
    # writer.add_scalar('Training Accuracy', top1acc, epoch)
    torch.cuda.empty_cache()
    pbar.close()

    writer.flush()
    return net, losses


def validation(device, net, val_loader, epoch, model_name, save_figures, writer):
    """
    Calculates validation accuracy for trained model, writes it to Tensorboard Summarywriter.
    Also saves figures with attention maps if save_figures is set to True.
    Parameters
    ----------
    device: torch.device
        The device on which the network is loaded
    net: torch.nn.Module
        The model to evaluate
    val_loader: torch.utils.data.DataLoader
        Data loader for the validation set
    epoch: int
        Current epoch, used to save results
    model_name: str
        Name of the model, used to save results
    save_figures: bool
        Whether to save the attention maps
    writer: torch.utils.tensorboard.SummaryWriter
        The object to write metrics to
    """
    net.eval()
    net.to(device)
    pbar = tqdm(val_loader, position=0, leave=True)
    top_class = []
    all_scores, all_scores_g = [], []
    all_labels = []
    all_maxes = torch.Tensor().to(device)
    XX = {}
    all_scores_c = []
    with torch.no_grad():
        for i, (X, y, _) in enumerate(tqdm(val_loader)):

            feats, maps, scores_g, _scores, p_feats, G = net(X.to(device))
            c_score = G['c_score']


            # if save_figures:
            #     if 'X' not in XX:
            #         XX['X'] = X.detach().cpu()
            #         XX['y'] = y.detach().cpu()
            #         XX['p_feats'] = p_feats.detach().cpu()
            #         XX['maps'] = maps.detach().cpu()
            #     else:
            #         XX['X'] = torch.cat([XX['X'], X.detach().cpu()], dim=0)
            #         XX['y'] = torch.cat([XX['y'], y.detach().cpu()], dim=0)
            #         XX['p_feats'] = torch.cat([XX['p_feats'], p_feats.detach().cpu()], dim=0)
            #         XX['maps'] = torch.cat([XX['maps'], maps.detach().cpu()], dim=0)

            _scores = _scores.detach().cpu()
            scores_g = scores_g.detach().cpu()
            all_scores.append(_scores.argmax(dim=-1))
            all_scores_g.append(scores_g.argmax(dim=-1))
            if c_score is not None:
                all_scores_c.append(c_score.cpu().argmax(dim=-1))
            lab = y
            all_labels.append(lab)

            # for j in range(scores.shape[0]):
            #     probs = _scores[j, :].softmax(dim=0).cpu()
            #     # probs = _scores[j, :].softmax(dim=1).sum(dim=0).cpu()
            #     preds = torch.argmax(probs, dim=-1).cpu()
            #     top_class.append(1 if preds == lab[j].cpu() else 0)

            map_max = maps.max(-1)[0].max(-1)[0][:, :-1].detach()
            all_maxes = torch.cat((all_maxes, map_max), 0)
            # vis_cluster(X, p_feats, maps, epoch, model_name, device, y, scores, iteration=i)
            # Saving the attention maps\
            if save_figures and i % 100 == 0:
                # if i % 300 == 0:
                #     vis_pred(X, maps, epoch, model_name, device, y, scores, iteration=i)
                save_maps(X, maps, epoch, model_name, device, i // 100)

            # if save_figures and i % 40 == 10:
            #     # vis_cluster(XX['X'], p_feats=XX['p_feats'], maps=XX['maps'], model_name=model_name, y=XX['y'], iteration=i)
            #     visualize_topk(XX['X'], p_feats=XX['p_feats'], maps=XX['maps'], model_name=model_name, y=XX['y'], iteration=i, net=net)
            #     XX = {}
            # if save_figures:
                # if XX['X'].shape[0] > 300:
                #         XX = {}

    top1acc = (torch.cat(all_scores) == torch.cat(all_labels)).float().mean()
    top1acc_g = (torch.cat(all_scores_g) == torch.cat(all_labels)).float().mean()
    top1acc_c = torch.zeros([1])
    if len(all_scores_c)> 0:
        top1acc_c = (torch.cat(all_scores_c) == torch.cat(all_labels)).float().mean()
    writer.add_scalar('Validation Accuracy', top1acc, epoch)
    pbar.close()
    print(f'Validation Accuracy in epoch {epoch}: p:{top1acc.item():.3f} g:{top1acc_g.item():.3f} c:{top1acc_c.item():.3f}')
    writer.flush()



def train_baseline(net: torch.nn.Module, optimizer: torch.optim, train_loader: torch.utils.data.DataLoader,
          device: torch.device, epoch: int, loss_fn: torch.nn.Module,
          writer: torch.utils.tensorboard.SummaryWriter) -> (torch.nn.Module, [float]):


    net.train()
    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    sum = 0 ;
    for i, (X, lab, index) in enumerate(train_loader):
        lab = lab.to(device)
        scores = net(X.to(device))
        # Classification loss
        total_loss = loss_fn(scores, lab).mean()
        sum += total_loss.item()
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        pbar.set_description(
            (
                f"e:{epoch + 1}; l: {total_loss.item():.3f}({sum/(i+1):.3f}); "
            )
        )
        pbar.update()

    torch.cuda.empty_cache()
    pbar.close()
    writer.flush()


def validation_baseline(device, net, val_loader, epoch, model_name, save_figures, writer):
    """
    Calculates validation accuracy for trained model, writes it to Tensorboard Summarywriter.
    Also saves figures with attention maps if save_figures is set to True.
    Parameters
    ----------
    device: torch.device
        The device on which the network is loaded
    net: torch.nn.Module
        The model to evaluate
    val_loader: torch.utils.data.DataLoader
        Data loader for the validation set
    epoch: int
        Current epoch, used to save results
    model_name: str
        Name of the model, used to save results
    save_figures: bool
        Whether to save the attention maps
    writer: torch.utils.tensorboard.SummaryWriter
        The object to write metrics to
    """
    net.eval()
    net.to(device)
    pbar = tqdm(val_loader, position=0, leave=True)
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for i, (X, y, _) in enumerate(tqdm(val_loader)):
            scores = net(X.to(device))
            scores = scores.detach().cpu()
            all_scores.append(scores.argmax(dim=-1))
            lab = y
            all_labels.append(lab)

    top1acc = (torch.cat(all_scores) == torch.cat(all_labels)).float().mean()
    writer.add_scalar('Validation Accuracy', top1acc, epoch)
    pbar.close()
    print(f'Validation Accuracy in epoch {epoch}: p:{top1acc.item():.3f}')
    writer.flush()



if __name__ == "__main__":
    pass
