import torch
import numpy as np
from kmeans_pytorch import kmeans
from sklearn.cluster import DBSCAN

from lib import rigid_transform, landmark_coordinates, save_maps, vis_pred, vis_cluster, update_losses
import torch.nn.functional as Fn
from tqdm import tqdm
import umap

def find_centers(A_p, Feat, P):
    centroids = [[[] for i in range(A_p.shape[-1])] for j in P]
    reducer = umap.UMAP(metric="cosine", min_dist=0.15, n_neighbors=9, random_state=0, n_jobs=1)
    O_max, O_arg = A_p.sum(dim=1).max(dim=1)
    cluster = DBSCAN(eps=0.7, min_samples=3, metric='euclidean', n_jobs=5)
    for ppp in P:
        cc = 0
        p_bar = tqdm(range(A_p.shape[-1]), desc=f'Clustering {ppp}')
        for j in p_bar:

            I = (O_arg == j).nonzero().squeeze()
    #             conf = (A_p[I,pp].max(dim=1)[0] /A_p_max[yi]) > 0.08
            conf = (A_p[I,ppp,j] /O_max[I]) > 0.05
            I = I[conf]
#             print(j, conf.sum())
            if conf.sum() < 10:
                continue
            embedding = reducer.fit_transform(Feat[I,ppp,:].detach().cpu().numpy())

            L = cluster.fit_predict(embedding)#cluster.fit_predict(p_feats[conf][:, :, 5].detach().cpu().numpy())
            uniq_l = np.unique(L)
            for l in uniq_l:
                if l == -1 :
                    continue
                II = I[L==l]
                AA = A_p[II,ppp,j]
                values, ind = torch.topk(AA, min(len(AA), 16), dim=0)
                ind = II[ind]
                centroids[ppp][j].append(ind)
            cc +=1

        print(f"{ppp}'Size: {cc} ")
    return centroids

def mine_centers(device, net, project_loader, epoch=0, model_name=""):
    # img_iter = tqdm(enumerate(project_loader), total=len(project_loader), desc='Extracting feats', ncols=0)
    part_pooled, maps, Y, Feat, indices  = getFeats(device, net, project_loader)

    N = len(indices)
    w = net.proto_cls.weight
    p = part_pooled.shape[1]
    A_p = torch.stack([Fn.linear(part_pooled[:, i], w[:, i * 256:i * 256 + 256].cpu()) for i in range(p)], dim=1)
    N_c = A_p.shape[-1]

    centroids = find_centers(A_p, Feat, range(p))
    centers = []
    labels = []
    sec = []
    badPart = []

    ind = []
    cc, cc2, cc3, cc5 = 0, 0, 0, 0
    maxCluster = 0
    for pp in range(p):
        labels.append([])
        concept = 0
        for j in range(A_p.shape[-1]):
            labels[pp].append([])
            maxCluster = max(maxCluster, len(centroids[pp][j]))
            sizeCluster = len(centroids[pp][j])
            concept += (sizeCluster > 0)
            for cc4, ii in enumerate(centroids[pp][j]):

                centers.append(part_pooled[ii, pp, :].mean(dim=0))
                labels[pp][j].append(cc)
                cc += 1
                ind.append([cc2,cc3, cc4, cc5, cc5+sizeCluster]) # part_index, part_index*class_index, cluster_index, startIndex, endIndex
            cc3+=1
            cc5 += sizeCluster
            if sizeCluster > 0:
                sec.append(sizeCluster)
        cc2 += 1
        if concept < N_c * 0.5 :
            badPart.append(pp)
        # print(pp, concept)
    w_c = torch.zeros(A_p.shape[-1], cc)
    net.badPart = badPart
    # print("Re-initializing bad parts", badPart)
    # for bad in badPart:
    #     net.resetPart(bad)
    for pp in range(p):
        for j in range(A_p.shape[-1]):
            for i in labels[pp][j]:
                w_c[j, i] = 1
    print(f"{cc} concepts are found!")

    return torch.stack(centers).to(device), torch.Tensor(sec).to(device).long(), w_c.to(device), torch.Tensor(ind).to(device).long()


def getFeats(device, net, loader):
    net = net.eval()
    img_iter = tqdm(enumerate(loader), total=len(loader), desc='Collecting features', ncols=0)
    maps, pooled, Y, Feat, = [], [], [], []
    # g_feat, g_patch, XG = [], [], []

    indices = []
    H, W = 0, 0
    net.eval()
    with torch.no_grad():
        for i, (Xs, ys, index) in img_iter:
            indices.extend(index.numpy())
            batch_size, _, H, W = Xs.shape
            feats, m, scores, _scores, p_feats, G = net(Xs.to(device))
            # g_feat.append(G["g_feat"].detach().cpu())
            # g_patch.append(G["g_index"].detach().cpu())
            # XG.append(G["XG"].detach().cpu())
            Y.append(ys.detach().cpu())
            maps.append(m.detach().cpu())
            pooled.append(p_feats.detach().cpu())
            Feat.append(feats.detach().cpu())
    N = len(indices)
    #     p, d = 8, 256
    #     part_pooled = pooled.reshape(-1, p, d)
    return torch.cat(pooled, dim=0), torch.cat(maps, dim=0), torch.cat(Y, dim=0), \
        torch.cat(Feat, dim=0).permute(0, 2, 1), np.asarray(indices)