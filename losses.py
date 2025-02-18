import torch
import torch.nn as nn
import einops
import torch.nn.functional as Fn
import numpy as np
import torchvision

from lib import batch_rigid_transform

gray_transform = torchvision.transforms.RandomGrayscale(p=0.4)

class MarginalCenterLoss(nn.Module):

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(MarginalCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, M1=1.5, M2=0.3):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
        #           torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat.addmm_(1, -2, x, self.centers.t())
        distmat = torch.cdist(x, self.centers)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]
        ind = dist > M2
        loss = dist[ind].mean()
        if torch.isnan(loss):
            loss =  torch.tensor([0.0], requires_grad=True, device=x.device)
        cenDis = torch.cdist(self.centers, self.centers)[~torch.eye(self.num_classes).bool()]
        # cenDis = Fn.cosine_similarity(self.centers[:,None], self.centers[None,:],dim=-1).abs()[~torch.eye(self.num_classes).bool()]
        loss_orth = (M1 - cenDis[cenDis < M1]).mean()
        # loss_orth = (cenDis).mean()
        if torch.isnan(loss_orth):
            loss_orth =  torch.tensor([0.0], requires_grad=True, device=x.device)
        # normed_feature = torch.nn.functional.normalize(self.centers, dim=1)
        # similarity = torch.matmul(normed_feature, normed_feature.t())
        # similarity = torch.sub(similarity, torch.eye(self.num_classes).to(x.device))
        # loss_orth = torch.mean(torch.square(similarity))

        return loss + loss_orth


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.cdist(x, self.centers)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class PartCenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, num_parts, feat_dim=2, use_gpu=True):
        super(PartCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.num_parts = num_parts
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.centers = CenterLoss(num_classes * num_parts , feat_dim, use_gpu)



    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        part_labels = einops.repeat(labels, 'b -> b p', p = self.num_parts) * self.num_parts + torch.arange(self.num_parts).cuda()
        part_x = einops.rearrange(x, 'b p d -> (b p) d')
        part_labels = einops.rearrange(part_labels, 'b p ... -> (b p) ...')

        loss = self.centers(part_x, part_labels)

        return loss


def conc_loss(centroid_x: torch.Tensor, centroid_y: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor,
              maps: torch.Tensor) -> torch.Tensor:
    """
    Calculates the concentration loss, which is the weighted sum of the squared distance of the landmark
    Parameters
    ----------
    centroid_x: torch.Tensor
        The x coordinates of the map centroids
    centroid_y: torch.Tensor
        The y coordinates of the map centroids
    grid_x: torch.Tensor
        The x coordinates of the grid
    grid_y: torch.Tensor
        The y coordinates of the grid
    maps: torch.Tensor
        The attention maps

    Returns
    -------
    loss_conc: torch.Tensor
        The concentration loss
    """
    spatial_var_x = ((centroid_x.unsqueeze(-1).unsqueeze(-1) - grid_x) / grid_x.shape[-1]) ** 2
    spatial_var_y = ((centroid_y.unsqueeze(-1).unsqueeze(-1) - grid_y) / grid_y.shape[-2]) ** 2
    spatial_var_weighted = (spatial_var_x + spatial_var_y) * maps
    loss_conc = spatial_var_weighted[:, 0:-1, :, :].mean()
    return loss_conc


from losses import MarginalCenterLoss




def similarity_loss(p_feats, labels) -> torch.Tensor:
    return p_feats.mean()
    b, K, d = p_feats.shape
    feat = einops.rearrange(p_feats, 'b k d->  k b d')

    idx = labels[:, None].eq(labels)
    num_samples = idx[0].sum().item()
    p_size = b // num_samples

    z = einops.rearrange(p_feats, '(p n) ...->  p n ...', p = p_size, n= num_samples)
    y = einops.rearrange(labels, '(p n) ...->  p n ...', p = p_size, n= num_samples)
    loss = 0
    for k in range(K):
        for i in range(num_samples):
            for j in range(num_samples):
                if i != j:
                    l, s = contrastive_loss(z[:,i, k], z[:,j, k], tau=0.2, hyp_c=0.1)
                    loss += l

    # dis = torch.cdist(feat, feat) # similarity of the same parts between samples
    # idx = labels[:, None].eq(labels)
    # disN = dis * ~idx
    # disN = disN.sum(dim=-1)/disN.count_nonzero(dim=-1)
    # idx.fill_diagonal_(False)
    # disP = dis * idx
    # disP = disP.sum(dim=-1) / disP.count_nonzero(dim=-1)
    #
    # # idx = dis > 0.1
    # # if dis.shape[-1] == 0 or idx.sum() == 0:
    # #     return dis[idx].sum()
    # # return dis[idx].mean()
    # return F.margin_ranking_loss(disN, disP, target=torch.ones_like(disN), margin=0.3)
    return loss/ (num_samples * num_samples * K)

def orth_loss(num_parts: int, landmark_features: torch.Tensor, device) -> torch.Tensor:
    """
    Calculates the orthogonality loss, which is the mean of the cosine similarities between every pair of landmarks
    Parameters
    ----------
    num_parts: int
        The number of landmarks
    landmark_features: torch.Tensor, [batch_size, feature_dim, num_landmarks + 1 (background)]
        Tensor containing the feature vector for each part
    device: torch.device
        The device to use
    Returns
    -------
    loss_orth: torch.Tensor
        The orthogonality loss
    """
    normed_feature = torch.nn.functional.normalize(landmark_features, dim=1)
    similarity = torch.matmul(normed_feature.permute(0, 2, 1), normed_feature)
    similarity = torch.sub(similarity, torch.eye(num_parts + 1).to(device))
    loss_orth = torch.mean(torch.square(similarity))

    # if labels is not None:
    #     sim = torch.einsum('adp, bdp -> pab', normed_feature, normed_feature) # similarity of the same parts between samples
    #     idx = labels[:, None].eq(labels)
    #     idx.fill_diagonal_(False)
    #     f = torch.square(sim[:, idx])
    #     if f.shape[-1] == 0:
    #         return loss_orth
    #
    #     ff = sim * idx
    #     mm = ff.sum(dim=-1) / idx.sum(dim=1)
    #     # loss_sim = torch.mean(f)
    #     ss = similarity.mean(dim=-1)
    #
    #     loss_orth = loss_orth + MarginLoss(mm, ss.t(), torch.ones_like(mm))

    return loss_orth


def equiv_loss(X: torch.Tensor, maps: torch.Tensor, net: torch.nn.Module, device: torch.device, num_parts: int, G: dict, epoch: int, p_feats) \
        -> torch.Tensor:
    """
    Calculates the equivariance loss, which we calculate from the cosine similarity between the original attention map
    and the inversely transformed attention map of a transformed image.
    Parameters
    ----------
    X: torch.Tensor
        The input image
    maps: torch.Tensor
        The attention maps
    net: torch.nn.Module
        The model
    device: torch.device
        The device to use
    num_parts: int
        The number of landmarks

    Returns
    -------
    loss_equiv: torch.Tensor
        The equivariance loss
    """
    # Forward pass

    b,_, w, h = maps.shape
    mask = torch.ones(1,1,w,h)

    angle = np.random.rand(b) * 180 - 90
    translate = np.random.rand(b,2) * 0.2 - 0.1
    scale = np.random.rand(b) * 0.5 + 0.9
    # translate2 = [(t * maps.shape[-1] / X.shape[-1]) for t in translate]
    X = gray_transform(X)

    transf_img, _ = batch_rigid_transform(X, angle, translate, scale=scale, invert=False)
    equiv_p_feats, equiv_map, _, _p, _, equiv_G = net(transf_img.to(device))

    proto_feats = G['x4']
    g_feat = G['g_feat']
    equiv_proto_feats = equiv_G['x4']
    equiv_g_feat = equiv_G['g_feat']


    # Compare to original attention map, and penalise high difference
    rot_back, maskI = batch_rigid_transform(equiv_map, angle, translate, scale=scale, invert=True, device=device)
    maskI = maskI[:,0,:,:].reshape(b, 1, -1).to(device)
    num_elements_per_map = maps.shape[-2] * maps.shape[-1]
    orig_attmap_vector = torch.reshape(maps[:, :-1, :, :], (-1, num_parts, num_elements_per_map))
    transf_attmap_vector = torch.reshape(rot_back[:, 0:-1, :, :], (-1, num_parts, num_elements_per_map))
    cos_sim_equiv = Fn.cosine_similarity(maskI*orig_attmap_vector, maskI*transf_attmap_vector, -1)
    loss_equiv = 1 - torch.mean(cos_sim_equiv)
    loss_equiv += (1 - Fn.cosine_similarity(equiv_p_feats.permute(0,2,1), p_feats, -1).mean())

    loss_equiv2 = 0

    embv1 = proto_feats.reshape(b, proto_feats.shape[1], -1)
    embv2 = equiv_proto_feats
    embv2, maskI_x4 = batch_rigid_transform(embv2, angle, translate, scale=scale, invert=True, device=device)
    embv2 = embv2.reshape_as(embv1)
    maskI_x4 = maskI_x4[:,0,:,:].reshape(b, 1, -1).to(device)
    cos_sim_equiv = (Fn.cosine_similarity(embv1.detach()*maskI_x4, embv2*maskI_x4, 1) + Fn.cosine_similarity(embv1*maskI_x4, embv2.detach()*maskI_x4, 1))/2
    loss_equiv2 = 1 - torch.mean(cos_sim_equiv)

    EPS = 1e-10
    # tanh_loss = -(torch.log(torch.tanh(torch.sum(g_feat,dim=0))+EPS).mean() + torch.log(torch.tanh(torch.sum(equiv_g_feat,dim=0))+EPS).mean())/2.
    return loss_equiv + loss_equiv2
    # if epoch > 10:
    #     return loss_equiv + loss_equiv2
    # else:
    #     return loss_equiv2
     #+ tanh_loss

def separation_loss(masks):
    b,k,w,h = masks.shape
    means = masks[:, :-1].sum(dim=[1, 2, 3]) / (w * h)
    penalties = 0.1 - means


    loss_dp = 0
    # K = masks.shape[1]
    # for i in range(K):
    #     for j in range(i + 1, K):
    #         loss_dp += ((((masks[:, i] - masks[:, j]) ** 2).sum(dim=1) / (masks.shape[-1]))).sum()
    # loss_dp = - loss_dp / (masks.shape[0] * K * (K - 1) / 2)
    return loss_dp + penalties[penalties > 0].sum()


def center_loss(landmark_features: torch.Tensor, net: torch.nn.Module):
    b,c,k = landmark_features.shape

    parts , background = landmark_features[:,:,:-1], landmark_features[:,:,-1].unsqueeze(dim=-1)
    dist = torch.cdist(parts.permute(0,2,1), background.permute(0,2,1))
    loss = (2.0 - dist[dist < 2.0]).mean() # dist to background
    if torch.isnan(loss):
        loss = 0
    parts = einops.rearrange(parts, 'b c k -> (b k) c')
    labels = einops.repeat(torch.arange(k-1), 'k -> (b k)', b=b).to(parts.device)
    return net.centerLoss(parts, labels) + loss



def kernel_divergence(K, A):
    D = torch.einsum('KnC, Knm, KmL -> KCL', A, K, A)
    diag = D.diagonal(dim1=1, dim2=2)
    DcD = (diag[:, :, None] * diag[:, None, :] + 1e-6).sqrt()
    return D / (DcD+1e-5)

def clustering_loss(p_feats, cluster_assignments, sigma=0.1):
    """
    based on paper(Deep Divergence-Based Approach to Clustering): https://arxiv.org/abs/1902.04981
    Parameters
    ----------
    p_feats
    cluster_assignments
    sigma

    Returns
    -------

    """
    n,k,c = cluster_assignments.shape

    A = cluster_assignments.permute(1,0,2)
    K = torch.cdist(p_feats.permute(1,0,2), p_feats.permute(1,0,2))
    K = torch.exp(-K.pow(2)/(2*sigma))
    D_a = kernel_divergence(K, A)
    E = torch.eye(c).to(p_feats.device)
    M = torch.exp(-torch.norm((A[:, :, None, :] - E), dim=-1, p=2))
    D_m = kernel_divergence(K, M)
    AA = torch.einsum('KnC, KmC -> Knm', A, A)
    return torch.triu(D_a, diagonal=1).sum() / k + torch.triu(AA, diagonal=1).sum()/(n*k) + torch.triu(D_m, diagonal=1).sum() / k

def clustering_loss2(c_feats, labels, index, sigma=0.1):
    c = labels.max().item() + 2
    A = Fn.one_hot(labels[index].long() + 1, c)[:, :, 1:]
    n, k, d = c_feats.shape

    A = A.permute(1, 0, 2).to(c_feats.device)
    K = torch.cdist(c_feats.permute(1, 0, 2), c_feats.permute(1, 0, 2))
    K = torch.exp(-K.pow(2) / (2 * sigma))
    D_a = kernel_divergence(K, A.float())
    return torch.triu(D_a, diagonal=1).sum() / k

def clustering_loss3(p_feats, cluster_assignments, sigma=0.1):
    """
    based on paper(Deep Divergence-Based Approach to Clustering): https://arxiv.org/abs/1902.04981
    Parameters
    ----------
    p_feats
    cluster_assignments
    sigma

    Returns
    -------

    """
    n,k,c = cluster_assignments.shape

    A = cluster_assignments.permute(1,0,2)
    K = torch.cdist(p_feats.permute(1,0,2), p_feats.permute(1,0,2))
    K = torch.exp(-K.pow(2)/(2*sigma))
    D_a = kernel_divergence(K, A)
    E = torch.eye(c).to(p_feats.device)
    M = torch.exp(-torch.norm((A[:, :, None, :] - E), dim=-1, p=2))

    return torch.triu(D_a, diagonal=1).sum() / k


def concept_loss(concept, names, namesToConcept):
    b, p, c = concept.shape
    loss, count= 0, 0
    for i in range(p):
        for j in range(len(names)):
            if names[j] in namesToConcept[i]:
                conc = namesToConcept[i][names[j]]
                mask = conc > -9.9
                loss += (concept[j,i][mask] - conc[mask]).pow(2).mean()
                count += 1
    if count > 0:
        return loss / count
    return torch.tensor([0.0], requires_grad=True, device=concept.device)

def rho_loss(data_rho, rho, size_average=True):
    dkl = - rho * torch.log(data_rho) - (1 - rho) * torch.log(1 - data_rho)  # calculates KL divergence
    if size_average:
        _rho_loss = dkl.mean()
    else:
        _rho_loss = dkl.sum()
    return _rho_loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)



    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


def concept_mining_loss(net, part_pooled, Feat, labels):
    w = net.proto_cls.weight
    p = part_pooled.shape[1]
    # A_p = torch.stack([Fn.linear(part_pooled[:, i], w[:, i * 256:i * 256 + 256].cpu()) for i in range(p)], dim=1)
    F = Feat[:, net.C_ind.long()]
    newF = Fn.cosine_similarity(F, net.C_centers, dim=-1)
    out = Fn.linear(newF, net.W_c)
    return Fn.cross_entropy(out, labels)


