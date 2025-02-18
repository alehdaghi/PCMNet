import math

import einops
import torch
from torch.nn import BatchNorm2d, Softmax2d
from torchvision.models.resnet import ResNet
from typing import Tuple
from torch import nn
import torch.nn.functional as F

from losses import MarginalCenterLoss, PartCenterLoss


class PPCNet(torch.nn.Module):
    def __init__(self, init_model: ResNet, num_landmarks: int = 8,
                 num_classes: int = 2000, landmark_dropout: float = 0.3) -> None:
        """
        Parameters
        ----------
        init_model: ResNet
            The pretrained ResNet model
        num_landmarks: int
            Number of landmarks to detect
        num_classes: int
            Number of classes for the classification
        landmark_dropout: float
            Probability of dropping out a given landmark
        """
        super().__init__()

        # The base model
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        self.relu = init_model.relu
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4
        self.finalpool = torch.nn.AdaptiveAvgPool2d(1)

        # New part of the model
        self.softmax: Softmax2d = torch.nn.Softmax2d()
        self.batchnorm = BatchNorm2d(11)
        self.fc_landmarks = torch.nn.Conv2d(1024 + 2048, num_landmarks + 1, 1, bias=False)
        self.fc_class_landmarks = torch.nn.Linear(1024 + 2048, num_classes, bias=False)
        # self.modulation = torch.nn.Parameter(torch.ones((1, num_landmarks + 1, 256)))
        self.modulation = torch.nn.Parameter(torch.ones((1,1024 + 2048,num_landmarks + 1)))
        self.dropout = torch.nn.Dropout(landmark_dropout)
        self.dropout_full_landmarks = torch.nn.Dropout1d(landmark_dropout)

        self.proto_enc = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=True))
                for i in range(self.num_landmarks)])
        self.global_enc = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.global_cls = NonNegLinear(512, num_classes, bias=False)

        self.proto_cls = nn.Linear((self.num_landmarks) * 256, num_classes, bias=False)


        self.pool_layer = nn.Sequential(nn.AdaptiveMaxPool2d(output_size=(1, 1)), nn.Flatten())
        self.centerLoss = MarginalCenterLoss(num_classes=self.num_landmarks, feat_dim=1024 + 2048)

        self.badPart= list(range(self.num_landmarks))


    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        x: torch.Tensor
            Input image

        Returns
        -------
        all_features: torch.Tensor
            Features per landmark
        maps: torch.Tensor
            Attention maps per landmark
        scores: torch.Tensor
            Classification scores per landmark
        """
        # Pretrained ResNet part of the model
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        x4 = self.layer4(l3)
        # x4U = torch.nn.functional.upsample_bilinear(x4, size=(l3.shape[-2], l3.shape[-1]))
        x4U = torch.nn.functional.interpolate(x4, size=(l3.shape[-2], l3.shape[-1]), mode = "bilinear")
        x = torch.cat((x4U, l3), dim=1)

        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps resnet, a = convolution kernel
        batch_size = x.shape[0]
        ab = self.fc_landmarks(x)
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1)
        a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, x.shape[-2], x.shape[-1])
        a_sq = a_sq.permute(1, 0, 2, 3)
        maps = b_sq - 2 * ab + a_sq
        maps = -maps

        # Softmax so that the attention maps for each pixel add up to 1
        maps = self.softmax(maps)

        b, c, w, h = x4.shape
        p, n = self.num_landmarks, self.num_classes

        # Use maps to get weighted average features per landmark
        feature_tensor = x
        # all_features = torch.einsum('bkwh, bdwh -> bdk', maps, feature_tensor) / (w * h)
        XK = ((maps).unsqueeze(1) * feature_tensor.unsqueeze(2))
        mapsD = F.interpolate(maps, (w, h), mode= 'bilinear')
        XK2 = ((mapsD).unsqueeze(1) * x4.unsqueeze(2))


        all_features = XK.mean(-1).mean(-1)
        proto_feats = torch.stack([self.proto_enc[i](XK2[:, :, i]) for i in range(self.num_landmarks)], dim=1)

        XG = mapsD[:, -1:].detach() * x4
        XG = self.global_enc(XG)
        XG1 = F.softmax(XG, dim=1)
        XG = XG1 * torch.relu(XG)
        b, c, w, h = XG.shape
        g_feat, g_index = XG.view(b, XG.shape[1], -1).max(-1)
        g_index = torch.stack([g_index // h, g_index % h], -1)
        scores = self.global_cls(g_feat)
        proto_feats_modulated = proto_feats.mean(-1).mean(-1) #* self.modulation
        proto_feats_modulated2 = self.dropout_full_landmarks(proto_feats_modulated)

        concepts = None
        c_score = None
        if 'C_centers' in self._buffers and self.C_centers != None:
            W_p = self.proto_cls.weight
            A_p = torch.stack([F.linear(proto_feats_modulated[:, i], W_p[:, i * 256:i * 256 + 256]) for i in range(p)],
                              dim=1)
            # A_p = A_p.reshape(b, -1).softmax(dim=-1)
            A_p_r = A_p.view(b, -1)[:,self.C_ind[:,1].long()]
            # Feat = all_features[:, :, self.C_ind[:,0].long()].permute(0,2,1)
            Feat = proto_feats_modulated[:, self.C_ind[:,0].long(), :]
            concepts = (0.5+0.5*F.cosine_similarity(Feat, self.C_centers, dim=-1)) * A_p_r
            if True or not self.training:
                Arr = torch.split(concepts, self.C_sec.tolist(), dim=1)
                concepts_s = []
                for C in Arr:
                    newC = torch.zeros_like(C)
                    v, i = C.max(dim=-1)
                    newC[range(len(i)), i] = v
                    concepts_s.append(newC)
                concepts = torch.cat(concepts_s, dim=1)
            elif y is not None:
                Y = y.cpu()
                C = torch.ones_like(concepts)
                for i in range(len(y)):
                    for pp in range(p):
                        if self.C_label[pp][Y[i]][0] < self.C_label[pp][Y[i]][1]:
                            C[i, self.C_label[pp][Y[i]][0]:self.C_label[pp][Y[i]][1]] = 0
                            arg = concepts[i, self.C_label[pp][Y[i]][0]:self.C_label[pp][Y[i]][1]].argmax(dim=-1) + \
                                  self.C_label[pp][Y[i]][0]
                            C[i, arg] = 1
                concepts = concepts * C

            c_score = F.linear(concepts, self.W_c)
        else:
            c_score = None #self.concept_cls(cluster_assign.reshape(batch_size, -1))


        p_scores = self.proto_cls(proto_feats_modulated2.reshape(batch_size, -1))

        # Classification based on the landmarks
        all_features_modulated = all_features


        # pos, label, features = self.depth(x, maps)
        G = {
            'x4': x4,
            "cluster_assign": None,
            "g_index" : g_index,
            "g_feat": g_feat,
            "XG": XG,
            "c_score": c_score,
            "p_scores_c": None, #p_scores_c
            'concepts': concepts

            # 'proto_concept': proto_concept
        }


        # return p_scores
        return all_features, maps, scores, p_scores, proto_feats_modulated, G


    def compute_Clabel(self ):
        N_c = self.C_ind.shape[0]
        self.C_label = torch.zeros((self.num_landmarks, self.num_classes, 2),dtype=torch.long)
        C_ind = self.C_ind.cpu()
        i = 0
        while(i < N_c):
            p, c , _, s, e = C_ind[i]
            self.C_label[p][c%self.num_classes] = C_ind[i,-2:]
            i = e
        return

    def resetPart(self, partIndex):
        n = self.fc_landmarks.in_channels
        for k in self.fc_landmarks.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.fc_landmarks.weight[partIndex].data.uniform_(-stdv, stdv)
        self.proto_enc[partIndex][0].reset_parameters()
        torch.nn.init.xavier_uniform_(self.proto_cls.weight[:, partIndex*256:(partIndex+1)*256])
        torch.nn.init.normal_(self.centerLoss.centers[partIndex])






# adapted from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
class NonNegLinear(nn.Module):
    """Applies a linear transformation to the incoming data with non-negative weights`
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(NonNegLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.rand((out_features, in_features), **factory_kwargs))
        self.normalization_multiplier = nn.Parameter(torch.ones((1,),requires_grad=True))
        if bias:
            self.bias = nn.Parameter(torch.rand(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input,torch.relu(self.weight), self.bias)

class NonNegConv2d(nn.Conv2d):
    def __init__(self,*args, **kwargs) -> None:
        # factory_kwargs = {"device": device, "dtype": dtype}
        super(NonNegConv2d, self).__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, torch.relu(self.weight), self.bias)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.01)
        nn.init.zeros_(m.bias.data)
