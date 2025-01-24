import torch.nn as nn
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class IslandLoss(nn.Module):

    def __init__(self, features_dim, num_class=16, lamda=1., lamda1=10., scale=1.0, batch_size=18):

        super(IslandLoss, self).__init__()
        self.lamda = lamda
        self.lamda1 = lamda1
        self.num_class = num_class
        self.scale = scale
        self.batch_size = batch_size
        self.feat_dim = features_dim
        self.feature_centers = nn.Parameter(torch.randn(num_class, features_dim).to(device))

    def forward(self, output_features, y_truth):
        batch_size = y_truth.size(0)
        num_class = self.num_class
        output_features = output_features.view(batch_size, -1)
        assert output_features.size(-1) == self.feat_dim

        factor = self.scale / batch_size

        centers_batch = self.feature_centers.index_select(0, y_truth.long())

        diff = output_features - centers_batch
        loss_center = 1 / 2.0 * (diff.pow(2).sum()) * factor
        centers = self.feature_centers
        centers_mod = torch.sum(centers * centers, dim=1, keepdim=True).sqrt()
        item1_sum = 0
        for j in range(num_class):
            dis_sum_j_others = 0
            for k in range(j + 1, num_class):
                dot_kj = torch.sum(centers[j] * centers[k])
                fenmu = (centers_mod[j] * centers_mod[k] + 1e-9)
                cos_dis = dot_kj / fenmu
                dis_sum_j_others += cos_dis + 1.
            item1_sum += dis_sum_j_others
        loss_island = self.lamda * (loss_center + self.lamnda1 * item1_sum)

        return loss_island