import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import math

class ResNetModel(nn.Module):
    def __init__(self, num_classes, resnet_version='resnet18', binary_classification=False):
        super(ResNetModel, self).__init__()
        self.binary_classification = binary_classification

        self.model = getattr(models, resnet_version)()  # Load a pretrained ResNet model

        self.load_checkpoint('pretrain_weights/resnet_pretrain.pth')  # Load checkpoint file (workaround for hyperion proxy problem)

        # Freeze layers
        for p in self.model.parameters():
            p.requires_grad = False
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)  # Replace the classifier layer

    def forward(self, x):
        if self.binary_classification:
            return torch.sigmoid(self.model(x))[:, 0]
        else:
            return self.model(x)

    def load_checkpoint(self, checkpoint_path):
        weights = torch.load(checkpoint_path)
        self.model.load_state_dict(weights)


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, efficientnet_version='efficientnet-b0',
                 binary_classification=False, contrastive_learning=False, frozen_encoder=False):
        super(EfficientNetModel, self).__init__()
        self.binary_classification = binary_classification
        self.contrastive_learning = contrastive_learning
        self.frozen_encoder = frozen_encoder

        projection_dimension = 128

        self.model = EfficientNet.from_name(efficientnet_version)  # Load without pretrained weights

        num_features = self.model._fc.in_features

        if contrastive_learning:
            self.model._fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, projection_dimension))  # Replace the classifier layer with a projection head
        else:
            self.model._fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes))  # Replace the classifier layer with a projection head
            # self.model._fc = nn.Linear(num_features, num_classes) # Replace the classifier layer

    def forward(self, x):
        if self.binary_classification:
            # print("Sigmoid")
            return torch.sigmoid(self.model(x))
        else:
            return self.model(x)


class EfficientNetModelAttention(nn.Module):
    def __init__(self, num_classes, efficientnet_version='efficientnet-b0',
                 binary_classification=False, contrastive_learning=False, frozen_encoder=False):
        super(EfficientNetModelAttention, self).__init__()
        self.binary_classification = binary_classification
        self.frozen_encoder = frozen_encoder


        self.model = EfficientNet.from_name(efficientnet_version)  # Load without pretrained weights

        self.model._avg_pooling = nn.Identity()
        self.model._dropout = nn.Identity()
        num_features = self.model._fc.in_features
        print(num_features)
        self.model._fc = nn.Identity()

        # self.attention = AttentionMultiHead(num_features, 512, 4)
        self.attention = SelfAttentionCNN(in_dim=1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        projection_dimension = 128
        if contrastive_learning:
            print("Contrastive Learning!")
            self.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, projection_dimension))  # Replace the classifier layer with a projection head
        else:
            self.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes))

    def forward(self, x):
        if self.binary_classification:
            features = self.model.extract_features(x)
            context, weights = self.attention(features)
            # print(context.shape)
            context = self.avgpool(context)
            context = context.view(context.size(0), -1)
            output = self.fc(context)
            return torch.sigmoid(output), weights
            # return torch.sigmoid(output)
        #
        else:
            features = self.model.extract_features(x)
            context, weights = self.attention(features)
            context = self.avgpool(context)
            context = context.view(context.size(0), -1)
            output = self.fc(context)
            return output

class AttentionMultiHead(nn.Module):

    def __init__(self, input_size, hidden_size, nr_heads):
        super(AttentionMultiHead, self).__init__()
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList([])
        self.heads.extend([SelfAttention(input_size, hidden_size) for idx_head in range(nr_heads)])
        self.linear_out = nn.Linear(nr_heads * hidden_size, input_size)
        return

    def forward(self, input_vector):
        all_heads = []
        for head in self.heads:
            out = head(input_vector)
            all_heads.append(out)
        z_out_concat = torch.cat(all_heads, dim=2)
        z_out_out = F.relu(self.linear_out(z_out_concat))
        # print(z_out_out.shape)
        return z_out_out


class SelfAttention(nn.Module):

    def __init__(self, input_size, out_size):
        super(SelfAttention, self).__init__()
        self.dk_size = out_size
        self.query_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.key_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.value_linear = nn.Linear(in_features=input_size, out_features=out_size)
        self.softmax = nn.Softmax()
        self.apply(self.init_weights)
        return

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input_vector):
        query_out = F.relu(self.query_linear(input_vector))
        key_out = F.relu(self.key_linear(input_vector))

        value_out = F.relu(self.value_linear(input_vector))
        # out_q_k = torch.bmm(query_out, key_out.transpose(1, 2))
        out_q_k = torch.div(torch.bmm(query_out, key_out.transpose(1, 2)), math.sqrt(self.dk_size))
        softmax_q_k = self.softmax(out_q_k)
        out_combine = torch.bmm(softmax_q_k, value_out)
        return out_combine


class SelfAttentionCNN(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttentionCNN, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out, attention