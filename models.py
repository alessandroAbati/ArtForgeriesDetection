import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import math

"""class ResNetModel(nn.Module):
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
            return torch.sigmoid(self.model(x))
        else:
            return self.model(x)

    def load_checkpoint(self, checkpoint_path):
        weights = torch.load(checkpoint_path)
        self.model.load_state_dict(weights)"""

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()

        self.model = models.resnet101(pretrained=True)  # Load a pretrained ResNet model
        self.model.fc = nn.Identity() # "Remove" fully connected layer

    def forward(self, x):
        output = self.model(x) # shape: [batch_size, emb_dim=2048]
        return output, None


class EfficientNetModel(nn.Module):
    def __init__(self, efficientnet_version='efficientnet-b0'):
        """
            EfficientNet model (https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/).

            Args:
                efficientnet_version (string, optional): EfficientNet model version.

            What we changed:
            1. All the layers after the convolutional_head (_conv_head) had been set to Identity.
            2. We added an AdaptiveAvgPool2d final layer to extract the features from the model.

            This allow us to use the extract_features() function of the model.
            The extract_features() function return the features map that we can use for the Gram based contrastive learning.
            Otherwise, we pass the features map to the avgpool layer to flatten the features.      
        """
        super(EfficientNetModel, self).__init__()
        # self.model = EfficientNet.from_name(efficientnet_version)  # Load without pretrained weights
        self.model = EfficientNet.from_pretrained(efficientnet_version)

        self.model._avg_pooling = nn.Identity()
        self.model._dropout = nn.Identity()
        self.model._fc = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch of images with shape [batch_size, channels, width, height]

        Returns:
            context (torch.Tensor): flatten extracted features with shape [batch_size, emd_dim=1280 (efficientnet-b0)].
            features (torch.Tensor): extracted features with shape [batch_size, emd_dim, 16, 16]
        """
        features = self.model.extract_features(x) # size: [batch_size, emd_dim, 16, 16]
        context = self.avgpool(features) # size: [batch_size, emb_dim, 1, 1]
        context = context.view(context.size(0), -1) # Flattening to shape [batch_size, emb_dim]
        return context, features

class Head(nn.Module):
    def __init__(self,
                 encoder_model, 
                 num_classes, 
                 contrastive_learning=False):
        """
            Head model - head for the EfficientNet model.

            Args:
                encoder_model (callable): encoder model to get the embedding dimension (output feature vector dimesion)
                num_classes (int): number of classes for the classifier head
                binary_classification (bool, optional): control binary classification
                contrastive_learning (bool, optional): contol contrastive learning, if True -> the projection head will be used        
        """
        super(Head, self).__init__()
        if isinstance(encoder_model, ResNetModel):
            emb_dim = 2048
        else:
            # If encoder is EfficientNet
            emb_dim = 1280   

        if contrastive_learning:
            # Projection head
            projection_dimension = 128 # Output of the projection head
            self.fc = nn.Sequential(
                nn.Linear(emb_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, projection_dimension))  # Replace the classifier layer with a projection head
        else:
            # Classifier head
            self.fc = nn.Sequential(
                nn.Linear(emb_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): output of the EfficientNet - tensor with shape [batch_size, emb_dim=1280] containing the features extracted with EfficientNet.

        Returns:
            torch.Tensor: output (logits) of the fully connected head.
        """
        output = self.fc(x)
        return output


class EfficientNetModelAttention(nn.Module):
    def __init__(self, efficientnet_version='efficientnet-b0'):
        """
            EfficientNet model (https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/).

            Args:
                efficientnet_version (string, optional): EfficientNet model version.

            What we changed:
            1. All the layers after the convolutional_head (_conv_head) had been set to Identity.
            2. We added an attention layer that acts on the features map extracted from the EfficientNet.
            3. We added an AdaptiveAvgPool2d final layer to extract the features from the model.

            The attention output is passed to the avgpool layer to flatten the features.      
        """
        super(EfficientNetModelAttention, self).__init__()

        self.model = EfficientNet.from_pretrained(efficientnet_version)

        self.model._avg_pooling = nn.Identity()
        self.model._dropout = nn.Identity()
        self.model._fc = nn.Identity()

        # Attention layer
        # self.attention = SelfAttentionCNN(in_dim=1280)
        self.attention = AttentionMultiHead(input_size=1280, hidden_size=1280, nr_heads=4)


        # Avg Pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch of images with shape [batch_size, channels, width, height]

        Returns:
            context (torch.Tensor): flatten extracted features with shape [batch_size, emd_dim=1280 (efficientnet-b0)].
        """
        features = self.model.extract_features(x) # size: [batch_size, emd_dim, 16, 16]
        context, weights = self.attention(features) # context shape: [batch_size, emb_dim, 16, 16]
        context = self.avgpool(context) # shape: [batch_size, emb_dim, 1, 1]
        output = context.view(context.size(0), -1) # Flattening to shape [batch_size, emb_dim]
        return output, weights

class AttentionMultiHead(nn.Module):

    def __init__(self, input_size, hidden_size, nr_heads):
        super(AttentionMultiHead, self).__init__()
        self.hidden_size = hidden_size
        self.heads = nn.ModuleList([])
        self.heads.extend([SelfAttentionCNN(input_size) for idx_head in range(nr_heads)])
        self.linear_out = nn.Linear(nr_heads * hidden_size, input_size)
        return

    def forward(self, input_vector):
        all_heads = []
        all_weights = []
        for head in self.heads:
            out, weights = head(input_vector)
            all_heads.append(out)
            all_weights.append(weights)
        z_out_concat = torch.cat(all_heads, dim=1)
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
        attention_temp = torch.bmm(attention, attention.permute(0,2,1))
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out_att = out.view(batch_size, C, width, height)
        out = self.gamma * out_att + x
        return out, attention_temp