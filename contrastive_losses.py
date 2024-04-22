import torch
import torch.nn as nn
import torch.nn.functional as F

class SupContLoss(nn.Module):
    # This class is based on the loss function presented in Supervised Contrastive Learning (https://arxiv.org/abs/2004.11362)
    def __init__(self, positive_samples: int, temperature=0.1):
        super(SupContLoss, self).__init__()
        """
        Args:
            positive_samples (int): Number of positive samples (including the anchor)
            temperature (float): Temperature parameter of the supervised contastive loss function
        """
        self.m = positive_samples
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): tensor containing the features of the samples in the batch.
                                      Shape should be [m+n, feature_dim] where m is for positive samples (including the anchor),
                                      and n is the number of negative samples.
                                      Batch must be ordered as [anchor, positives, negatives]

        Returns:
            torch.Tensor: The contrastive loss value.
        """
        # Normalize the feature vectors to the unit sphere (L2 norm)
        normalized_features = F.normalize(features, p=2, dim=1)

        # Calculate similarities: the first vector (anchor) with all others including the positive and negatives
        similarities = torch.matmul(normalized_features, normalized_features[0].unsqueeze(1)).squeeze() # Dot product of anchor with all other features
        
        # From index = 1, the first self.m samples are positive samples
        positives = torch.sum(torch.exp(similarities[1:self.m] / self.temperature)) # Exponentiate the similarities and scale by the temperature

        # Sum the exponentiated similarities for all samples except the anchor
        negatives_and_positive = torch.sum(torch.exp(similarities[1:] / self.temperature))

        # Compute the contrastive loss using negative log likelihood
        loss = -torch.log(positives / negatives_and_positive)
        return loss

class GramMatrixSimilarityLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(GramMatrixSimilarityLoss, self).__init__()
        self.margin = margin  # Margin for contrastive loss

    def forward(self, gram_matrices):
        """
        Args:
            gram_matrices (torch.Tensor): tensor containing the Gram matrices of the samples in the batch.
                                          Shape should be [batch_size, C, C] where C is the number of channels.
                                          Batch must be ordered as [anchor, positives, negatives]

        Returns:
            torch.Tensor: The contrastive loss value for the batch.
        """
        anchor_gram = gram_matrices[0]
        positive_gram = gram_matrices[1]

        # Compute similarity between anchor and positive
        positive_similarity = torch.mean((anchor_gram - positive_gram) ** 2)

        # Compute similarities between anchor and negatives
        negative_similarities = []
        for i in range(2, gram_matrices.size(0)):
            negative_similarity = torch.mean((anchor_gram - gram_matrices[i]) ** 2)
            negative_similarities.append(negative_similarity)
        negative_similarity = torch.mean(torch.stack(negative_similarities))

        loss = positive_similarity + torch.clamp(self.margin - negative_similarity, min=0.0) # Clamps all elements into the range [ min=0, max=None ].
        return loss

if __name__ == "__main__":
    # Example Gram matrices for the purpose of this example
    # Let's assume C=20 for simplicity, batch_size=10 (1 anchor, 1 positive, 8 negatives)
    # gram_matrices = torch.randn(10, 20, 20)
    # gram_similarity_loss = GramMatrixSimilarityLoss(margin=1.0)
    # loss = gram_similarity_loss(gram_matrices)
    # print("Gram Matrix Similarity Loss:", loss.item())

    # Example tensor [2+n, 500] where n is number of negatives
    # Randomly generated features for the purpose of this example
    features = torch.randn(10, 500)  # n=8 negatives, 1 anchor, 1 positive
    sup_cont_loss = SupContLoss(temperature=0.05)
    loss = sup_cont_loss(features)
    print("Contrastive Loss:", loss.item())
