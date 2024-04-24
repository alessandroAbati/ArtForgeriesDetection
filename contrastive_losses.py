import torch
import torch.nn as nn
import torch.nn.functional as F

class SupContLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupContLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): tensor containing the features of the samples in the batch.
                                      Shape should be [4+n, feature_dim] where 4 is for the anchor and its positive,
                                      and n is the number of negatives.

        Returns:
            torch.Tensor: The contrastive loss value.
        """
        # Normalize the feature vectors to the unit sphere (L1 norm now)
        normalized_features = F.normalize(features, p=2, dim=1)

        # Calculate similarities with the anchor (index=0)
        similarities = torch.matmul(normalized_features, normalized_features[0].unsqueeze(1)).squeeze() # Dot product of anchor with all other features

        # Get similarities between anchor and positives (from index=1, the first 3 samples are positive sample)
        positives = torch.sum(torch.exp(similarities[1:4] / self.temperature)) # Exponentiate the similarities and scale by the temperature

        # Get similarities between anchor and all the other samples
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
                                          Shape should be [4+n, C, C] where 4 is for the anchor and its positive,
                                          n is the number of negatives,
                                          and C is the number of channels.
                                          Batch must be ordered as [anchor, positives, negatives]

        Returns:
            torch.Tensor: The contrastive loss value for the batch.
        """
        anchor_gram = gram_matrices[0]

        # Compute similarity between anchor and positive
        positive_similarities = []
        for i in range(1, 4):
            positive_similarity = torch.mean((anchor_gram - gram_matrices[i]) ** 2)
            positive_similarities.append(positive_similarity)
        positive_similarity = torch.mean(torch.stack(positive_similarities))

        # Compute similarities between anchor and negatives
        negative_similarities = []
        for j in range(4, gram_matrices.size(0)):
            negative_similarity = torch.mean((anchor_gram - gram_matrices[j]) ** 2)
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
