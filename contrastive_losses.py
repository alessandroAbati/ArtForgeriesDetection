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
                                      Shape should be [2+n, feature_dim] where 2 is for the anchor and its positive,
                                      and n is the number of negatives.

        Returns:
            torch.Tensor: The contrastive loss value.
        """
        # Normalize the feature vectors to the unit sphere (L2 norm)
        normalized_features = F.normalize(features, p=2, dim=1)

        # Calculate similarities: Dot product of anchor with all other features
        # The first vector (anchor) with all others including the positive and negatives
        similarities = torch.matmul(normalized_features, normalized_features[0].unsqueeze(1)).squeeze()

        # Exponentiate the similarities and scale by the temperature
        # The first term is the positive sample (index 1)
        positives = torch.exp(similarities[1] / self.temperature)

        # Sum the exponentiated similarities for all including the positive (to apply softmax)
        # Starting from index 1 to include the positive once in the denominator
        negatives_and_positive = torch.sum(torch.exp(similarities[1:] / self.temperature))

        # Compute the contrastive loss using negative log likelihood
        loss = -torch.log(positives / negatives_and_positive)
        return loss


class GramMatrixSimilarityLoss(nn.Module):
    def __init__(self):
        super(GramMatrixSimilarityLoss, self).__init__()

    def forward(self, gram_matrices):
        """
        Args:
            gram_matrices (torch.Tensor): tensor containing the Gram matrices of the samples in the batch.
                                          Shape should be [batch_size, C, C] where C is the number of channels.

        Returns:
            torch.Tensor: The mean similarity loss for the batch.
        """
        # Extract the Gram matrix for the anchor
        anchor_gram = gram_matrices[0]

        # Calculate similarities with the anchor
        similarities = []
        for i in range(1, gram_matrices.size(0)):
            # Element-wise similarity (can be defined as negative squared distance)
            similarity = -torch.mean((anchor_gram - gram_matrices[i]) ** 2)
            similarities.append(similarity)

        # Concatenate all similarity scores and compute mean loss
        loss = -torch.mean(torch.stack(similarities))
        return loss

if __name__ == "__main__":
    # Example Gram matrices for the purpose of this example
    # Let's assume C=20 for simplicity, batch_size=10 (1 anchor, 1 positive, 8 negatives)
    gram_matrices = torch.randn(10, 20, 20)
    gram_similarity_loss = GramMatrixSimilarityLoss()
    loss = gram_similarity_loss(gram_matrices)
    print("Gram Matrix Similarity Loss:", loss.item())

    # Example tensor [2+n, 500] where n is number of negatives
    # Randomly generated features for the purpose of this example
    """features = torch.randn(10, 500)  # n=8 negatives, 1 anchor, 1 positive
    sup_cont_loss = SupContLoss(temperature=0.05)
    loss = sup_cont_loss(features)
    print("Contrastive Loss:", loss.item())"""
