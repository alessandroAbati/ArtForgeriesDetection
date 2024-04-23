import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, random_split

from models import EfficientNetModel
from dataset_v2 import WikiArtDataset
from utils import load_config

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def plot_labels(features, true_labels, pred_labels, binary_labels, title):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    markers = {0: 'o', 1: 'x'}  # 0 for non-AI, 1 for AI-generated
    colors = {0: 'blue', 1: 'red'}  # 0 for 'False', 1 for 'True'

    # Plotting with predicted labels
    ax = axes[0]
    for feature, pred_label, binary_label in zip(features, pred_labels, binary_labels):
        ax.scatter(feature[0], feature[1], c=colors[pred_label], marker=markers[binary_label], alpha=0.5)
    ax.set_title('Predicted Labels with AI Generation Indicator')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    # Plotting with true labels
    ax = axes[1]
    for feature, true_label, binary_label in zip(features, true_labels, binary_labels):
        ax.scatter(feature[0], feature[1], c=colors[true_label], marker=markers[binary_label], alpha=0.5)
    ax.set_title('True Labels with AI Generation Indicator')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='True, Non-AI'),
        Line2D([0], [0], marker='x', color='blue', markerfacecolor='blue', markersize=10, label='True, AI-generated'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False, Non-AI'),
        Line2D([0], [0], marker='x', color='red', markerfacecolor='red', markersize=10, label='False, AI-generated')
    ]
    fig.suptitle(f"{title}")
    axes[0].legend(handles=legend_elements, loc='best')
    plt.show()

def extract_features(data_settings, model_settings, train_settings, logger):
    # Dataset
    dataset = WikiArtDataset(data_dir=data_settings['dataset_path'], binary=data_settings['binary'])
    train_size = int(0.8 * len(dataset)) # 80% training set
    _, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=False)

    # Model
    if model_settings['model_type'] == 'resnet':
        raise ValueError("The style plot is not supported for 'resnet' model, please change the settings in the config file")
    elif model_settings['model_type'] == 'efficientnet':
        model = EfficientNetModel(num_classes=model_settings['num_classes'], binary_classification=model_settings['binary']).to(device)
        print("Model loaded")
    else:
        raise ValueError("Model type in config.yaml should be 'resnet' or 'efficientnet'")

    # Loading checkpoint
    ckpt = torch.load(f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_fine.pth", map_location=device)
    model_weights = ckpt['model_state_dict']
    model.load_state_dict(model_weights)
    print("Model's pretrained weights loaded!")

    # Printing layer names:
    #for name, module in model.named_modules():
    #    print(name, module)

    extracted_features = []
    binary_labels = []
    pred_labels = np.array([])
    labels_list = np.array([])

    def hook(module, input, output):
        # output is the output of the hooked layer
        #print(f"output: {output.squeeze().shape}\n{output.squeeze()}")
        extracted_features.append(output.squeeze().detach().cpu())

    # Register the hook
    hook_handle = model.model._avg_pooling.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for images, labels, AI_labels in val_loader:
            images = images.to(device)
            labels_list = np.concatenate((labels_list, labels))
            for label in AI_labels:
                binary_labels.append(label)
            output = model(images)  # The hook captures the features
            preds = torch.argmax(output, dim=1).cpu()
            pred_labels = np.concatenate((pred_labels, preds))
            
    hook_handle.remove() # Remove the hook to avoid memory leaks

    # Concatenate all the features
    features_tensor = torch.cat(extracted_features, dim=0)
    features_np = features_tensor.numpy()
    binary_labels = np.array(binary_labels)

    # K-means clustering
    kmeans = KMeans(n_clusters=2)
    cluster_labels = kmeans.fit_predict(features_np)

    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_np)

    # Plotting
    plot_labels(reduced_features, labels_list, pred_labels, binary_labels, title='CNN predicted labels')
    plot_labels(reduced_features, labels_list, cluster_labels, binary_labels, title='K-means predicted labels')


def main():
    config = load_config()

    data_setting = config['data_settings']
    model_setting = config['model']
    train_setting = config['train']

    """wandb_logger = Logger(
        f"finertuning_efficentnetb0_lr=0.0001_",
        project='ArtForg')
    logger = wandb_logger.get_logger()
    """
    logger = None

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    
    extract_features(data_setting, model_setting, train_setting, logger)

if __name__ == '__main__':
    main()
