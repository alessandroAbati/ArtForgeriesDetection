import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, random_split

from models import EfficientNetModel, Head
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

def extract_features(data_settings, model_settings, train_settings):
    # Dataset
    dataset = WikiArtDataset(data_dir=data_settings['dataset_path'], binary=data_settings['binary'])
    train_size = int(0.8 * len(dataset)) # 80% training set
    _, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=False)

    # Model
    if model_settings['model_type'] == 'resnet':
        raise ValueError("The style plot is not supported for 'resnet' model, please change the settings in the config file")
    elif model_settings['model_type'] == 'efficientnet':
        model = EfficientNetModel().to(device)
        model_comp = EfficientNetModel().to(device)
        model_head = Head(encoder_model=model, num_classes=model_settings['num_classes']).to(device)
        model_head_comp = Head(encoder_model=model_comp, num_classes=model_settings['num_classes']).to(device)

        print("Model loaded")
    else:
        raise ValueError("Model type in config.yaml should be 'resnet' or 'efficientnet'")

    # Loading checkpoint
    ckpt = torch.load( f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_multi_binary={data_settings['binary']}_contrastive={data_settings['contrastive']}.pth", map_location=device)
    model_weights = ckpt['model_state_dict']
    model.load_state_dict(model_weights)

    ckpt = torch.load(
        f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_multi_binary={data_settings['binary']}_contrastive={data_settings['contrastive']}.pth",
        map_location=device)
    model_weights = ckpt['model_state_dict']
    model_comp.load_state_dict(model_weights)

    ckpt = torch.load(
        f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_multi_head_binary={data_settings['binary']}_contrastive={data_settings['contrastive']}.pth",
        map_location=device)
    model_weights = ckpt['model_state_dict']
    model_head.load_state_dict(model_weights)

    ckpt = torch.load(
        f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_multi_head_binary={data_settings['binary']}_contrastive={data_settings['contrastive']}.pth",
        map_location=device)
    model_weights = ckpt['model_state_dict']
    model_head_comp.load_state_dict(model_weights)

    models_differ = 0
    for key_item_1, key_item_2 in zip(model.state_dict().items(), model_comp.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

    for param in model.parameters():
        param.requires_grad = False

    # for name, param in model.named_parameters():
    #     print(name, param)
    print("Model's pretrained weights loaded!")

    extracted_features = []
    extracted_features2 = []
    binary_labels = []
    pred_labels = np.array([])
    labels_list = np.array([])

    model.eval()
    model_head.eval()
    with torch.no_grad():
        for images, labels, AI_labels in val_loader:
            images = images.to(device)
            labels_list = np.concatenate((labels_list, labels))
            for label in AI_labels:
                binary_labels.append(label)
            outputs, _ = model(images)
            extracted_features.append(outputs.detach().cpu())

            output = model_head(outputs)
            preds = torch.argmax(output, dim=1).cpu()
            pred_labels = np.concatenate((pred_labels, preds))

    model_comp.eval()
    model_head_comp.eval()
    with torch.no_grad():
        for images, labels, AI_labels in val_loader:
            images = images.to(device)
            labels_list = np.concatenate((labels_list, labels))
            for label in AI_labels:
                binary_labels.append(label)
            outputs, _ = model_comp(images)
            extracted_features2.append(outputs.detach().cpu())
            output = model_head_comp(outputs)
            preds = torch.argmax(output, dim=1).cpu()
            pred_labels = np.concatenate((pred_labels, preds))

    # Concatenate all the features
    features_tensor = torch.cat(extracted_features, dim=0)
    features_tensor2 = torch.cat(extracted_features2, dim=0)

    cosine = torch.nn.CosineSimilarity()
    print(cosine(features_tensor, features_tensor2))
    if torch.equal(features_tensor, features_tensor2):
        print("The tensors are equal")
    else:
        print("Difference!")
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

    if data_setting['binary']: model_setting['num_classes']=2 # Force binary classification if binary setting is True


    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    
    extract_features(data_setting, model_setting, train_setting)

if __name__ == '__main__':
    main()
