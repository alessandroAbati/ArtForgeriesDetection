import torch
import os
from torch.utils.data import DataLoader, random_split
from utils import load_config, calculate_metrics
from models import ResNetModel, EfficientNetModel, EfficientNetModelAttention, Head
from logger import Logger
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F

from dataset_v2 import WikiArtDataset

torch.manual_seed(42)

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def visualize_attention(img, attention_map):
    """
    Visualize attention map on the image
    img: [3, W, H] PyTorch tensor (image)
    attention_map: [W*H, W*H] PyTorch tensor (attention map)
    """
    img = img.squeeze(0)
    attention_map = attention_map.squeeze(0)
    img = img.permute(1, 2, 0) # [W, H, 3]
    attention_map = attention_map.cpu().detach().numpy()
    # attention_map = np.max(attention_map, axis=0) # Average over all heads
    attention_map = np.mean(attention_map, axis=0) # Average over all heads
    attention_map = attention_map.reshape(int(np.sqrt(attention_map.shape[0])), int(np.sqrt(attention_map.shape[0]))) # Reshape to square shape
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))

    # Resize image to match attention map size
    # img_resized = F.interpolate(img.unsqueeze(0), size=attention_map.shape, mode='bilinear', align_corners=False)
    # img_resized = img_resized.squeeze(0).permute(1, 2, 0) # [W, H, 3]
    # print(img_resized.shape)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img.cpu().detach().numpy())
    plt.title('Image')

    plt.subplot(1, 2, 2)
    im = plt.imshow(attention_map, cmap='Greys_r') # Overlay attention map
    plt.title('Attention Map')
    plt.colorbar(im, fraction=0.046, pad=0.04) # Add colorbar as legend
    plt.show()

def validate_loop(model, val_loader, criterion, binary_loss, attention = False, model_head = None):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels, AI in val_loader:
            images, labels = images.to(device), labels.to(device)
            if attention:
                outputs, weights = model(images)
                if labels.item() == 1.0:
                    visualize_attention(images, weights)
                    print(outputs)
            else:
                outputs = model(images)
                outputs = model_head(outputs)

            labels = labels.type(torch.LongTensor).to(device)
            if binary_loss:
                loss = criterion(outputs[:, 1], labels.float())
            else:
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    avg_loss = running_loss / len(val_loader)
    return avg_loss, all_preds, all_labels

def inference(train_dataset, val_dataset, data_settings, model_settings, train_settings, frozen_encoder=False, contrastive=False):
    print(f"Length Val dataset: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, 1, shuffle=False)

    attention = False
    # Model
    if model_settings['model_type'] == 'resnet':
        model = ResNetModel(resnet_version='resnet101', num_classes=model_settings['num_classes']).to(device)
    elif model_settings['model_type'] == 'efficientnet':
        model = EfficientNetModel().to(device)
        model_head = Head(num_classes=model_settings['num_classes'],
                          binary_classification=data_settings['binary']).to(device)
        print("Model loaded")
    elif model_settings['model_type'] == 'efficientnetAttention':
        model = EfficientNetModelAttention(num_classes=model_settings['num_classes'],
                                           binary_classification=data_settings['binary']).to(device)
        print("Model with Attention loaded")
        attention = True
    else:
        raise ValueError("Model type in config.yaml should be 'resnet' or 'efficientnet'")

    # Loading checkpoint
    binary_loss = False
    ckpt = torch.load( f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_binary={data_settings['binary']}_contrastive={data_settings['contrastive']}_3.pth")
    model.load_state_dict(ckpt['model_state_dict'])

    ckpt = torch.load(
        f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_binary={data_settings['binary']}_contrastive={data_settings['contrastive']}_head_3.pth",
        map_location=device)
    model_weights = ckpt['model_state_dict']
    model_head.load_state_dict(model_weights)
    # for param in model.parameters():
    #     print(param.data)

    # model.load_state_dict(torch.load(f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_binary_contrastive_weights.pth"))

    print("Model's pretrained weights loaded!")
    binary_loss = True

    # Loss
    if data_settings['binary']:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    val_loss, val_preds, val_labels = validate_loop(model, val_loader, criterion, binary_loss, attention = attention, model_head=model_head)
    acc, prec, rec, f1_score, conf_matrix = calculate_metrics(val_preds, val_labels, model_settings)
    print(f'Accuracy: {acc}, Precision: {prec},  Recall: {rec}, F1-Score: {f1_score}, Validation Loss: {val_loss:.4f}')
    f, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(conf_matrix.clone().detach().cpu().numpy(), annot=True, ax=ax)
    plt.show()

def main():
    config = load_config()

    data_settings = config['data_settings']
    model_setting = config['model']
    train_setting = config['train']

    if data_settings['binary']: model_setting['num_classes']=2 # Force binary classification if binary setting is True

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    dataset = WikiArtDataset(data_dir=data_settings['dataset_path'], binary=data_settings['binary'])  # Add parameters as needed
    train_size = int(0.8 * len(dataset)) # 80% training set
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    inference(train_dataset, val_dataset, data_settings, model_setting, train_setting)

if __name__ == '__main__':
    main()