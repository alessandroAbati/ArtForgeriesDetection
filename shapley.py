import torch
import os
from torch.utils.data import DataLoader, random_split
from utils import load_config
from models import ResNetModel, EfficientNetModel, EfficientNetModelAttention
from logger import Logger
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F
from dataset_v2 import WikiArtDataset

import shap

torch.manual_seed(42)

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# def shapley(model, val_loader, criterion, binary_loss)

def seed_torch(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run_shap_deep_explainer(model, val_loader):
    batch = next(iter(val_loader))
    images, _, _ = batch
    images = images.to(device)

    background = images[:12]
    test_images = images[10:20]
    print(test_images.shape)

    e = shap.GradientExplainer(model, background)
    shap_values = e.shap_values(test_images, nsamples=10)
    print(shap_values)
    print(shap_values.shape)
    for i in range(test_images.size(0)):
        shap.image_plot([shap_values[i,:,:,:,j].transpose(1,2,0) for j in range(2)], test_images[i].permute(1,2,0).cpu().detach().numpy())

def validate_loop(model, val_loader, criterion, binary_loss, attention = False):
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
                    print(outputs)
            else:
                outputs = model(images)

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

def calculate_metrics(preds, labels, model_settings):
    if model_settings['num_classes'] == 2:
        accuracy = Accuracy(task='binary', num_classes=model_settings['num_classes']).to(device)
        precision = Precision(task='binary', average='weighted', num_classes=model_settings['num_classes']).to(device)
        recall = Recall(task='binary', average='weighted', num_classes=model_settings['num_classes']).to(device)
        f1 = F1Score(task='binary', average='weighted', num_classes=model_settings['num_classes']).to(device)
        confusion_matrix = ConfusionMatrix(task='binary', num_classes=model_settings['num_classes']).to(device)
    else:
        accuracy = Accuracy(task='multiclass', num_classes=model_settings['num_classes']).to(device)
        precision = Precision(task='multiclass', average='macro', num_classes=model_settings['num_classes']).to(device)
        recall = Recall(task='multiclass', average='macro', num_classes=model_settings['num_classes']).to(device)
        f1 = F1Score(task='multiclass', average='macro', num_classes=model_settings['num_classes']).to(device)
        confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=model_settings['num_classes']).to(device)

    acc = accuracy(preds, labels)
    prec = precision(preds, labels)
    rec = recall(preds, labels)
    f1_score = f1(preds, labels)
    conf_matrix = confusion_matrix(preds, labels)

    return acc, prec, rec, f1_score, conf_matrix

def inference(train_dataset, val_dataset, data_settings, model_settings, train_settings, frozen_encoder=False, contrastive=False):
    print(f"Length Val dataset: {len(val_dataset)}")
    val_loader = DataLoader(val_dataset, 1, shuffle=False)
    val_loader_shap = DataLoader(val_dataset, 128, shuffle=False)


    attention = False
    # Model
    if model_settings['model_type'] == 'resnet':
        model = ResNetModel(resnet_version='resnet101', num_classes=model_settings['num_classes']).to(device)
    elif model_settings['model_type'] == 'efficientnet':
        model = EfficientNetModel(num_classes=model_settings['num_classes'], checkpoint_path=None,
        binary_classification=model_settings['binary']).to(device)
        print("Model loaded")
    elif model_settings['model_type'] == 'efficientnetAttention':
        model = EfficientNetModelAttention(num_classes=model_settings['num_classes'], checkpoint_path=None,
                                           binary_classification=model_settings['binary']).to(device)
        print("Model with Attention loaded")
        # attention = True
    else:
        raise ValueError("Model type in config.yaml should be 'resnet' or 'efficientnet'")

    # Loading checkpoint
    binary_loss = False
    ckpt = torch.load(f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_binary.pth")
    model.load_state_dict(ckpt['model_state_dict'])
    # for param in model.parameters():
    #     print(param.data)

    # model.load_state_dict(torch.load(f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_binary_contrastive_weights.pth"))

    print("Model's pretrained weights loaded!")
    binary_loss = True

    # Loss
    if model_settings['binary']:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    val_loss, val_preds, val_labels = validate_loop(model, val_loader, criterion, binary_loss, attention = attention)
    acc, prec, rec, f1_score, conf_matrix = calculate_metrics(val_preds, val_labels, model_settings)
    run_shap_deep_explainer(model, val_loader_shap)
    print(f'Accuracy: {acc}, Precision: {prec},  Recall: {rec}, F1-Score: {f1_score}, Validation Loss: {val_loss:.4f}')
    f, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(conf_matrix.clone().detach().cpu().numpy(), annot=True, ax=ax)
    plt.show()


def main():
    config = load_config()

    data_settings = config['data_settings']
    model_setting = config['model']
    train_setting = config['train']

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    seed_torch()
    dataset = WikiArtDataset(data_dir=data_settings['dataset_path'], binary=data_settings['binary'])  # Add parameters as needed
    train_size = int(0.8 * len(dataset)) # 80% training set
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    inference(train_dataset, val_dataset, data_settings, model_setting, train_setting)

if __name__ == '__main__':
    main()