import yaml
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

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