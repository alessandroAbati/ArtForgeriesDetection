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


from contrastive_losses import SupContLoss, GramMatrixSimilarityLoss
from dataset_v2 import WikiArtDataset

torch.manual_seed(42)

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def weighted_bce_loss(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def contrastive_learning(class_train_dataset, 
                         class_val_dataset, 
                         data_settings, 
                         model_settings, 
                         train_settings, 
                         logger,
                         criterion='contloss'):
    """
    Execute contrastive learning on the encoder (using projection head).
    Then, execute the training of the classifier head.

    :param class_train_dataset: tarining dataset for the classifier training
    :param class_val_dataset: validation dataset for the classifier training
    :param data_settings: data settings dictionary
    :param model_settings: model settings dictionary
    :param train_settings: train settings dictionary
    :param logger: wandb logger
    :param criterion: criterion for contrastive learing
    """ 

    assert criterion in ['contloss', 'gram'], f"Criterion {criterion} is not valid, please choose between 'contloss' and 'gram'"
    
    # Contrastive Learning Dataset
    train_dataset = WikiArtDataset(data_dir=data_settings['dataset_path'], 
                             binary=data_settings['binary'],
                             contrastive=data_settings['contrastive'],
                             contrastive_batch_size=data_settings['contrastive_batch_size'],
                             index=class_train_dataset.indices)
    val_dataset = WikiArtDataset(data_dir=data_settings['dataset_path'], 
                             binary=data_settings['binary'],
                             contrastive=data_settings['contrastive'],
                             contrastive_batch_size=data_settings['contrastive_batch_size'],
                             index=class_val_dataset.indices)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model
    if model_settings['model_type'] == 'resnet':
        raise ValueError("resnet is not supported for contrastive learning, please change the model settings in the config file")
    elif model_settings['model_type'] == 'efficientnet':
        model = EfficientNetModel().to(device)
    elif model_settings['model_type'] == 'efficientnetAttention':
        model = EfficientNetModelAttention().to(device)
        print("Model loaded")
    else:
        raise ValueError("Model type in config.yaml should be 'resnet' or 'efficientnet' or 'efficientnetAttention")
    
    # Model projection head
    model_head = Head(encoder_model=model, num_classes=model_settings['num_classes'], contrastive_learning=data_settings['contrastive']).to(device)

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['learning_rate'])
    optimizer_head = torch.optim.Adam(model_head.parameters(), lr=train_settings['learning_rate'])

    # Loading checkpoint of the first fine-tuning
    ckpt = torch.load(f"checkpoints/efficientnet_multiclass.pth", map_location=device)
    model_weights = ckpt['model_state_dict']
    model.load_state_dict(model_weights)

    # Contrastive Loss
    if criterion == 'contloss':
        criterion = SupContLoss(temperature=0.1)
    elif criterion == 'gram':
        assert isinstance(model, EfficientNetModel), f"'efficientnetAttention' model is not supported for 'gram' criterion. Please change the model or the criterion."
        criterion = GramMatrixSimilarityLoss(margin=1.0)

    # Training loop
    min_loss = float('inf')
    for epoch in range(50):
        # Training loop
        model.train()
        model_head.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.squeeze(0)  # Squeeze to shape [contrastive_batch, img_width, img_hight]
            labels = torch.stack(labels, dim=0).reshape(len(labels))  # Reshape labels to tensor of shape [contrastive_batch]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            optimizer_head.zero_grad()
            outputs, features = model(images)
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)  # Normalize the feature map
            outputs = model_head(outputs)

            if isinstance(criterion, GramMatrixSimilarityLoss):
                # features shape: [batch, channels, width, height]
                flattened_feature_map = features.reshape(data_settings['contrastive_batch_size'], 1280, -1)  # Flatten height and width into a single dimension -> shape [batch, channels, width*height]
                normalized_feature_map = torch.nn.functional.normalize(flattened_feature_map, p=2, dim=-1)  # Normalize the feature map
                gram_matrices = torch.bmm(normalized_feature_map,
                                          normalized_feature_map.transpose(1, 2))  # shape [batch, channels, channels]
                loss = criterion(gram_matrices)
            elif isinstance(criterion, SupContLoss):
                loss = criterion(outputs)

            loss.backward()
            optimizer.step()
            optimizer_head.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        model_head.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.squeeze(0)  # Squeeze to shape [contrastive_batch, img_width, img_hight]
                labels = torch.stack(labels, dim=0).reshape(len(labels))  # Reshape labels to tensor of shape [contrastive_batch]
                images, labels = images.to(device), labels.to(device)
                outputs, features = model(images)
                outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)  # Normalize the feature map
                outputs = model_head(outputs)

                if isinstance(criterion, GramMatrixSimilarityLoss):
                    # features shape: [batch, channels, width, height]
                    flattened_feature_map = features.reshape(data_settings['contrastive_batch_size'], 1280, -1)  # Flatten height and width into a single dimension -> shape [batch, channels, width*height]
                    normalized_feature_map = torch.nn.functional.normalize(flattened_feature_map, p=2, dim=-1)  # Normalize the feature map
                    gram_matrices = torch.bmm(normalized_feature_map,
                                            normalized_feature_map.transpose(1, 2))  # shape [batch, channels, channels]
                    loss = criterion(gram_matrices)
                elif isinstance(criterion, SupContLoss):
                    loss = criterion(outputs)
                running_loss += loss.item()
            avg_val_loss = running_loss / len(val_loader)
        print(f'Epoch: {epoch + 1}, Contrastive Train Loss: {avg_train_loss:.4f}, Contrastive Val Loss: {avg_val_loss:.4f}')

        # Save checkpoint if improvement
        if avg_val_loss < min_loss:
            print('Saving model...')
            ckpt = {'model_state_dict': model.state_dict()}
            torch.save(ckpt, f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_contrastive.pth")
            min_loss = avg_val_loss

    ckpt = {'model_state_dict': model.state_dict()}
    torch.save(ckpt, f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_contrastive_final.pth")

    # Train the classifier head
    train(class_train_dataset,
          class_val_dataset, 
          data_settings, 
          model_settings, 
          train_settings, 
          logger,
          contrastive=True) 


def train(train_dataset, 
          val_dataset, 
          data_settings, 
          model_settings, 
          train_settings, 
          logger,
          contrastive=False):
    """
    Execute training of the selected model for classification tasks.

    :param train_dataset: tarining dataset for the classifier training
    :param val_dataset: validation dataset for the classifier training
    :param data_settings: data settings dictionary
    :param model_settings: model settings dictionary
    :param train_settings: train settings dictionary
    :param logger: wandb logger
    :param contrastive: bool to execute training after the contarstive learning
    """
    train_loader = DataLoader(train_dataset, batch_size=train_settings['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=False)

    print("DataLoader")
    print(f"train idx: {train_loader._index_sampler}")

    # Model
    if model_settings['model_type'] == 'resnet':
        model = ResNetModel().to(device)
        print("ResNet loaded")
    elif model_settings['model_type'] == 'efficientnet':
        model = EfficientNetModel().to(device)
        print("EfficientNet loaded")
    elif model_settings['model_type'] == 'efficientnetAttention':
        model = EfficientNetModelAttention().to(device)
        print("EfficientNet with Attention loaded")
    else:
        raise ValueError("Model type in config.yaml should be 'resnet' or 'efficientnet' or 'efficientnetAttention'")

    # Model classifier head
    model_head = Head(encoder_model=model, num_classes=model_settings['num_classes']).to(device)

    # Loading checkpoint
    binary_loss = False
    if data_settings['binary']:
        binary_loss = True
        if contrastive:
            ckpt = torch.load(f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_contrastive.pth")
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            ckpt = torch.load(f"checkpoints/efficientnet_multiclass.pth")
            model_weights = (ckpt['model_state_dict'])
            model_weights = {key: value for key, value in model_weights.items() if not key.startswith('attention')}
            model.load_state_dict(model_weights, strict=False)
        print("Model's pretrained weights loaded!")

    # Optimizers
    optimizer = 0
    if not contrastive:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['learning_rate'])
    optimizer_head = torch.optim.Adam(model_head.parameters(), lr=train_settings['learning_rate'])
    
    # Loss
    if data_settings['binary']:
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10]).to(device)) # Sigmoid is implemented in the loss # 24.53
        criterion = torch.nn.BCEWithLogitsLoss() # Sigmoid is implemented in the loss # 24.53

    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Training and validation loop
    min_loss = float('inf')
    for epoch in range(train_settings['epochs']):
        if contrastive:
            model.eval()
            model_head.train()
            train_loss = train_loop(model, model_head, train_loader, criterion, optimizer, optimizer_head, binary_loss)
            model_head.eval()
            val_loss, val_preds, val_labels = validate_loop(model, model_head, val_loader, criterion, binary_loss)
        else:
            model.train()
            model_head.train()
            train_loss = train_loop(model, model_head, train_loader, criterion, optimizer, optimizer_head, binary_loss)
            model.eval()
            model_head.eval()
            val_loss, val_preds, val_labels = validate_loop(model, model_head, val_loader, criterion, binary_loss)

        # Calculate metrics
        acc, prec, rec, f1_score, conf_matrix = calculate_metrics(val_preds, val_labels, model_settings)

        # Logging
        print(f'Epoch: {epoch + 1}, Train_Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        logger.log({'train_loss': train_loss})
        logger.log({'validation_loss': val_loss})
        logger.log({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1_score})
        f, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(conf_matrix.clone().detach().cpu().numpy(), annot=True, ax=ax)
        logger.log({"confusion_matrix": wandb.Image(f)})
        plt.close(f)

        # Save checkpoint if improvement
        if val_loss < min_loss:
            print(f'Loss decreased ({min_loss:.4f} --> {val_loss:.4f}). Saving model ...')
            ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict()}
            torch.save(ckpt, f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_multi_binary={data_settings['binary']}_contrastive={data_settings['contrastive']}.pth")
            ckpt_head = {'epoch': epoch, 'model_state_dict': model_head.state_dict()}
            torch.save(ckpt_head, f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_multi_head_binary={data_settings['binary']}_contrastive={data_settings['contrastive']}.pth")
            min_loss = val_loss

def train_loop(model, model_head, train_loader, criterion, optimizer, optimizer_head, binary_loss):
    running_loss = 0.0
    for images, labels, AI in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_head.zero_grad()
        if model.training: # If model is in training mode
            optimizer.zero_grad()
            outputs, _ = model(images)
        else:
            with torch.no_grad():
                outputs, _ = model(images)
        outputs = model_head(outputs)
        labels = labels.type(torch.LongTensor).to(device)
        if binary_loss:
            loss = criterion(outputs[:, 1], labels.float())
            # one_hot_labels = F.one_hot(labels, num_classes=2).to(device)
            # original_tensor = torch.FloatTensor([1.0, 5.0]).to(device)
            # batch_tensor = original_tensor.repeat(one_hot_labels.size(0), 1).to(device)
            # loss = weighted_bce_loss(outputs, one_hot_labels.float(), original_tensor)
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer_head.step()
        if model.training:
            optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate_loop(model, model_head, val_loader, criterion, binary_loss):
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels, AI in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
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

def main():
    config = load_config()

    data_settings = config['data_settings']
    model_setting = config['model']
    train_setting = config['train']

    wandb_logger = Logger(
        f"EfficientContrastive",
        project='INM705_Final-Results')
    logger = wandb_logger.get_logger()


    if data_settings['binary']: model_setting['num_classes']=2 # Force binary classification if binary setting is True

    print("\n############## DATA SETTINGS ##############")
    print(data_settings)
    print()
    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    print("\n############## TRAIN SETTINGS ##############")
    print(train_setting)
    print()

    # Define classification dataset
    dataset = WikiArtDataset(data_dir=data_settings['dataset_path'], binary=data_settings['binary'])
    train_size = int(0.8 * len(dataset))  # 80% training set
    class_train_dataset, class_val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    if data_settings['contrastive']:
        assert data_settings['binary']==True, f"Only binary setting True is supported for contrastive"

        # The classifier head will be trained automatically after the contrastive learning of the encoder
        contrastive_learning(class_train_dataset,
                             class_val_dataset,
                             data_settings,
                             model_setting,
                             train_setting,
                             logger,
                             criterion='contloss')
    else:
        train(class_train_dataset, 
              class_val_dataset, 
              data_settings, 
              model_setting, 
              train_setting, 
              logger,
              contrastive=False)

if __name__ == '__main__':
    main()