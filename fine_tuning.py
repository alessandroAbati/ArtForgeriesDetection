import torch
import os
from torch.utils.data import DataLoader, random_split
from utils import load_config
from models import ResNetModel, EfficientNetModel
from logger import Logger
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from contrastive_losses import SupContLoss
from dataset_v2 import WikiArtDataset

torch.manual_seed(42)

# os.environ['https_proxy'] = "http://hpc-proxy00.city.ac.uk:3128" # Proxy to train with hyperion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def contrastive_learning(data_settings, model_settings, train_settings, logger):
    # Dataset
    dataset = WikiArtDataset(data_dir=data_settings['dataset_path'], binary=data_settings['binary'], contrastive=data_settings['contrastive'], contrastive_batch_size=data_settings['contrastive_batch_size'])  # Add parameters as needed
    train_size = int(0.8 * len(dataset)) # 80% training set
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    print(f"Length Train dataset: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Model
    if model_settings['model_type'] == 'resnet':
        raise ValueError("resnet is not supported for contrastive learning, please change the model in the config file")
    elif model_settings['model_type'] == 'efficientnet':
        model = EfficientNetModel(num_classes=model_settings['num_classes'], 
                                  checkpoint_path=None, 
                                  binary_classification=model_settings['binary'], 
                                  contrastive_learning=True).to(device)
        print("Model loaded")
    else:
        raise ValueError("Model type in config.yaml should be 'resnet' or 'efficientnet'")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['learning_rate'])

    # Loading checkpoint of the first fine-tuning
    ckpt = torch.load(f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_fine.pth", map_location=device)
    model_weights = ckpt['model_weights']
    for name, param in model.named_parameters():
        if "fc" not in name:  # Exclude final fully connected layer
            param.data = model_weights[name]

    # Contrastive Loss
    criterion = SupContLoss(temperature=0.1)

    # Training loop
    min_loss = float('inf')
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.squeeze(0) # Squeeze to shape [contrastive_batch, img_width, img_hight]
            labels = torch.stack(labels, dim=0).reshape(len(labels)) # Reshape labels to tensor of shape [contrastive_batch]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            #print(f"Output: {outputs.shape}")
            labels = labels.type(torch.LongTensor).to(device)
            loss = criterion(outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch: {epoch+1}, Contrastive_Loss: {avg_loss:.4f}')
        # Save checkpoint if improvement
        if avg_loss < min_loss:
            print(f'Saving model ...')
            ckpt = {'epoch': epoch, 'model_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
            torch.save(ckpt, f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_contrastive.pth")
            min_loss = avg_loss

    

def train(data_settings, model_settings, train_settings, logger):
    # Dataset
    dataset = WikiArtDataset(data_dir=data_settings['dataset_path'], binary=data_settings['binary'])  # Add parameters as needed
    train_size = int(0.8 * len(dataset)) # 80% training set
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    print(f"Length Train dataset: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=train_settings['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=False)

    # Model
    if model_settings['model_type'] == 'resnet':
        model = ResNetModel(resnet_version='resnet101',num_classes=model_settings['num_classes']).to(device)
    elif model_settings['model_type'] == 'efficientnet':
        model = EfficientNetModel(num_classes=model_settings['num_classes'], checkpoint_path=None, binary_classification=model_settings['binary']).to(device)
        print("Model loaded")
    else:
        raise ValueError("Model type in config.yaml should be 'resnet' or 'efficientnet'")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['learning_rate'])

    # Loading checkpoint
    epoch_start = 0
    binary_loss = False
    if model_settings['continue_train']:
        ckpt = torch.load(f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}.pth", map_location=device)
        model_weights = ckpt['model_weights']
        model.load_state_dict(model_weights)
        optimizer_state = ckpt['optimizer_state']
        optimizer.load_state_dict(optimizer_state)
        epoch_start = ckpt['epoch']
        print("Model's pretrained weights loaded!")
    if model_settings['binary']:
        ckpt = torch.load(f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_fine.pth", map_location=device)
        model_weights = ckpt['model_weights']
        for name, param in model.named_parameters():
            if "fc" not in name:  # Exclude final fully connected layer
                param.data = model_weights[name]
        # model.load_state_dict(model_weights)
        print("Model's pretrained weights loaded!")
        binary_loss = True

    # Loss
    if model_settings['binary']:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()


    # Training and validation loop
    min_loss = float('inf')
    for epoch in range(epoch_start, train_settings['epochs']):
        train_loss = train_loop(model, train_loader, criterion, optimizer, binary_loss)
        val_loss, val_preds, val_labels = validate_loop(model, val_loader, criterion, binary_loss)

        # Calculate metrics
        acc, prec, rec, f1_score, conf_matrix = calculate_metrics(val_preds, val_labels, model_settings)

        print(f'Epoch: {epoch+1}, Train_Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        logger.log({'train_loss': train_loss}) 
        logger.log({'validation_loss': val_loss})
        logger.log({'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1_score})
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #cax = ax.matshow(conf_matrix.clone().detach().cpu().numpy(), cmap='bone')
        #fig.colorbar(cax)

        # Save the figure to a wandb artifact
        #wandb.log({"confusion_matrix": wandb.Image(fig)})

    	# Close the figure to prevent it from being displayed in the notebook
        #plt.close(fig)
        f, ax = plt.subplots(figsize = (15,10)) 
        sns.heatmap(conf_matrix.clone().detach().cpu().numpy(), annot=True, ax=ax)
        logger.log({"confusion_matrix": wandb.Image(f) })
        plt.close(f)
        # Save checkpoint if improvement
        if val_loss < min_loss:
            print(f'Loss decreased ({min_loss:.4f} --> {val_loss:.4f}). Saving model ...')
            ckpt = {'epoch': epoch, 'model_weights': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
            if binary_loss:
                torch.save(ckpt, f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_binary.pth")
            else:
                torch.save(ckpt, f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}.pth")
            min_loss = val_loss

def train_loop(model, train_loader, criterion, optimizer, binary_loss):
    model.train()
    running_loss = 0.0
    for images, labels, AI in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.type(torch.LongTensor).to(device)
        if binary_loss:
            loss = criterion(outputs[:,1], labels.float())
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    return avg_loss   

def validate_loop(model, val_loader, criterion, binary_loss):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels, AI in val_loader:
            images, labels = images.to(device), labels.to(device)
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
        precision = Precision(task='binary', average='macro', num_classes=model_settings['num_classes']).to(device)
        recall = Recall(task='binary', average='macro', num_classes=model_settings['num_classes']).to(device)
        f1 = F1Score(task='binary', average='macro', num_classes=model_settings['num_classes']).to(device)
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
    
def main():
    config = load_config()

    data_setting = config['data_settings']
    model_setting = config['model']
    train_setting = config['fine_tuning']

    wandb_logger = Logger(
        f"finertuning_efficentnetb0_lr=0.0001_",
        project='ArtForg')
    logger = wandb_logger.get_logger()

    print("\n############## MODEL SETTINGS ##############")
    print(model_setting)
    print()
    
    #train(data_setting, model_setting, train_setting, logger)
    contrastive_learning(data_setting, model_setting, train_setting, logger)

if __name__ == '__main__':
    main()
