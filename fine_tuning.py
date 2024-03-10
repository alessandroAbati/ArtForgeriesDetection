import torch
from torch.utils.data import DataLoader, random_split
from utils import load_config
from models import ResNetModel, EfficientNetModel
from dataset import WikiArtDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model_settings, train_settings):
    # Dataset
    dataset = WikiArtDataset(mode='train', load_data=True)  # Add parameters as needed
    train_size = int(0.8 * len(dataset)) # 80% training set
    train_dataset, _ = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=train_settings['batch_size'], shuffle=True)

    # Model
    if model_settings['model_type'] == 'resnet':
        model = ResNetModel(num_classes=model_settings['num_classes']).to(device)
    elif model_settings['model_type'] == 'efficientnet':
        model = EfficientNetModel(num_classes=model_settings['num_classes']).to(device)
    else:
        raise ValueError("Model type in config.yaml should be 'resnet' or 'efficientnet'")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings['learning_rate'])

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Logger
    # TBD

    # Train loop
    train_loop(model, train_loader, criterion, optimizer, model_settings, train_settings)

def train_loop(model, train_loader, criterion, optimizer, model_settings, train_settings):
    for epoch in range(train_settings['epochs']):
        model.train()
        min_loss = float('inf')
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f'Training Epoch: {epoch+1}, Loss: {avg_loss:.4f}')

        # Save checkpoint if improvement
        if avg_loss < min_loss:
            print(f'Loss decreased ({min_loss:.4f} --> {avg_loss:.4f}). Saving model ...')
            torch.save(model.state_dict(), f"{model_settings['checkpoint_folder']}/{model_settings['model_type']}_ckt_{epoch+1}.pth")
            min_loss = avg_loss

def validate(model_settings, train_settings):
    # Dataset
    dataset = WikiArtDataset(load_data=True)
    _, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    val_loader = DataLoader(val_dataset, batch_size=train_settings['batch_size'], shuffle=False)

    # Model
    if model_settings['model_type'] == 'resnet':
        model = ResNetModel(num_classes=model_settings['num_classes']).to(device)
    elif model_settings['model_type'] == 'efficientnet':
        model = EfficientNetModel(num_classes=model_settings['num_classes']).to(device)
    else:
        raise ValueError("Model type in config.yaml should be 'resnet' or 'efficientnet'")

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Logger
    # TBD

    # Validate loop
    validate_loop(model, val_loader, criterion)

def validate_loop(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    avg_loss = running_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}')

def main():
    config = load_config()

    model_setting = config['model']
    train_setting = config['fine_tuning']
    
    train(model_setting, train_setting)
    validate(model_setting, train_setting)

if __name__ == '__main__':
    main()
