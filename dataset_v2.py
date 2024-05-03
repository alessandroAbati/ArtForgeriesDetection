import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import io
import glob
import os
from torch.utils.data import random_split
from torchvision import transforms
import numpy as np

torch.manual_seed(42)

class WikiArtDataset(Dataset):
    artists = ["boris-kustodiev", "camille-pissarro", "childe-hassam", "claude-monet", "edgar-degas", "eugene-boudin",
               "gustave-dore", "ilya-repin", "ivan-aivazovsky", "ivan-shishkin", "john-singer-sargent", "marc-chagall",
               "martiros-saryan", "nicholas-roerich", "pablo-picasso", "paul-cezanne", "pierre-auguste-renoir",
               "pyotr-konchalovsky", "raphael-kirchner", "rembrandt", "salvador-dali", "vincent-van-gogh",
               "hieronymus-bosch", "leonardo-da-vinci", "albrecht-durer", "edouard-cortes", "sam-francis", "juan-gris",
               "lucas-cranach-the-elder", "paul-gauguin", "konstantin-makovsky", "egon-schiele", "thomas-eakins",
               "gustave-moreau", "francisco-goya", "edvard-munch", "henri-matisse", "fra-angelico", "maxime-maufra",
               "jan-matejko", "mstislav-dobuzhinsky", "alfred-sisley", "mary-cassatt", "gustave-loiseau",
               "fernando-botero", "zinaida-serebriakova", "georges-seurat", "isaac-levitan",
               "joaqu\u00e3\u00adn-sorolla", "jacek-malczewski", "berthe-morisot", "andy-warhol", "arkhip-kuindzhi",
               "niko-pirosmani", "james-tissot", "vasily-polenov", "valentin-serov", "pietro-perugino",
               "pierre-bonnard", "ferdinand-hodler", "bartolome-esteban-murillo", "giovanni-boldini", "henri-martin",
               "gustav-klimt", "vasily-perov", "odilon-redon", "tintoretto", "gene-davis", "raphael",
               "john-henry-twachtman", "henri-de-toulouse-lautrec", "antoine-blanchard", "david-burliuk",
               "camille-corot", "konstantin-korovin", "ivan-bilibin", "titian", "maurice-prendergast", "edouard-manet",
               "peter-paul-rubens", "aubrey-beardsley", "paolo-veronese", "joshua-reynolds", "kuzma-petrov-vodkin",
               "gustave-caillebotte", "lucian-freud", "michelangelo", "dante-gabriel-rossetti", "felix-vallotton",
               "nikolay-bogdanov-belsky", "georges-braque", "vasily-surikov", "fernand-leger", "konstantin-somov",
               "katsushika-hokusai", "sir-lawrence-alma-tadema", "vasily-vereshchagin", "ernst-ludwig-kirchner",
               "mikhail-vrubel", "orest-kiprensky", "william-merritt-chase", "aleksey-savrasov", "hans-memling",
               "amedeo-modigliani", "ivan-kramskoy", "utagawa-kuniyoshi", "gustave-courbet", "william-turner",
               "theo-van-rysselberghe", "joseph-wright", "edward-burne-jones", "koloman-moser", "viktor-vasnetsov",
               "anthony-van-dyck", "raoul-dufy", "frans-hals", "hans-holbein-the-younger", "ilya-mashkov",
               "henri-fantin-latour", "m.c.-escher", "el-greco", "mikalojus-ciurlionis", "james-mcneill-whistler",
               "karl-bryullov", "jacob-jordaens", "thomas-gainsborough", "eugene-delacroix", "canaletto"]
    label_to_artist = {i + 1: artist for i, artist in enumerate(artists)}

    def __init__(self, data_dir, img_size=(512, 512), transform=None, binary=False, 
                 contrastive=False, contrastive_batch_size=4, index=None, test=False):
        """
        Args:
            data_dir (string): Directory with wikiart dataset files.
            img_size (int, int): Desired image size.
            transform (callable, optional): Optional transform to be applied on a sample.
            binary (bool, optional): Parameter to select multiclass or binary data, default is multiclass.
            contrastive (bool, optional): Parameter to select contrastive dataset, default is false.
            contrastive_batch_size (int, optional): Batch size for the contrastive learning
            index (list, optional): Index used in train/val split for the contrastive learning
            test (bool, optional): Parameter to select test dataset, default is false.
        """
        self.test = test
        self.contrastive = contrastive
        self.contrastive_batch_size = contrastive_batch_size
        self.img_size = img_size
        self.index = index

        if transform is None:
            if contrastive:
                self.transform = transforms.Compose([
                        transforms.Resize(self.img_size),
                        transforms.RandomCrop(320, 320),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(5),
                        transforms.ToTensor(),
                    ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transform

        parquet_files = glob.glob(os.path.join(data_dir, 'batch*.parquet')) # Get wikiart parquet files

        if test:
            self.data_frame = pd.DataFrame({'image': [None], 'label': [None], 'AI': [None]})
            for filename in os.listdir('test/fake'): # Get new data from test/fake folder
                filepath = os.path.join('test/fake', filename)
                if os.path.isfile(filepath):
                    new_row = self.data_frame.iloc[0].copy()
                    new_row['image'] = {'path': filepath}
                    new_row['label'] = 1.0
                    if 'ai' in filename:
                        new_row['AI'] = True
                    else:
                        new_row['AI'] = False
                    self.data_frame = pd.concat([self.data_frame, pd.DataFrame([new_row])], ignore_index=True) # Concatenate new data

            for filename in os.listdir('test/real'): # Get new data from Forgery folder ("real" forgeries)
                filepath = os.path.join('test/real', filename)
                if os.path.isfile(filepath):
                    new_row = self.data_frame.iloc[0].copy()
                    new_row['image'] = {'path': filepath}
                    new_row['label'] = 0.0
                    new_row['AI'] = False
                    self.data_frame = pd.concat([self.data_frame, pd.DataFrame([new_row])], ignore_index=True)
            self.data_frame.drop(index=0)
            self.data_frame.reset_index(drop=True)
            print(f"Test dataset: {self.data_frame.shape}\n{self.data_frame}")
            return

        if not binary:
            # Read each parquet file and concatenate them into a single DataFrame for multiclass
            self.data_frame = pd.concat(
                [pd.read_parquet(file, filters=[[('artist', "in", [22, 16, 4, 2, 13, 17, 3, 18, 6, 15])]]) for file in
                 parquet_files], ignore_index=True)
        else:
            # Van Gogh only
            self.data_frame = pd.concat(
                [pd.read_parquet(file, filters=[[('artist', "in", [22])]]) for file in parquet_files],
                ignore_index=True)
            
        self.data_frame.drop(self.data_frame.index[self.data_frame['artist'] == 0], inplace=True) # Remove rows with label=0 ('Unknown artist')

        self.data_frame['label'] = np.nan # Create label column
        self.data_frame['AI'] = False # Create AI label column -> indicates if the forgery is AI generated ot not 

        # Cast to 0-9 integer labels
        self.data_frame.loc[self.data_frame['artist'] == 22, 'label'] = 0
        if not binary:
            self.data_frame.loc[self.data_frame['artist'] == 16, 'label'] = 1
            self.data_frame.loc[self.data_frame['artist'] == 4, 'label'] = 2
            self.data_frame.loc[self.data_frame['artist'] == 2, 'label'] = 3
            self.data_frame.loc[self.data_frame['artist'] == 13, 'label'] = 4
            self.data_frame.loc[self.data_frame['artist'] == 17, 'label'] = 5
            self.data_frame.loc[self.data_frame['artist'] == 3, 'label'] = 6
            self.data_frame.loc[self.data_frame['artist'] == 18, 'label'] = 7
            self.data_frame.loc[self.data_frame['artist'] == 6, 'label'] = 8
            self.data_frame.loc[self.data_frame['artist'] == 15, 'label'] = 9

        if binary: # Get additional data for the binary dataset
    
            for filename in os.listdir('AI/'): # Get new data from AI folder (AI generated forgeries)
                filepath = os.path.join('AI/', filename)
                if os.path.isfile(filepath):
                    new_row = self.data_frame.iloc[0].copy()
                    new_row['image'] = {'path': filepath}
                    new_row['label'] = 1.0
                    new_row['AI'] = True
                    self.data_frame = pd.concat([self.data_frame, pd.DataFrame([new_row])], ignore_index=True) # Concatenate new data

            for filename in os.listdir('Forgery/'): # Get new data from Forgery folder ("real" forgeries)
                filepath = os.path.join('Forgery/', filename)
                if os.path.isfile(filepath):
                    new_row = self.data_frame.iloc[0].copy()
                    new_row['image'] = {'path': filepath}
                    new_row['label'] = 1.0
                    new_row['AI'] = False
                    self.data_frame = pd.concat([self.data_frame, pd.DataFrame([new_row])], ignore_index=True)

            

        # Filtering the dataset based on index
        if index:
            self.data_frame = self.data_frame.loc[index]
        
        print(f"Dataset_binary={binary}_contrastive={contrastive} dimension: {len(self.data_frame)}")

        if binary:
            # Calculate proportions
            self.count_forgery = self.data_frame['label'].value_counts().get(1.0, 0)
            self.count_real = self.data_frame['label'].value_counts().get(0.0, 0)
            print(f"Proportion of Forgery/AI: {self.count_forgery / len(self.data_frame)}, {self.count_real / len(self.data_frame)}")

        if self.contrastive: # Prepare dataframes for contrastive learning (if selected)
            self.minority_data = self.data_frame[self.data_frame['label'] == 1].reset_index(drop=True)
            self.majority_data = self.data_frame[self.data_frame['label'] == 0].reset_index(drop=True)
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.5),
            ])
            print(f"Contrastive Dataset --- Minority class: {self.minority_data.shape} --- Majority class: {self.majority_data.shape}")

        # Creating batches
        # self.data_frame.to_parquet(f'wikiart_data_batches/data_batches_filtered/van{0}.parquet')

    def __len__(self):
        if self.contrastive: # If contrastive is selected the lenght will be the number of samples of the minority class
            return len(self.minority_data)
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.contrastive:
            # Get the anchor using idx
            if self.minority_data.iloc[idx]['image']['path']:
                anchor_image_path = self.minority_data.iloc[idx]['image']['path']
                anchor_image = Image.open(anchor_image_path).convert('RGB')
            else:
                anchor_image_bytes = self.minority_data.iloc[idx]['image']['bytes']
                image = Image.open(io.BytesIO(anchor_image_bytes)).convert('RGB')
            anchor_label = [self.minority_data.iloc[idx]['label'].astype(int)]

            # First positive sample
            pos_idx = np.random.choice(self.minority_data.index, 1, replace=False)[0] # Get 1 sample from the minority class
            positive_image = Image.open(self.minority_data.iloc[pos_idx]['image']['path']).convert('RGB') if \
            self.minority_data.iloc[pos_idx]['image']['path'] else Image.open(
                io.BytesIO(self.minority_data.iloc[pos_idx]['image']['bytes'])).convert('RGB')
            
            # Second positive sample (augmented anchor)
            positive_augmented_anchor = self.augmentation(anchor_image)

            # Third positive sample (augmented positive)
            positive_augmented_positive = self.augmentation(positive_image) 

            # Negative samples
            negative_indices = np.random.choice(self.majority_data.index, self.contrastive_batch_size - 4, replace=False) # Select self.contrastive_batch_size - 4 random negative samples
            negative_images = [Image.open(self.majority_data.iloc[neg_idx]['image']['path']).convert('RGB') if
                               self.majority_data.iloc[neg_idx]['image']['path'] else Image.open(
                io.BytesIO(self.majority_data.iloc[neg_idx]['image']['bytes'])).convert('RGB') for neg_idx in
                               negative_indices]
            negative_labels = [self.majority_data.iloc[neg_idx]['label'].astype(int) for neg_idx in negative_indices]

            # Apply transformations
            anchor_image = self.transform(anchor_image)
            positive_augmented_anchor = self.transform(positive_augmented_anchor)
            positive_image = self.transform(positive_image)
            positive_augmented_positive = self.transform(positive_augmented_positive)
            negative_images = [self.transform(neg_image) for neg_image in negative_images]

            # Combine all samples into one batch
            images = torch.stack([anchor_image, positive_augmented_anchor, positive_image, positive_augmented_positive] + negative_images)
            labels = anchor_label + anchor_label + anchor_label + anchor_label + negative_labels
            return images, labels

        # Get the images
        if self.data_frame.iloc[idx]['image']['path']: # For AI and Forgery images we use the path
            filepath = self.data_frame.iloc[idx]['image']['path']
            image = Image.open(filepath).convert('RGB')
        else:
            image_bytes = self.data_frame.iloc[idx]['image']['bytes'] # Get the image bytes in the dataframe 'image' column (dictonary with the key 'bytes')
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB') # Convert byte data to Image

        # Creating label variable using the column 'label'
        label = self.data_frame.iloc[idx]['label'].astype(int)

        # Creating AI (label) variable using the column 'AI' 
        AI = self.data_frame.iloc[idx]['AI']

        # Apply transformations
        image = self.transform(image)

        return image, label, AI


if __name__ == "__main__":

    """dataset = WikiArtDataset(data_dir=os.path.join('wikiart_data_batches', 'data_batches_filtered'))  # Add your image transformations if needed
    # dataset = WikiArtDataset(data_dir='wikiart_data_batches/batch3')  # Add your image transformations if needed
    train_len = int(len(dataset) * 0.8)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False)

    print(len(dataset))
    for batch in train_dataloader:
        images, label, AI = batch
        #print(images)
        print(f"Label {label}")
        break"""

    """# Contarstive learning daatset test
    dataset = WikiArtDataset(data_dir=os.path.join('wikiart_data_batches', 'data_batches_filtered'), binary=True,
                             contrastive=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in dataloader:
        images, labels = batch
        images = images.squeeze(0)
        labels = torch.stack(labels, dim=0).reshape(len(labels))
        print(f"Images: {images.shape}\nlabels: {labels.shape}\n{labels}")
        break"""
    
    dataset = WikiArtDataset(data_dir=os.path.join('wikiart_data_batches', 'data_batches_filtered'), test=True)