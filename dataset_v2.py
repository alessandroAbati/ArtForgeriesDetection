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

    artists = ["boris-kustodiev", "camille-pissarro", "childe-hassam", "claude-monet", "edgar-degas", "eugene-boudin", "gustave-dore", "ilya-repin", "ivan-aivazovsky", "ivan-shishkin", "john-singer-sargent", "marc-chagall", "martiros-saryan", "nicholas-roerich", "pablo-picasso", "paul-cezanne", "pierre-auguste-renoir", "pyotr-konchalovsky", "raphael-kirchner", "rembrandt", "salvador-dali", "vincent-van-gogh", "hieronymus-bosch", "leonardo-da-vinci", "albrecht-durer", "edouard-cortes", "sam-francis", "juan-gris", "lucas-cranach-the-elder", "paul-gauguin", "konstantin-makovsky", "egon-schiele", "thomas-eakins", "gustave-moreau", "francisco-goya", "edvard-munch", "henri-matisse", "fra-angelico", "maxime-maufra", "jan-matejko", "mstislav-dobuzhinsky", "alfred-sisley", "mary-cassatt", "gustave-loiseau", "fernando-botero", "zinaida-serebriakova", "georges-seurat", "isaac-levitan", "joaqu\u00e3\u00adn-sorolla", "jacek-malczewski", "berthe-morisot", "andy-warhol", "arkhip-kuindzhi", "niko-pirosmani", "james-tissot", "vasily-polenov", "valentin-serov", "pietro-perugino", "pierre-bonnard", "ferdinand-hodler", "bartolome-esteban-murillo", "giovanni-boldini", "henri-martin", "gustav-klimt", "vasily-perov", "odilon-redon", "tintoretto", "gene-davis", "raphael", "john-henry-twachtman", "henri-de-toulouse-lautrec", "antoine-blanchard", "david-burliuk", "camille-corot", "konstantin-korovin", "ivan-bilibin", "titian", "maurice-prendergast", "edouard-manet", "peter-paul-rubens", "aubrey-beardsley", "paolo-veronese", "joshua-reynolds", "kuzma-petrov-vodkin", "gustave-caillebotte", "lucian-freud", "michelangelo", "dante-gabriel-rossetti", "felix-vallotton", "nikolay-bogdanov-belsky", "georges-braque", "vasily-surikov", "fernand-leger", "konstantin-somov", "katsushika-hokusai", "sir-lawrence-alma-tadema", "vasily-vereshchagin", "ernst-ludwig-kirchner", "mikhail-vrubel", "orest-kiprensky", "william-merritt-chase", "aleksey-savrasov", "hans-memling", "amedeo-modigliani", "ivan-kramskoy", "utagawa-kuniyoshi", "gustave-courbet", "william-turner", "theo-van-rysselberghe", "joseph-wright", "edward-burne-jones", "koloman-moser", "viktor-vasnetsov", "anthony-van-dyck", "raoul-dufy", "frans-hals", "hans-holbein-the-younger", "ilya-mashkov", "henri-fantin-latour", "m.c.-escher", "el-greco", "mikalojus-ciurlionis", "james-mcneill-whistler", "karl-bryullov", "jacob-jordaens", "thomas-gainsborough", "eugene-delacroix", "canaletto"]
    label_to_artist = {i + 1: artist for i, artist in enumerate(artists)}

    def __init__(self, data_dir, img_max_size=(512, 512), transform=None, binary=False, contrastive=False, contrastive_batch_size=4):
        """
        Args:
            data_dir (string): Directory with parquet files (not used for images, only for reading parquet files).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.contrastive = contrastive
        self.contrastive_batch_size = contrastive_batch_size
        self.img_max_size = img_max_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_max_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
            ])
        else: 
            self.transform = transform

        # Use glob to find all parquet files in the directory
        # Create batches
        # parquet_files = glob.glob(os.path.join(data_dir, 'train-0**.parquet'))
        # For filtered batches
        parquet_files = glob.glob(os.path.join(data_dir, 'batch*.parquet'))


        # Read each parquet file and concatenate them into a single DataFrame for multiclass
        if not binary:
            self.data_frame = pd.concat([pd.read_parquet(file, filters=[[('artist',"in",[22,16,4,2,13,17,3,18,6,15])]]) for file in parquet_files], ignore_index=True )
        # Van Gogh only
        else:
            self.data_frame = pd.concat([pd.read_parquet(file, filters=[[('artist',"in",[22])]]) for file in parquet_files], ignore_index=True )
        # Remove rows with label=0 ('Unknown artist')
        self.data_frame.drop(self.data_frame.index[self.data_frame['artist'] == 0], inplace=True)
        
        # Cast to 0-10 integers label
        self.data_frame['label'] = np.nan
        self.data_frame['AI'] = False

        self.data_frame.loc[self.data_frame['artist']==22, 'label'] = 0
        if not binary:
            self.data_frame.loc[self.data_frame['artist']==16, 'label'] = 1
            self.data_frame.loc[self.data_frame['artist']==4, 'label'] = 2
            self.data_frame.loc[self.data_frame['artist']==2, 'label'] = 3
            self.data_frame.loc[self.data_frame['artist']==13, 'label'] = 4
            self.data_frame.loc[self.data_frame['artist']==17,'label'] = 5
            self.data_frame.loc[self.data_frame['artist']==3, 'label'] = 6
            self.data_frame.loc[self.data_frame['artist']==18, 'label'] = 7
            self.data_frame.loc[self.data_frame['artist']==6, 'label'] = 8
            self.data_frame.loc[self.data_frame['artist']==15, 'label'] = 9
        print(self.data_frame.head())
        print(len(self.data_frame))

        if binary:
        # Read in Vincent Fakes
            for filename in os.listdir('AI/'):
                filepath = os.path.join('AI/', filename)
                if os.path.isfile(filepath):
                    # image = Image.open(filepath).convert('RGB').tobytes()
                    new_row = self.data_frame.iloc[0].copy()
                    new_row['image'] = {'path': filepath}
                    new_row['label'] = 1.0
                    new_row['AI'] = True

                    self.data_frame = pd.concat([self.data_frame, pd.DataFrame([new_row])], ignore_index=True)

            for filename in os.listdir('Forgery/'):
                filepath = os.path.join('Forgery/', filename)
                if os.path.isfile(filepath):
                    # image = Image.open(filepath).convert('RGB').tobytes()
                    new_row = self.data_frame.iloc[0].copy()
                    new_row['image'] = {'path': filepath}
                    new_row['label'] = 1.0
                    new_row['AI'] = False

                    self.data_frame = pd.concat([self.data_frame, pd.DataFrame([new_row])], ignore_index=True)

        print(self.data_frame.tail())
        print(len(self.data_frame))
        count_forgery = self.data_frame['label'].value_counts().get(1.0, 0)
        print(count_forgery)
        count_real = self.data_frame['label'].value_counts().get(0.0, 0)
        print(count_real)
        print(f"Proportion of Forgery/AI: {count_forgery/1996}, {count_real/1996}")

        if self.contrastive:
            self.minority_data = self.data_frame[self.data_frame['label'] == 1].reset_index(drop=True)
            self.majority_data = self.data_frame[self.data_frame['label'] == 0].reset_index(drop=True)
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.5),
            ])

        # Creating batches
        # self.data_frame.to_parquet(f'wikiart_data_batches/data_batches_filtered/van{0}.parquet')

    def __len__(self):
        if self.contrastive:
            return len(self.minority_data)  # As many batches as minority samples
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.contrastive:
            # Get the anchor using idx
            if self.minority_data.iloc[idx]['image']['path']:
                anchor_image_path = self.minority_data.iloc[idx]['image']['path']
                anchor_image = Image.open(anchor_image_path).convert('RGB')
            else:
                anchor_image_bytes = self.minority_data.iloc[idx]['image']['bytes']
                image = Image.open(io.BytesIO(anchor_image_bytes)).convert('RGB')

            anchor_label = [self.minority_data.iloc[idx]['label'].astype(int)]
            pos_idx = np.random.choice(self.minority_data.index, 1, replace=False)[0]
            # print(pos_idx)
            # print(self.minority_data.index)
            # print(self.minority_data.columns)
            # print(self.minority_data)

            positive_image = Image.open(self.minority_data.iloc[pos_idx]['image']['path']).convert('RGB') if self.minority_data.iloc[pos_idx]['image']['path'] else Image.open(io.BytesIO(self.minority_data.iloc[pos_idx]['image']['bytes'])).convert('RGB')
            positive_anchor_image = self.augmentation(anchor_image) # Positive sample (augmented anchor)
            positive_image_pos = self.augmentation(positive_image) # Positive sample (augmented anchor)
            positive_label = anchor_label

            # Select two random negative samples
            negative_indices = np.random.choice(self.majority_data.index, self.contrastive_batch_size-2, replace=False)
            negative_images = [Image.open(self.majority_data.iloc[neg_idx]['image']['path']).convert('RGB') if self.majority_data.iloc[neg_idx]['image']['path'] else Image.open(io.BytesIO(self.majority_data.iloc[neg_idx]['image']['bytes'])).convert('RGB') for neg_idx in negative_indices]
            negative_labels = [self.majority_data.iloc[neg_idx]['label'].astype(int) for neg_idx in negative_indices]

            # Apply transformations
            anchor_image = self.transform(anchor_image)
            positive_anchor_image = self.transform(positive_anchor_image)
            positive_image_pos = self.transform(positive_image_pos)
            negative_images = [self.transform(neg_image) for neg_image in negative_images]

            # Combine all samples into one batch
            images = torch.stack([anchor_image, positive_anchor_image, positive_image_pos] + negative_images)
            labels = anchor_label + positive_label + positive_label + negative_labels
            return images, labels

        if torch.is_tensor(idx):
            idx = idx.tolist()
        # For Vincent Fakes
        if self.data_frame.iloc[idx]['image']['path']:
            filepath = self.data_frame.iloc[idx]['image']['path']
            image = Image.open(filepath).convert('RGB')
            # plt.imshow(image)
            # plt.show()
        else:
        # Get the image bytes in the dataframe 'image' column (dictonary with the key 'bytes')
            image_bytes = self.data_frame.iloc[idx]['image']['bytes']
            # Convert byte data to Image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Creating label variable using the column 'label'
        label = self.data_frame.iloc[idx]['label'].astype(int)

        # Apply transformations
        image = self.transform(image)

        AI = self.data_frame.iloc[idx]['AI']
        
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
    
    #Contarstive learning daatset test
    dataset = WikiArtDataset(data_dir=os.path.join('wikiart_data_batches', 'data_batches_filtered'), binary=True, contrastive=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in dataloader:
        images, labels = batch
        images = images.squeeze(0)
        labels = torch.stack(labels, dim=0).reshape(len(labels))
        print(f"Images: {images.shape}\nlabels: {labels.shape}\n{labels}") 
        break
