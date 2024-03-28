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

class WikiArtDataset(Dataset):

    artists = ["boris-kustodiev", "camille-pissarro", "childe-hassam", "claude-monet", "edgar-degas", "eugene-boudin", "gustave-dore", "ilya-repin", "ivan-aivazovsky", "ivan-shishkin", "john-singer-sargent", "marc-chagall", "martiros-saryan", "nicholas-roerich", "pablo-picasso", "paul-cezanne", "pierre-auguste-renoir", "pyotr-konchalovsky", "raphael-kirchner", "rembrandt", "salvador-dali", "vincent-van-gogh", "hieronymus-bosch", "leonardo-da-vinci", "albrecht-durer", "edouard-cortes", "sam-francis", "juan-gris", "lucas-cranach-the-elder", "paul-gauguin", "konstantin-makovsky", "egon-schiele", "thomas-eakins", "gustave-moreau", "francisco-goya", "edvard-munch", "henri-matisse", "fra-angelico", "maxime-maufra", "jan-matejko", "mstislav-dobuzhinsky", "alfred-sisley", "mary-cassatt", "gustave-loiseau", "fernando-botero", "zinaida-serebriakova", "georges-seurat", "isaac-levitan", "joaqu\u00e3\u00adn-sorolla", "jacek-malczewski", "berthe-morisot", "andy-warhol", "arkhip-kuindzhi", "niko-pirosmani", "james-tissot", "vasily-polenov", "valentin-serov", "pietro-perugino", "pierre-bonnard", "ferdinand-hodler", "bartolome-esteban-murillo", "giovanni-boldini", "henri-martin", "gustav-klimt", "vasily-perov", "odilon-redon", "tintoretto", "gene-davis", "raphael", "john-henry-twachtman", "henri-de-toulouse-lautrec", "antoine-blanchard", "david-burliuk", "camille-corot", "konstantin-korovin", "ivan-bilibin", "titian", "maurice-prendergast", "edouard-manet", "peter-paul-rubens", "aubrey-beardsley", "paolo-veronese", "joshua-reynolds", "kuzma-petrov-vodkin", "gustave-caillebotte", "lucian-freud", "michelangelo", "dante-gabriel-rossetti", "felix-vallotton", "nikolay-bogdanov-belsky", "georges-braque", "vasily-surikov", "fernand-leger", "konstantin-somov", "katsushika-hokusai", "sir-lawrence-alma-tadema", "vasily-vereshchagin", "ernst-ludwig-kirchner", "mikhail-vrubel", "orest-kiprensky", "william-merritt-chase", "aleksey-savrasov", "hans-memling", "amedeo-modigliani", "ivan-kramskoy", "utagawa-kuniyoshi", "gustave-courbet", "william-turner", "theo-van-rysselberghe", "joseph-wright", "edward-burne-jones", "koloman-moser", "viktor-vasnetsov", "anthony-van-dyck", "raoul-dufy", "frans-hals", "hans-holbein-the-younger", "ilya-mashkov", "henri-fantin-latour", "m.c.-escher", "el-greco", "mikalojus-ciurlionis", "james-mcneill-whistler", "karl-bryullov", "jacob-jordaens", "thomas-gainsborough", "eugene-delacroix", "canaletto"]
    label_to_artist = {i + 1: artist for i, artist in enumerate(artists)}

    def __init__(self, data_dir, img_max_size=(256, 256), transform=None):
        """
        Args:
            data_dir (string): Directory with parquet files (not used for images, only for reading parquet files).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_max_size = img_max_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.img_max_size),
                transforms.ToTensor()
            ])
        else: 
            self.transform = transform

        # Use glob to find all parquet files in the directory
        parquet_files = glob.glob(os.path.join(data_dir, 'batch*.parquet'))

        # Read each parquet file and concatenate them into a single DataFrame
        self.data_frame = pd.concat([pd.read_parquet(file, filters=[[('artist',"in",[22,16,4,2,13,17,3,18,6,15])]]) for file in parquet_files], ignore_index=True )

        # Remove rows with label=0 ('Unknown artist')
        self.data_frame.drop(self.data_frame.index[self.data_frame['artist'] == 0], inplace=True)
        
        # Cast to 0-10 integers label
        self.data_frame['label'] = np.nan
        self.data_frame.loc[self.data_frame['artist']==22, 'label'] = 0
        self.data_frame.loc[self.data_frame['artist']==16, 'label'] = 1
        self.data_frame.loc[self.data_frame['artist']==4, 'label'] = 2
        self.data_frame.loc[self.data_frame['artist']==2, 'label'] = 3
        self.data_frame.loc[self.data_frame['artist']==13, 'label'] = 4
        self.data_frame.loc[self.data_frame['artist']==17,'label'] = 5
        self.data_frame.loc[self.data_frame['artist']==3, 'label'] = 6
        self.data_frame.loc[self.data_frame['artist']==18, 'label'] = 7
        self.data_frame.loc[self.data_frame['artist']==6, 'label'] = 8
        self.data_frame.loc[self.data_frame['artist']==15, 'label'] = 9


        #self.data_frame.to_parquet(f'wikiart_data_batches/data_batches_filtered/batch{2}.parquet')

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image bytes in the dataframe 'image' column (dictonary with the key 'bytes')
        image_bytes = self.data_frame.iloc[idx]['image']['bytes']

        # Convert byte data to Image
        image = Image.open(io.BytesIO(image_bytes))

        # Creating label variable using the column 'label'
        label = self.data_frame.iloc[idx]['label'].astype(int)

        # Apply transformations
        image = self.transform(image)

        return image, label


if __name__ == "__main__":

    dataset = WikiArtDataset(data_dir=os.path.join('wikiart_data_batches', 'data_batches_filtered'))  # Add your image transformations if needed
    train_len = int(len(dataset) * 0.8)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    print(len(dataset))
    for batch in train_dataloader:
        images, label = batch
        #print(images)
        print(label)
