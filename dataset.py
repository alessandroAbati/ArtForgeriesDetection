import os
import numpy as np
import torch
import pandas as pd
from torch.utils import data as data
from PIL import Image
from torchvision import transforms as transforms

from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from sklearn.model_selection import train_test_split


class WikiArtDataset(data.Dataset):

    def __init__(self, mode="train", root_dir=None, source_dir=None, load_data=False,
                 img_max_size=(256, 256)):
        super(WikiArtDataset, self).__init__()
        self.img_max_size = img_max_size
        # self.dir = os.path.join(root_dir, source_dir)
        self.dir = "/"

        self.names = ["Unknown Artist", "boris-kustodiev", "camille-pissarro", "childe-hassam", "claude-monet",
                      "edgar-degas",
                      "eugene-boudin", "gustave-dore", "ilya-repin", "ivan-aivazovsky", "ivan-shishkin",
                      "john-singer-sargent", "marc-chagall", "martiros-saryan", "nicholas-roerich",
                      "pablo-picasso", "paul-cezanne", "pierre-auguste-renoir", "pyotr-konchalovsky",
                      "raphael-kirchner", "rembrandt", "salvador-dali", "vincent-van-gogh",
                      "hieronymus-bosch", "leonardo-da-vinci", "albrecht-durer", "edouard-cortes", "sam-francis",
                      "juan-gris", "lucas-cranach-the-elder", "paul-gauguin",
                      "konstantin-makovsky", "egon-schiele", "thomas-eakins", "gustave-moreau", "francisco-goya",
                      "edvard-munch", "henri-matisse", "fra-angelico", "maxime-maufra",
                      "jan-matejko", "mstislav-dobuzhinsky", "alfred-sisley", "mary-cassatt", "gustave-loiseau",
                      "fernando-botero", "zinaida-serebriakova", "georges-seurat",
                      "isaac-levitan", "joaqu\u00e3\u00adn-sorolla", "jacek-malczewski", "berthe-morisot",
                      "andy-warhol", "arkhip-kuindzhi", "niko-pirosmani", "james-tissot",
                      "vasily-polenov", "valentin-serov", "pietro-perugino", "pierre-bonnard", "ferdinand-hodler",
                      "bartolome-esteban-murillo", "giovanni-boldini", "henri-martin",
                      "gustav-klimt", "vasily-perov", "odilon-redon", "tintoretto", "gene-davis", "raphael",
                      "john-henry-twachtman", "henri-de-toulouse-lautrec", "antoine-blanchard",
                      "david-burliuk", "camille-corot", "konstantin-korovin", "ivan-bilibin", "titian",
                      "maurice-prendergast", "edouard-manet", "peter-paul-rubens", "aubrey-beardsley", "paolo-veronese",
                      "joshua-reynolds", "kuzma-petrov-vodkin", "gustave-caillebotte", "lucian-freud", "michelangelo",
                      "dante-gabriel-rossetti", "felix-vallotton", "nikolay-bogdanov-belsky", "georges-braque",
                      "vasily-surikov", "fernand-leger", "konstantin-somov", "katsushika-hokusai",
                      "sir-lawrence-alma-tadema", "vasily-vereshchagin", "ernst-ludwig-kirchner", "mikhail-vrubel",
                      "orest-kiprensky", "william-merritt-chase", "aleksey-savrasov", "hans-memling",
                      "amedeo-modigliani", "ivan-kramskoy", "utagawa-kuniyoshi", "gustave-courbet", "william-turner",
                      "theo-van-rysselberghe", "joseph-wright", "edward-burne-jones", "koloman-moser",
                      "viktor-vasnetsov", "anthony-van-dyck", "raoul-dufy", "frans-hals", "hans-holbein-the-younger",
                      "ilya-mashkov", "henri-fantin-latour", "m.c.-escher", "el-greco", "mikalojus-ciurlionis",
                      "james-mcneill-whistler", "karl-bryullov", "jacob-jordaens", "thomas-gainsborough",
                      "eugene-delacroix", "canaletto"]

        dataset = load_dataset("huggan/wikiart")
        artist_names = dataset["train"]["artist"]
        if load_data:
            df = pd.DataFrame({"artist": artist_names})
            df.to_csv("wikiart_dataset.csv", index=False)

        self.filtered_indices = np.where(np.array(artist_names) != 0)[0]
        self.data = dataset["train"]
        print(self.data.shape)

        self.transform = transforms.Compose([
            transforms.Resize(self.img_max_size),
            transforms.ToTensor()
        ])

        self.index2artist = {idx: artist for idx, artist in enumerate(self.names)}
        print(self.index2artist)

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.dir, self.data_csv.iloc[idx, 0])  # Adjust the path
        filtered_idx = int(self.filtered_indices[idx])
        batch = self.data[filtered_idx]
        print(batch)
        image = batch['image']
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        artist = batch['artist']
        artist_label = artist # Get the artist label
        artist_name = self.index2artist[artist_label]

        if self.transform:
            image = self.transform(image)

        return image, artist_label, artist_name


if __name__ == "__main__":

    dataset = WikiArtDataset(mode="train", load_data=True)  # Add your image transformations if needed
    train_len = int(len(dataset) * 0.8)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    instances = 0
    print(len(dataset))
    for batch in train_dataloader:
        images, label, name = batch
        print(label)
        print(name)
