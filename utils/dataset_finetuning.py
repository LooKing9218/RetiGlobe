import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


# class ImageBaseDataset(Dataset):
#     def __init__(
#         self,
#         transform=None,
#     ):
#
#         self.transform = transform
#
#     def __getitem__(self, index):
#         raise NotImplementedError
#
#     def __len__(self):
#         raise NotImplementedError
#
#     def read_from_jpg(self, img_path):
#         raise NotImplementedError
#
#
#     def read_from_dicom(self, img_path):
#         raise NotImplementedError
#
#     def _resize_img(self, img, scale):
#         raise NotImplementedError




class ShantouImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path, mode='val'):
        print("============= {} =============".format(mode))
        import torchvision.transforms as transforms
        from torchvision.transforms import Compose
        if mode == "train":
            self.transform = Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.df = pd.read_csv(csv_file)
        self.data_path = data_path

        self.filenames, self.labels = self.df['Image'], self.df['Label']
        # print("self.filenames ===== {}".format(len(self.filenames)))

        super().__init__()

    def get_imgs(self, img_path, transform=None):
        # tranform images
        try:
            img = Image.open(str(img_path)).convert('RGB')
        except:
            print("img_path ======== {}".format(img_path))
        # img = Image.

        if transform is not None:
            img = transform(img)


        return img

    # def get_label(self):

    def __getitem__(self, index):

        key = self.filenames[index]
        image_path = os.path.join(self.data_path,key)
        # print("image_path ===== {}".format(image_path))

        imgs = self.get_imgs(image_path, self.transform)
        labels = self.labels[index]

        # print("img.shape ===== {}".format(imgs.shape))
        return imgs, labels, key

    def __len__(self):
        return len(self.filenames)
class ShantouImageDatasetIdenOOD(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path, mode='val'):
        print("============= {} =============".format(mode))
        import torchvision.transforms as transforms
        from torchvision.transforms import Compose
        if mode == "train":
            self.transform = Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.df = pd.read_csv(csv_file)
        self.data_path = data_path

        self.filenames, self.labels = self.df['Image'], self.df['Label']
        # print("self.filenames ===== {}".format(len(self.filenames)))

        super().__init__()

    def get_imgs(self, img_path, transform=None):
        # tranform images
        try:
            img = Image.open(str(img_path)).convert('RGB')
        except:
            print("img_path ======== {}".format(img_path))
        # img = Image.

        if transform is not None:
            img = transform(img)


        return img

    # def get_label(self):

    def __getitem__(self, index):

        key = self.filenames[index]
        image_path = os.path.join(self.data_path,key)
        # print("image_path ===== {}".format(image_path))

        imgs = self.get_imgs(image_path, self.transform)
        labels = self.labels[index]

        # print("img.shape ===== {}".format(imgs.shape))
        return imgs, labels, key

    def __len__(self):
        return len(self.filenames)


class ShantouImageDatasetKNN(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path, mode='val'):
        print("============= {} =============".format(mode))
        import torchvision.transforms as transforms
        from torchvision.transforms import Compose
        if mode == "train":
            self.transform = Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.df = pd.read_csv(csv_file)
        self.data_path = data_path

        self.filenames, self.labels = self.df['Image'], self.df['Label']
        # print("self.filenames ===== {}".format(len(self.filenames)))

        super().__init__()

    def get_imgs(self, img_path, transform=None):
        # tranform images
        try:
            img = Image.open(str(img_path)).convert('RGB')
        except:
            print("img_path ======== {}".format(img_path))
        # img = Image.

        if transform is not None:
            img = transform(img)


        return img

    # def get_label(self):

    def __getitem__(self, index):

        key = self.filenames[index]
        image_path = os.path.join(self.data_path,key)
        # print("image_path ===== {}".format(image_path))

        imgs = self.get_imgs(image_path, self.transform)
        labels = self.labels[index]

        # print("img.shape ===== {}".format(imgs.shape))
        return imgs, labels

    def __len__(self):
        return len(self.filenames)

class ShantouImageDatasetKNNImagePath(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path, mode='val'):
        print("============= {} =============".format(mode))
        import torchvision.transforms as transforms
        from torchvision.transforms import Compose
        self.transform = Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.df = pd.read_csv(csv_file)
        self.data_path = data_path

        self.filenames, self.labels = self.df['Image'], self.df['Label']
        # print("self.filenames ===== {}".format(len(self.filenames)))

        super().__init__()

    def get_imgs(self, img_path, transform=None):
        # tranform images
        try:
            img = Image.open(str(img_path)).convert('RGB')
        except:
            print("img_path ======== {}".format(img_path))
        # img = Image.

        if transform is not None:
            img = transform(img)


        return img

    # def get_label(self):

    def __getitem__(self, index):

        key = self.filenames[index]
        image_path = os.path.join(self.data_path,key)
        # print("image_path ===== {}".format(image_path))

        imgs = self.get_imgs(image_path, self.transform)
        labels = self.labels[index]

        # print("img.shape ===== {}".format(imgs.shape))
        return imgs, labels, key

    def __len__(self):
        return len(self.filenames)


class ShantouImageDatasetKNN_DA(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path, transform):

        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.data_path = data_path

        self.filenames, self.labels = self.df['Image'], self.df['Label']
        # print("self.filenames ===== {}".format(len(self.filenames)))

        super().__init__()

    def get_imgs(self, img_path, transform=None):
        # tranform images
        try:
            img = Image.open(str(img_path)).convert('RGB')
        except:
            print("img_path ======== {}".format(img_path))
        # img = Image.

        if transform is not None:
            img = transform(img)
            # print("img.shape ============ {}".format(img.shape))

        return img

    # def get_label(self):

    def __getitem__(self, index):

        key = self.filenames[index]
        image_path = os.path.join(self.data_path,key)
        # print("image_path ===== {}".format(image_path))

        imgs = self.get_imgs(image_path, self.transform)
        labels = self.labels[index]

        # print("img.shape ===== {}".format(imgs.shape))
        return imgs, labels

    def __len__(self):
        return len(self.filenames)



class ShantouImageDatasetKNNWithIden(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path, mode='val'):
        print("============= {} =============".format(mode))
        import torchvision.transforms as transforms
        from torchvision.transforms import Compose
        if mode == "train":
            self.transform = Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.transform = Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.df = pd.read_csv(csv_file)
        self.data_path = data_path

        self.filenames, self.labels, self.unSignals = self.df['Image'], self.df['Label'], self.df['USignals']
        self.all_datas = []
        self.all_labels = []
        for indx_Sample in range(len(self.filenames)):
            if self.unSignals[indx_Sample]:
                self.all_datas.append(self.filenames[indx_Sample])
                self.all_labels.append(self.labels[indx_Sample])
        print("self.all_datas ===== {}".format(len(self.all_datas)))

        super().__init__()

    def get_imgs(self, img_path, transform=None):
        # tranform images
        try:
            img = Image.open(str(img_path)).convert('RGB')
        except:
            print("img_path ======== {}".format(img_path))
        # img = Image.

        if transform is not None:
            img = transform(img)
        return img

    # def get_label(self):

    def __getitem__(self, index):

        key = self.all_datas[index]
        image_path = os.path.join(self.data_path,key)
        # print("image_path ===== {}".format(image_path))

        imgs = self.get_imgs(image_path, self.transform)
        labels = self.all_labels[index]

        # print("img.shape ===== {}".format(imgs.shape))
        return imgs, labels

    def __len__(self):
        return len(self.all_datas)
