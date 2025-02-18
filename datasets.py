import os
import pandas as pd
import skimage.draw
import torch.utils.data
import numpy as np
import torch.utils.data
import typing
from pycocotools.coco import COCO
import pickle
import torchvision.transforms.v2 as transforms
import torchvision.io
from PIL import Image

class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, split: float = 1, mode: str = 'train', train_samples: list = None,
                 image_size: int = 224, evaluate: bool = False):
        """
        CUB dataset
        Parameters
        ----------
        data_path: str
            Directory containing the 'attributes', 'images', and 'parts' folders
        split: float
            Percentage of training samples to use
        mode: str
            Whether to use the training, validation, or test split
        train_samples: list
            List of samples to exclude fpr the validation dataset
        image_size: int
            Size of the images
        evaluate: bool
            Set to true to evaluate parts (disables transforms such as normalization, crop, etc.)
        """
        self.data_path = data_path
        self.mode = mode
        self.image_size = image_size
        train_test = pd.read_csv(os.path.join(data_path, 'train_test_split.txt'), delim_whitespace=True, names=['id', 'train'])
        image_names = pd.read_csv(os.path.join(data_path, 'images.txt'), delim_whitespace = True, names=['id', 'filename'])
        labels = pd.read_csv(os.path.join(data_path, 'image_class_labels.txt'), delim_whitespace=True, names=['id', 'label'])
        image_parts = pd.read_csv(os.path.join(data_path, 'parts/part_locs.txt'), delim_whitespace=True, names=['id', 'part_id', 'x', 'y', 'visible'])
        dataset = train_test.merge(image_names, on='id')
        dataset = dataset.merge(labels, on='id')

        if mode == 'train':
            dataset = dataset.loc[dataset['train'] == 1]
            samples = np.arange(len(dataset))
            np.random.shuffle(samples)
            self.trainsamples = samples[:int(len(samples)*split)]
            dataset = dataset.iloc[self.trainsamples]
            self.transform = self.get_transforms(image_size, evaluate)[0]
        elif mode == 'test':
            dataset = dataset.loc[dataset['train'] == 0]
            dataset.to_csv('testset.tsv', sep='\t', columns=['filename'], index=False)
            self.transform = self.get_transforms(image_size, evaluate)[1]
        elif mode == 'val':
            dataset = dataset.loc[dataset['train'] == 1]
            if train_samples is None:
                raise RuntimeError('Please provide the list of training samples'
                                   'to the validation dataset')
            dataset = dataset.drop(dataset.index[train_samples])
            self.transform = self.get_transforms(image_size, evaluate)[1]

        # training images are labelled 1, test images labelled 0. Add these
        # images to the list of image IDs
        self.ids = np.array(dataset['id'])
        self.names = np.array(dataset['filename'])
        # Subtract 1 because classes run from 1-200 instead of 0-199
        self.labels = np.array(dataset['label']) - 1
        parts = {}
        for i in self.ids:
            parts[i] = image_parts[image_parts['id'] == i]
        self.parts = parts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.data_path + "/images/" + self.names[idx]
        im = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB)
        label = self.labels[idx]

        im = self.transform(im)
        return im, label, idx

    @staticmethod
    def get_transforms(image_size: typing.Union[int, typing.Sequence[int]], evaluate: bool = False):
        if not evaluate:
            train_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1),
                transforms.RandomAffine(degrees=45, translate=(0.2, 0.2), scale=(0.5, 0.9)),
                transforms.RandomCrop(image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.CenterCrop(size=image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.CenterCrop(size=image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.CenterCrop(size=image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        return train_transforms, test_transforms


import csv
class S_CAR(torch.utils.data.Dataset):
    def __init__(self, data_path: str, mode: str = 'train', image_size: int = 224, evaluate: bool = False):
        super(S_CAR, self).__init__()
        self.root = data_path
        self.size = image_size
        # self.num = args.num_classes
        self.train = mode == 'train'

        if mode == 'train':
            self.transform = CUBDataset.get_transforms(image_size, evaluate)[0]
        elif mode == 'test':
            self.transform = CUBDataset.get_transforms(image_size, evaluate)[1]
        elif mode == 'val':
            # if train_samples is None:
            #     raise RuntimeError('Please provide the list of training samples'
            #                        'to the validation dataset')
            self.transform = CUBDataset.get_transforms(image_size, evaluate)[1]

        # self.transform_ = transform
        self.classes_file = os.path.join(self.root, 'class_name.csv')  # <class_id> <class_name>
        self.train_file = os.path.join(self.root, 'car_annots_true_train.csv')  # <image_id> <bbox> <class_id>
        self.test_file = os.path.join(self.root, 'car_annots_true_test.csv')
        # self.images_file = os.path.join(self.root, 'images.txt')  # <image_id> <image_name>
        # self.train_test_split_file = os.path.join(self.root, 'train_test_split.txt')  # <image_id> <is_training_image>
        # self.bounding_boxes_file = os.path.join(self.root, 'bounding_boxes.txt')  # <image_id> <x> <y> <width> <height>

        self._train_ids = []
        self._test_ids = []
        self._class_label = {}
        # self._train_path_label = []
        # self._test_path_label = []

        self._train_test_read()

    def _train_test_read(self):


        with open(self.classes_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            next(rows, None)
            for row in rows:
                self._class_label[int(row[0])]=row[1]

        with open(self.train_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            next(rows, None)
            for row in rows:
                self._train_ids.append((row[0],row[5]))

        with open(self.test_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            next(rows, None)
            for row in rows:
                self._test_ids.append((row[0],row[5]))



    def __getitem__(self, index):
        imgFolder = ''
        if self.train:
            image_name, label = self._train_ids[index]
            imgFolder= 'cars_train/cars_train'
        else:
            image_name, label = self._test_ids[index]
            imgFolder = 'cars_test/cars_test'
        image_path = os.path.join(self.root, imgFolder, image_name)
        # img = Image.open(image_path)
        img = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.RGB)
        label = int(label)
        labelT = torch.from_numpy(np.array(label)) - 1
        if self.transform is not None:
            img = self.transform(img)
        return img, labelT, index

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)



class ShapeNet_CAR(torch.utils.data.Dataset):
    def __init__(self, data_path: str, mode: str = 'train', image_size: int = 224, evaluate: bool = False):
        super(ShapeNet_CAR, self).__init__()
        self.root = data_path
        self.size = image_size
        # self.num = args.num_classes
        self.train = mode == 'train'

        if mode == 'train':
            self.transform = CUBDataset.get_transforms(image_size, evaluate)[0]
        elif mode == 'test':
            self.transform = CUBDataset.get_transforms(image_size, evaluate)[1]
        elif mode == 'val':
            # if train_samples is None:
            #     raise RuntimeError('Please provide the list of training samples'
            #                        'to the validation dataset')
            self.transform = CUBDataset.get_transforms(image_size, evaluate)[1]

        # self.transform_ = transform

        self.train_file = os.path.join(self.root, 'train.csv')  # <image_id> <bbox> <class_id>
        self.test_file = os.path.join(self.root, 'test.csv')
        # self.images_file = os.path.join(self.root, 'images.txt')  # <image_id> <image_name>
        # self.train_test_split_file = os.path.join(self.root, 'train_test_split.txt')  # <image_id> <is_training_image>
        # self.bounding_boxes_file = os.path.join(self.root, 'bounding_boxes.txt')  # <image_id> <x> <y> <width> <height>

        self._train_ids = []
        self._test_ids = []
        self.class_id = {}

        # self._train_path_label = []
        # self._test_path_label = []



        self._train_test_read()

    def _train_test_read(self):

        if self.train:
            with open(self.train_file, newline='') as csvfile:
                rows = csv.reader(csvfile, delimiter=',')
                next(rows, None)
                for id, row in enumerate(rows):
                    self.class_id[row[0]] = id
                    for i in range(36):
                        self._train_ids.append((f'{row[1]}/{row[0]}/{i:02d}.png', id))
        else:
            with open(self.test_file, newline='') as csvfile:
                rows = csv.reader(csvfile, delimiter=',')
                next(rows, None)
                for id, row in enumerate(rows):
                    self.class_id[row[0]] = id
                    for i in range(36):
                        self._test_ids.append((f'{row[1]}/{row[0]}/{i:02d}.png', id))



    def __getitem__(self, index):
        imgFolder = ''
        if self.train:
            image_name, label = self._train_ids[index]
        else:
            image_name, label = self._test_ids[index]

        image_path = os.path.join(self.root, image_name)
        # img = Image.open(image_path)
        img = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.RGB)
        # label = self.class_id[oid]
        labelT = torch.from_numpy(np.array(label))
        if self.transform is not None:
            img = self.transform(img)
        return img, labelT

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)

from glob import glob
class SYSUDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode: str = 'train', image_size: typing.Sequence[int] = [512, 256], evaluate: bool = False):
        print(root)
        assert os.path.isdir(root)
        # assert mode in ['train', 'gallery', 'query']

        if mode == 'train':
            train_ids = open(os.path.join(root, 'exp', 'train_id.txt')).readline()
            val_ids = open(os.path.join(root, 'exp', 'val_id.txt')).readline()

            train_ids = train_ids.strip('\n').split(',')
            val_ids = val_ids.strip('\n').split(',')
            selected_ids = train_ids + val_ids
        else:
            test_ids = open(os.path.join(root, 'exp', 'test_id.txt')).readline()
            selected_ids = test_ids.strip('\n').split(',')

        if mode == 'train':
            self.transform = CUBDataset.get_transforms(image_size, mode == 'train')[0]
        else:
            self.transform = CUBDataset.get_transforms(image_size, mode != 'train')[1]

        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)

        img_paths = glob(os.path.join(root, 'cam?/**/*.jpg'), recursive=True)
        img_paths = [path for path in img_paths if int(path.split('/')[-2]) in selected_ids]

        if mode == 'gallery':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (1, 2, 4, 5)]
        elif mode == 'query':
            img_paths = [path for path in img_paths if int(path.split('/')[-3][-1]) in (3, 6)]

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split('/')[-3][-1]) for path in img_paths]
        self.num_ids = num_ids

        if mode == 'train':
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split('/')[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split('/')[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        # img = Image.open(path)
        img = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label #, cam, path, item

class PartImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, mode: str = 'train', get_masks: bool = False, image_size: int = 224,
                 evaluate: bool = False):
        """
        PartImageNet dataset
        Parameters
        ----------
        data_path: str
            Directory containing the 'train_train', 'train_test', and 'test' folders
        mode: str
            Whether to use the training or validation split
        get_masks: bool
            Whether to return the ground truth masks
        image_size: int
            Size of the images
        evaluate: bool
            Set to true to evaluate parts (disables transforms such as normalization, crop, etc.)
        """
        self.mode = mode
        self.data_path = data_path
        self.get_masks = get_masks
        dataset = pd.read_csv(data_path + "/" + "newdset.txt", sep='\t', names=["index", "test", "label", "class", "filename"])
        if mode == "train":
            self.dataset = dataset.loc[dataset['test'] == 0]
            self.transform = self.get_transforms(image_size, evaluate)[0]
        elif mode == "val":
            self.dataset = dataset.loc[dataset['test'] == 1]
            self.transform = self.get_transforms(image_size, evaluate)[1]
        elif mode == "test":
            self.dataset = dataset.loc[dataset['test'] == 1]
            self.transform = self.get_transforms(image_size, evaluate)[1]
        annFile = os.path.join(data_path, f"train.json")

        coco = COCO(annFile)
        self.coco = coco

    def getmasks(self, i):
        idx = self.dataset.iloc[i]['index']
        idx = int(idx)
        coco = self.coco
        img = coco.loadImgs(idx)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        cat_ids = [ann['category_id'] for ann in anns]
        polygons = []
        for ann in anns:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                polygons.append(poly)
        for cat, p in zip(cat_ids, polygons):
            mask = skimage.draw.polygon2mask((img['width'], img['height']), p)
            try:
                mask_tensor[cat] += torch.FloatTensor(mask)
            except NameError:
                mask_tensor = torch.zeros(size=(40, mask.shape[-2], mask.shape[-1]))
                mask_tensor[cat] += torch.FloatTensor(mask)
        try:
            mask_tensor = torch.where(mask_tensor > 0.1, 1, 0).permute(0, 2, 1)
            return mask_tensor
        except UnboundLocalError:
            # if an image has no ground truth parts
            return None

    def __len__(self):
        return len(self.dataset['index'])

    def __getitem__(self, idx):
        curr_row = self.dataset.iloc[idx]
        folder = curr_row['class']
        imgname = curr_row['filename']
        if self.mode == 'train':
            path = f"{self.data_path}/train_train/{folder}/{imgname}"
        elif self.mode == 'test':
            path = f"{self.data_path}/train_test/{folder}/{imgname}"
        im = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB)
        label = curr_row['label']
        im = self.transform(im)

        if not self.get_masks:
            return im, label

        mask = self.getmasks(idx)
        if mask == None:
            mask = torch.zeros(size=(40, im.shape[-2], im.shape[-1]))
        mask = transforms.Resize(size=(im.shape[-2], im.shape[-1]),
                interpolation=transforms.InterpolationMode.NEAREST)(mask)
        return im, label, mask

    @staticmethod
    def get_transforms(image_size: int, evaluate: bool = False):
        if not evaluate:
            train_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1),
                transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.RandomCrop(image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.CenterCrop(size=image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.ToDtype(torch.float32, scale=True)
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        return train_transforms, test_transforms


class CelebA(torch.utils.data.Dataset):
    def __init__(self, root, split='train', percentage=None, image_size: int = 256, evaluate: bool = False):
        """
        CelebA dataset
        Parameters
        ----------
        root: str
            Directory containing the 'unaligned' folder
        split: str
            Whether to use the training or test split
        percentage
        image_size: int
            Size of the images
        evaluate: bool
            Set to true to evaluate parts (disables transforms such as normalization, crop, etc.)
        """
        self.root = root
        self.split = split
        self.resize = image_size
        self.evaluate = evaluate

        # load the dictionary for data
        percentage_name = '_0' if percentage is None else '_'+str(int(percentage*100))
        save_name = os.path.join(root, split+'_unaligned'+percentage_name+'.pickle')
        self.shuffle = np.arange(202599)
        np.random.shuffle(self.shuffle)
        if os.path.exists(save_name) is False:
            print('Preparing the data...')
            self.generate_dict(save_name)
            print('Data dictionary created and saved.')
        with open(save_name, 'rb') as handle:
            save_dict = pickle.load(handle)


        self.images = save_dict['images']           # image filenames
        self.landmarks = save_dict['landmarks']     # 5 face landmarks
        self.targets = save_dict['targets']         # binary labels
        self.bboxes = save_dict['bboxes']           # x y w h
        self.sizes = save_dict['sizes']             # height width
        self.identities = save_dict['identities']
        if split == 'train':
            self.transform = self.get_transforms(image_size, evaluate)[0]
        else:
            self.transform = self.get_transforms(image_size, evaluate)[1]

        # select a subset of the current data split according the face area
        if percentage is not None:
            new_images = []
            new_landmarks = []
            new_targets = []
            new_bboxes = []
            new_sizes = []
            new_identities = []
            for i in range(len(self.images)):
                if float(self.bboxes[i][-1] * self.bboxes[i][-2]) >= float(self.sizes[i][-1] * self.sizes[i][-2]) * percentage:
                    new_images.append(self.images[i])
                    new_landmarks.append(self.landmarks[i])
                    new_targets.append(self.targets[i])
                    new_bboxes.append(self.bboxes[i])
                    new_sizes.append(self.sizes[i])
                    new_identities.append(self.identities[i])
            self.images = new_images
            self.landmarks = new_landmarks
            self.targets = new_targets
            self.bboxes = new_bboxes
            self.sizes = new_sizes
            self.identities = new_identities
        print('Number of classes in the ' + self.split + ' split: ' + str(max(self.identities)))
        print('Number of samples in the ' + self.split + ' split: '+ str(len(self.images)))


    # generate a dictionary for a certain data split
    def generate_dict(self, save_name):

        print('Start generating data dictionary as '+save_name)

        full_img_list = []
        ann_file = 'list_attr_celeba.txt'
        bbox_file = 'list_bbox_celeba.txt'
        size_file = 'list_imsize_celeba.txt'
        identity_file = 'identity_CelebA.txt'
        landmark_file = 'list_landmarks_unalign_celeba.txt'

        # load all the images according to the current split
        if self.split == 'train':
            imgfile = 'celebA_training.txt'
        elif self.split == 'val':
            imgfile = 'celebA_validating.txt'
        elif self.split == 'test':
            imgfile = 'celebA_testing.txt'
        elif self.split == 'fit':
            imgfile = 'MAFL_training.txt'
        elif self.split == 'eval':
            imgfile = 'MAFL_testing.txt'
        elif self.split == 'train_full':
            imgfile = 'celebA_training_full.txt'
        for line in open(os.path.join(self.root, imgfile), 'r'):
            full_img_list.append(line.split()[0])

        # prepare the indexes and convert annotation files to lists
        full_img_list_idx = [(int(s.rstrip(".jpg"))-1) for s in full_img_list]
        ann_full_list = [line.split() for line in open(os.path.join(self.root, ann_file), 'r')]
        bbox_full_list = [line.split() for line in open(os.path.join(self.root, bbox_file), 'r')]
        size_full_list = [line.split() for line in open(os.path.join(self.root, size_file), 'r')]
        landmark_full_list = [line.split() for line in open(os.path.join(self.root, landmark_file), 'r')]
        identity_full_list = [line.split() for line in open(os.path.join(self.root, identity_file), 'r')]

        # assertion
        assert len(ann_full_list[0]) == 41
        assert len(bbox_full_list[0]) == 5
        assert len(size_full_list[0]) == 3
        assert len(landmark_full_list[0]) == 11

        # select samples and annotations for the current data split
        # init the lists
        filename_list = []
        target_list = []
        landmark_list = []
        bbox_list = []
        size_list = []
        identity_list = []

        # select samples and annotations
        for i in full_img_list_idx:
            idx = self.shuffle[i]

            # assertion
            assert (idx+1) == int(ann_full_list[idx][0].rstrip(".jpg"))
            assert (idx+1) == int(bbox_full_list[idx][0].rstrip(".jpg"))
            assert (idx+1) == int(size_full_list[idx][0].rstrip(".jpg"))
            assert (idx+1) == int(landmark_full_list[idx][0].rstrip(".jpg"))

            # append the filenames and annotations
            filename_list.append(ann_full_list[idx][0])
            target_list.append([int(i) for i in ann_full_list[idx][1:]])
            bbox_list.append([int(i) for i in bbox_full_list[idx][1:]])
            size_list.append([int(i) for i in size_full_list[idx][1:]])
            landmark_list_xy = []
            for j in range(5):
                landmark_list_xy.append([int(landmark_full_list[idx][1+2*j]), int(landmark_full_list[idx][2+2*j])])
            landmark_list.append(landmark_list_xy)
            identity_list.append(int(identity_full_list[idx][1]))

        # expand the filename to the full path
        full_path_list = [os.path.join(self.root, 'unaligned', filename) for filename in filename_list]

        # create the dictionary and save it on the disk
        save_dict = {}
        save_dict['images'] = full_path_list
        save_dict['landmarks'] = landmark_list
        save_dict['targets'] = target_list
        save_dict['bboxes'] = bbox_list
        save_dict['sizes'] = size_list
        save_dict['identities'] = identity_list
        with open(save_name, 'wb') as handle:
            pickle.dump(save_dict, handle)

    def __getitem__(self, index):
        # load images and targets
        path = self.images[index]
        sample = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB)
        identity = self.identities[index] - 1
        image = np.array(sample)
        if image.shape[-2] > image.shape[-1]:
            factor = self.resize / image.shape[-1]
        else:
            factor = self.resize / image.shape[-2]
        # transform the image and target
        if self.transform is not None:
            sample = self.transform(sample)

        if not self.evaluate:
            return sample, identity

        # processing the landmarks
        landmark_locs = self.landmarks[index]
        landmark_locs = torch.LongTensor(landmark_locs).float()
        landmark_locs[:, 0] = landmark_locs[:, 0] * factor
        landmark_locs[:, 1] = landmark_locs[:, 1] * factor
        return sample, identity, landmark_locs

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_transforms(image_size: int, evaluate: bool = False):
        if not evaluate:
            train_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.ColorJitter(0.1),
                transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.RandomCrop(image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.CenterCrop(size=image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.ToDtype(torch.float32, scale=True)
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        return train_transforms, test_transforms

class CUB200(torch.utils.data.Dataset):
    """
    CUB dataset. Used for evaluating the model's part localization performance. Adapted from:
    https://github.com/zxhuang1698/interpretability-by-parts/.
    Parameters
    ----------
    root: str
        Root directory of the dataset.
    train: bool
        train/test data split.
    image_size: int
        Length of the shortest of edge of the resized image. Used for transforming landmarks and bounding boxes.
    evaluate: bool
        Whether to run the part evaluation. Set to true when evaluating part localization performance.
    """
    def __init__(self, root, train=True, image_size=448, evaluate: bool = True):
        self._root = root
        self._train = train
        self.newsize = image_size
        # 15 key points provided by CUB
        self.num_kps = 15

        if not os.path.isdir(root):
            os.mkdir(root)

        # Load all data into memory for best IO efficiency. This might take a while
        if self._train:
            self._train_data, self._train_labels, self._train_parts, self._train_boxes = self._get_file_list(train=True)
            assert (len(self._train_data) == 5994
                    and len(self._train_labels) == 5994)
            self._transform = self.get_transforms(image_size, evaluate=evaluate)[0]
        else:
            self._test_data, self._test_labels, self._test_parts, self._test_boxes = self._get_file_list(train=False)
            assert (len(self._test_data) == 5794
                    and len(self._test_labels) == 5794)
            self._transform = self.get_transforms(image_size, evaluate=evaluate)[1]

    def __getitem__(self, index):
        """
        Retrieve data samples.
        Args
        ----------
        index: int
            Index of the sample.
        Returns
        ----------
        image: torch.FloatTensor, [3, H, W]
            Image of the given index.
        target: int
            Label of the given index.
        parts: torch.FloatTensor, [15, 4]
            Landmark annotations.
        boxes: torch.FloatTensor, [5, ]
            Bounding box annotations.
        """
        # load the variables according to the current index and split
        if self._train:
            image_path = self._train_data[index]
            target = self._train_labels[index]
            parts = self._train_parts[index]
            boxes = self._train_boxes[index]

        else:
            image_path = self._test_data[index]
            target = self._test_labels[index]
            parts = self._test_parts[index]
            boxes = self._test_boxes[index]

        # load the image
        image = torchvision.io.read_image(image_path, torchvision.io.ImageReadMode.RGB)

        # numpy arrays to pytorch tensors
        parts = torch.from_numpy(parts).float()
        boxes = torch.from_numpy(boxes).float()

        # calculate the resize factor
        # if original image height is larger than width, the real resize factor is based on width
        if image.shape[1] >= image.shape[2]:
            factor = self.newsize / image.shape[2]
        else:
            factor = self.newsize / image.shape[1]

        # transform 15 landmarks according to the new shape
        # each landmark has a 4-element annotation: <landmark_id, column, row, existence>
        for j in range(self.num_kps):

            # step in only when the current landmark exists
            if abs(parts[j][-1]) > 1e-5:
                # calculate the new location according to the new shape
                parts[j][-3] = parts[j][-3] * factor
                parts[j][-2] = parts[j][-2] * factor

        # rescale the annotation of bounding boxes
        # the annotation format of the bounding boxes are <image_id, col of top-left corner, row of top-left corner, width, height>
        boxes[1:] *= factor

        # apply transformation
        if self._transform is not None:
            image = self._transform(image)
        return image, target, parts, boxes, image_path

    def __len__(self):
        """Return the length of the dataset."""
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _get_file_list(self, train=True):
        """Prepare the data for train/test split and save onto disk."""

        # load the list into numpy arrays
        image_path = self._root + '/CUB_200_2011/images/'
        id2name = np.genfromtxt(self._root + '/CUB_200_2011/images.txt', dtype=str)
        id2train = np.genfromtxt(self._root + '/CUB_200_2011/train_test_split.txt', dtype=int)
        id2part = np.genfromtxt(self._root + '/CUB_200_2011/parts/part_locs.txt', dtype=float)
        id2box = np.genfromtxt(self._root + '/CUB_200_2011/bounding_boxes.txt', dtype=float)

        # creat empty lists
        train_data = []
        train_labels = []
        train_parts = []
        train_boxes = []
        test_data = []
        test_labels = []
        test_parts = []
        test_boxes = []

        # iterating all samples in the whole dataset
        for id_ in range(id2name.shape[0]):
            # load each variable
            image = os.path.join(image_path, id2name[id_, 1])
            # Label starts with 0
            label = int(id2name[id_, 1][:3]) - 1
            parts = id2part[id_*self.num_kps : id_*self.num_kps+self.num_kps][:, 1:]
            boxes = id2box[id_]

            # training split
            if id2train[id_, 1] == 1:
                train_data.append(image)
                train_labels.append(label)
                train_parts.append(parts)
                train_boxes.append(boxes)
            # testing split
            else:
                test_data.append(image)
                test_labels.append(label)
                test_parts.append(parts)
                test_boxes.append(boxes)

        # return accoring to different splits
        if train == True:
            return train_data, train_labels, train_parts, train_boxes
        else:
            return test_data, test_labels, test_parts, test_boxes

    @staticmethod
    def get_transforms(image_size: int, evaluate: bool = False):
        if not evaluate:
            train_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1),
                transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.RandomCrop(image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.CenterCrop(size=image_size),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        else:
            train_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.ToDtype(torch.float32, scale=True)
            ])
            test_transforms = transforms.Compose([
                transforms.Resize(size=image_size, antialias=True),
                transforms.ToDtype(torch.float32, scale=True)
            ])
        return train_transforms, test_transforms




if __name__=='__main__':
    pass


import copy
from torch.utils.data import Sampler
from collections import defaultdict

class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic_R = defaultdict(list)
        self.index_dic_I = defaultdict(list)
        for i, identity in enumerate(data_source.ids):
            if data_source.cam_ids[i] in [3, 6]:
                self.index_dic_I[identity].append(i)
            else:
                self.index_dic_R[identity].append(i)
        self.pids = list(self.index_dic_I.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic_I[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = copy.deepcopy(self.index_dic_I[pid])
            idxs_R = copy.deepcopy(self.index_dic_R[pid])
            if len(idxs_I) < self.num_instances // 2 and len(idxs_R) < self.num_instances // 2:
                idxs_I = np.random.choice(idxs_I, size=self.num_instances // 2, replace=True)
                idxs_R = np.random.choice(idxs_R, size=self.num_instances // 2, replace=True)
            if len(idxs_I) > len(idxs_R):
                idxs_I = np.random.choice(idxs_I, size=len(idxs_R), replace=False)
            if len(idxs_R) > len(idxs_I):
                idxs_R = np.random.choice(idxs_R, size=len(idxs_I), replace=False)
            np.random.shuffle(idxs_I)
            np.random.shuffle(idxs_R)
            batch_idxs = []
            for idx_I, idx_R in zip(idxs_I, idxs_R):
                batch_idxs.append(idx_I)
                batch_idxs.append(idx_R)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        # for i, row in enumerate(data_source._train_ids): # for S_CAR
        #     id = row[1]
        #     self.index_dic[id].append(i)
        for i, id in enumerate(data_source.ids): #For Market
            self.index_dic[id].append(i)


        self.pids = list(self.index_dic.keys())
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
    def __iter__(self):
        print("Sampler iter ...!")
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs_I = copy.deepcopy(self.index_dic[pid])
            if len(idxs_I) < self.num_instances :
                idxs_I = np.random.choice(idxs_I, size=self.num_instances, replace=True)
            np.random.shuffle(idxs_I)
            batch_idxs = []
            for idx_I in idxs_I:
                batch_idxs.append(idx_I)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

import re
import os.path as osp
class MarketDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode='train', transform=None):
        root += '1501'
        assert os.path.isdir(root)
        assert mode in ['train', 'gallery', 'query']

        self.transform = transform

        if mode == 'train':
            img_paths = glob(os.path.join(root, 'bounding_box_train/*.jpg'), recursive=True)
        elif mode == 'gallery':
            img_paths = glob(os.path.join(root, 'bounding_box_test/*.jpg'), recursive=True)
        elif mode == 'query':
            img_paths = glob(os.path.join(root, 'query/*.jpg'), recursive=True)

        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        relabel = mode == 'train'
        self.img_paths = []
        self.cam_ids = []
        self.ids = []
        for fpath in img_paths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continuem
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            self.img_paths.append(fpath)
            self.ids.append(all_pids[pid])
            self.cam_ids.append(cam - 1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        # img = Image.open(path)
        img = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        # return img, label, cam, path, item
        return img, label, item