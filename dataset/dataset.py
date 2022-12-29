from itertools import product

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop
import random

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img_path):
        img = Image.open(img_path).convert('RGB')
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            mask_rate = 0.2
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world


        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        self.open_pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
            len_attr, len_obj = 0, 0
            while len_attr!=len(self.attrs) and len_obj!=len(self.objs):
                self.mask(mask_rate)
                len_attr, len_obj = self.data_statistic()
        elif self.phase == 'val':
            self.data = self.val_data
            self.mask(0)
            self.mask_data = self.val_data
        else:
            self.data = self.test_data
            self.mask(0)
            self.mask_data = self.test_data

        print('# train pairs: %d | # mask pairs: %d | # val pairs: %d | # test pairs: %d | # closed pairs: %d | # open pairs: %d' % (len(
            self.train_pairs), len(self.mask_train_pairs), len(self.val_pairs), len(self.test_pairs), len(self.pairs), len(self.open_pairs)))
        print('# train images: %d | # mask images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.mask_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.open_pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            unseen_mask = [1 if pair in self.pairs and pair not in self.train_pairs else 0 for pair in self.open_pairs]
            self.unseen_mask = torch.BoolTensor(unseen_mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

            self.pairs = self.open_pairs
            
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.len_mask = len(self.mask_data)


    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [self.root + '/images/' + image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs


    def mask(self, rate = 0.2):
        # self.data = list(map(list, zip(*self.data)))
        # self.data.extend([list(mask)])
        # self.data = list(map(list, zip(*self.data)))
        len_attr, len_obj = 0, 0
        while len_attr!=len(self.attrs) and len_obj!=len(self.objs):
            self.mask_mat = np.random.choice([0, 1], size=len(self.train_pairs), p=[1 - rate, rate])
            self.split_mask()
            len_attr, len_obj = self.data_statistic()
        print("We mask {} pairs in the training stage".format(len(self.train_pairs) - len(self.unmask_train_pairs)))



    def data_statistic(self):
        unmask_attr = []
        unmask_obj = []
        for attr, obj in self.unmask_train_pairs:
            unmask_attr.append(attr)
            unmask_obj.append(obj)
        unmask_attr = list(set(unmask_attr))
        unmask_obj = list(set(unmask_obj))
        return len(unmask_attr), len(unmask_obj)


    def split_mask(self):
        self.unmask_data = []
        self.mask_data = []
        self.unmask_train_pairs = []
        self.mask_train_pairs = []
        for data in self.train_data:
            if self.mask_mat[self.train_pairs.index((data[1], data[2]))]== 0:
                self.unmask_data.append(data)
                self.unmask_train_pairs.append((data[1], data[2]))
            else:
                self.mask_data.append(data)
                self.mask_train_pairs.append((data[1], data[2]))
        self.unmask_train_pairs = list(set(self.unmask_train_pairs))
        self.mask_train_pairs = list(set(self.mask_train_pairs))


    def add_data(self, add_dataset):
        for idx, data in enumerate(add_dataset):
            self.data.append(data)
            self.train_pairs.append((data[1], data[2]))
        self.train_pairs = list(set(self.train_pairs))
        print("We have added {} pairs to the unmask_dset".format(len(self.train_pairs) - len(self.unmask_train_pairs)))   
        print("We have added {} data to the unmask_dset".format(len(self.data) - len(self.unmask_data)))   

    def remove_data(self, remove_dataset):
        for data in remove_dataset:
            self.mask_data.remove(data)
        print("We have {} data in the mask_dset".format(len(self.mask_data)))   

    def sample(self, rate=0.2):
        if len(self.mask_data) < int(self.len_mask * rate):
            self.sample_data = self.mask_data
        else:
            self.sample_data = random.sample(self.mask_data, int(self.len_mask * rate))
        # for data in self.sample_data:
        #     self.unmask_data.append(data)
        #     self.mask_data.remove(data)
        #     self.unmask_train_pairs.append((data[1], data[2]))
        # self.unmask_train_pairs = list(set(self.unmask_train_pairs))
        # print("We mask {} pairs in the training stage".format(len(self.train_pairs) - len(self.unmask_train_pairs)))
        # self.train_pair_to_idx = dict(
        #     [(pair, idx) for idx, pair in enumerate(self.unmask_train_pairs)]
        # )

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train' or self.phase == 'mask':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        return data

    def __len__(self):
        return len(self.data)
