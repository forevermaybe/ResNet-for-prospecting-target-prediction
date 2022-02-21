import os
import os.path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
import numpy as np
import dataset.transfer as transforms1


def readdatas(root):
    '''Read geochemical element datas'''
    result = []
    d0 = np.loadtxt(root + 'CU.txt')
    d1 = np.loadtxt(root + 'AU.txt')
    d2 = np.loadtxt(root + 'MO.txt')
    d3 = np.loadtxt(root + 'SN.txt')
    d4 = np.loadtxt(root + 'BA.txt')
    d5 = np.loadtxt(root + 'AG.txt')
    d6 = np.loadtxt(root + 'SB.txt')
    d7 = np.loadtxt(root + 'HG.txt')
    result.append(d0)
    result.append(d1)
    result.append(d2)
    result.append(d3)
    result.append(d4)
    result.append(d5)
    result.append(d6)
    result.append(d7)
    result = np.array(result)
    result = result.transpose((1, 2, 0))
    return result


def getdata(datas, i, j, ix, jx, size):
    '''

    :param datas:
    :param i: row
    :param j: column
    :param ix: Horizontal offset
    :param jx: Vertical offset
    :param size: Data size
    :return:
    '''
    i = int(i)
    j = int(j)
    ix = int(ix)
    jx = int(jx)
    return datas[(i - 1) * size + ix:i * size + ix, (j - 1) * size + jx:j * size + jx, :]


class dataSet(VisionDataset):

    def __init__(self, root, transform_common=None, transform=None, target_transform=None, transforms=None, geoset=None,
                 oldsize=256):
        super(dataSet, self).__init__(root, transforms, transform, target_transform)
        label = 'train'
        labelpath = os.path.join(root, label)
        geos = []
        classesitems = os.listdir(labelpath)
        for classesitem in classesitems:
            datas = np.loadtxt(os.path.join(labelpath, classesitem))
            idx = int(classesitem.replace('.txt', ''))
            for data in datas:
                geos.append((data[0], data[1], data[2], data[3], idx))
        self.geos = geos
        self.transform_common = transform_common
        self.oldsize = oldsize
        self.geoset = geoset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        geo = getdata(self.geoset, self.geos[index][0], self.geos[index][1], self.geos[index][2], self.geos[index][3],
                        self.oldsize)
        if self.transform_common is not None:
            geo = self.transform_common(geo)
            geo = np.array(geo)
            geo = geo.transpose((2, 0, 1))
        return geo, self.geos[index][4]

    def __len__(self):
        return len(self.geos)


def get_train_loader(root, batch_size):
    dataset = dataSet(root, transform_common=transforms.Compose([
        transforms1.RandomHorizontalFlip(),
        transforms1.RandomResizedCrop(224, scale=(0.7, 1))
    ]), geoset=readdatas('data/'))
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return train_loader
