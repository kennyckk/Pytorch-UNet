import logging
import os

import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset,ConcatDataset
from tqdm import tqdm
import torchvision.transforms as T
import elasticdeform.torch as etorch

def unified_dataset(dir_img,dir_mask,img_scale,get_mask_val=False):
    dataset1 = PhC_U373Dataset(os.path.join(dir_img, "01"), os.path.join(dir_mask, "01"), img_scale,binary=True)
    dataset2 = PhC_U373Dataset(os.path.join(dir_img, "02"), os.path.join(dir_mask, "02"), img_scale,binary=True)

    #get the number of values
    mask_vals= dataset1.mask_values
    # since have 2 folders containing different datasets
    dataset = ConcatDataset([dataset1,dataset2])

    return dataset if not get_mask_val else dataset, mask_vals
def aug_transformation(deform=0.5,rotate=0.5, shift_p=0.5,shift_wh:tuple=(0.1,0.1)):
    transform_list=[]

    # elastic transformation
    transform_list.append(Elastic_Deformation(p=deform)) #the probability of having this aug is emphasized
    #rotation invariant
    transform_list.append(T.RandomApply(torch.nn.ModuleList([T.RandomRotation(180)]),p=rotate))
    #shift invarant no rotation only xy translation of maximum 30% of width/height
    transform_list.append(T.RandomApply(torch.nn.ModuleList([T.RandomAffine(0,translate=shift_wh)]),p=shift_p))
    #grey variation TBA

    return T.Compose(transform_list)

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def unique_mask_values(idx, mask_dir, mask_suffix,mask_limit=False): #idx here refer to one file inputted only
    if mask_limit:
        mask_file= list(mask_dir.glob(idx + '.*'))[0] #the mask file will already have the suffix
    else:
        mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0] #idx is not with any file format info
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask) #to extract unique class along the whole map e.g. 0,1,2,3
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1]) #reshape to 2D HWC --> H*W C
        return np.unique(mask, axis=0) #to extract array which is channel wise unique array
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '',img_prefix:str='',mask_limit=False,binary=False):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.img_prefix=img_prefix
        self.mask_limit=mask_limit
        self.binary=binary
        #the file will only be done with splittext if it exists and its name not start with "."
        #the splittext help to divide the filename and its suffix i.e. "filename","txt"
        if self.mask_limit: #custom for mask ground truth is limiting the training imgs
            self.ids=[splitext(file)[0] for file in listdir(mask_dir) if isfile(join(mask_dir,file)) and not file.startswith('.')]
            partial_fnc = partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix,mask_limit=True)
        else:
            self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
            partial_fnc=partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix)

        if not self.ids: #ids contains filename
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial_fnc, self.ids),
                total=len(self.ids)
            )) #it returns a list of single array or arrays [[0,1,2,3],... ]or [[[0,1,2],[1,2,3]],...]

        #concat all unique arrays found from each mask file and further extract unique arrays among whole mask datasets

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod #mask_values are the unique mask values
    def preprocess(mask_values, pil_img, scale, is_mask,binary=False):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # mask and img will be resized differently
        # pil_image object default resize
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img) # expect to be H,W or H,W,C

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64) # will be new H,W zero map
            if not binary: # for multiclass
                for i, v in enumerate(mask_values):
                    if img.ndim == 2:
                        mask[img == v] = i
                    else:
                        mask[(img == v).all(-1)] = i   #check all channel-wise pixels are all True (i.e. equal to that class)
            else: #for 2 class only
                mask[img!=0]=1
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...] #this is to add in one more col H,W--> 1,H,W
            else:
                img = img.transpose((2, 0, 1)) #if RGB --> HWC to CHW

            if (img > 1).any(): # make become [0,1]
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx] #if the mask_limit, the file name is e.g. man_seg001
        if self.mask_limit:
            mask_file = list(self.mask_dir.glob(name + '.*'))
            # the menseg001 --> 001--> t001.*
            img_file=list(self.images_dir.glob(self.img_prefix+name.replace(self.mask_suffix,"") + '.*'))
        else:
            mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
            img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        #should not have any impact when having mask_limit
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False,binary=self.binary)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True,binary=self.binary)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),  # 1(C),H,W,
            'mask': torch.as_tensor(mask.copy()).long().contiguous()  # H,W
        }
class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
class PhC_U373Dataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1,binary=False):
        super().__init__(images_dir, mask_dir, scale,mask_suffix='man_seg',img_prefix='t',mask_limit=True,binary=binary)
class Elastic_Deformation (object):
    def __init__(self,size=3,std=10,p=1):
        self.size=size
        self.std=std
        self.p=p
    def __call__(self, concated):
        if self.p>np.random.random():
            img, mask = concated[0], concated[1]
            displacement = torch.as_tensor(np.random.randn(2, self.size, self.size) * self.std)
            img_t, mask_t = etorch.deform_grid([img, mask], displacement, order=[3, 0])
            return torch.concatenate((torch.unsqueeze(img_t, 0), torch.unsqueeze(mask_t, 0)), 0)
        else:
            return concated
class ApplyAugmentation(Dataset):
    def __init__(self,dataset,transforms=None):
        self.dataset=dataset
        self.transforms=transforms

    def __getitem__(self, idx):
        data=self.dataset[idx]
        img=data['image']
        mask=data['mask']

        if self.transforms:
            # change to float for concate and add one dimension to 1,H,W
            mask = torch.unsqueeze(mask.float(),0)
            # concate them to process consistent aug for img and mask
            concated = torch.concatenate([img, mask], dim=0)  # become 2,H,W
            transformed = self.transforms(concated)  # undergo augmentation
            # print("the augmented data:",transformed.size())
            img = torch.unsqueeze(transformed[0], 0)  # become 1,H,W
            mask = transformed[1].long()  # become H,W dtype long

        return {
            'image': img,
            'mask': mask
        }

    def __len__(self):
        return len(self.dataset)

if __name__ =="__main__":
    # mask_dir = Path("../data/masks/01")
    # mask_suffix="man_seg"
    # ids=os.listdir(mask_dir)
    # _=ids.pop(0)
    # ids=[splitext(id)[0] for id in ids]
    # print(ids)
    # res=[]
    # for id in ids:
    #     res.append(unique_mask_values(id, mask_dir, mask_suffix, mask_limit=True))
    # res=np.concatenate(list(res))
    # print(np.unique(res,axis=0))

    dataset=PhC_U373Dataset('../data/imgs/01','../data/masks/01',0.5)
    data=next(iter(dataset))
    image=torch.squeeze(data['image'])
    mask=data['mask']
    print(image.size(), mask.size())
    show_img = np.uint8(image.numpy() * 255)
    show_mask = np.uint8((mask.numpy() / 1) * 255)

    Image.fromarray(show_img).save("../predicted_mask/img_test.png")
    Image.fromarray(show_mask).save("../predicted_mask/mask_test.png")