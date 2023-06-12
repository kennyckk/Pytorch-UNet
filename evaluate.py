import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,ConcatDataset
from tqdm import tqdm
import os

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.iou_score import iou_score
from utils.data_loading import unified_dataset,PhC_U373Dataset
from unet import UNet

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, iou=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    if iou:
        iou_scores=0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if iou:
                iou_scores+=iou_score(mask_pred,mask_true,net.n_classes)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #dice_score += multiclass_dice_coeff(mask_pred[:, :], mask_true[:, :], reduce_batch_first=False)
    net.train()
    return dice_score / max(num_val_batches, 1) if not iou else (dice_score / max(num_val_batches, 1),iou_scores/max(num_val_batches, 1))

if __name__ == "__main__":
    # to predict for dice score and
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_img="./data/imgs"
    dir_mask="./data/masks"
    img_scale=0.5
    num_class=2
    check_pt="checkpoint.pth"
    amp=False

    # load in model and trained weights
    unet= UNet(n_channels=1, n_classes=2, bilinear=False)
    unet.to(device)
    state_dict=torch.load(check_pt, map_location=device)
    state_dict.pop("mask_values")
    unet.load_state_dict(state_dict)
    print("loaded state_dict successfully")

    #load in data to be sample with
    dataset=unified_dataset(dir_img, dir_mask, img_scale)
    dataloader=DataLoader(dataset, shuffle=True,batch_size=1,pin_memory=True,drop_last=True)

    dice_score, iou_score =evaluate(unet,dataloader,device,amp,iou=True)

    print("the average iou scores for the inputted datasets are:{}".format(iou_score))
    print("the average Dice scores for the inputted datasets are:{}".format(dice_score))