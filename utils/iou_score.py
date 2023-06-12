import torch
import torch.nn.functional as F



@torch.inference_mode()
def iou_score (pred, target,num_class):
    smooth=1e-5
    if num_class>=2:
        pred=F.softmax(pred,dim=1).float() #B,C,H,W also normalize to 0,1 across class
        target=F.one_hot(target, num_classes=num_class).permute(0,3,1,2).float().contiguous() #B,C,H,W matching preds
        pred = pred.flatten(0, 1)  # BC,H,W
        target = target.flatten(0, 1)
        sum=(-1,-2)
    else:#for single class:
        pred=(F.sigmoid(pred) > 0.5).float() # B,H,W
        target=target.float() # target already B,H,W in [0,1]
        sum=(-1,-2)

    #calculate intersection
    intersect=(pred*target).sum(dim=sum)
    union=(pred+target).sum(dim=sum)-intersect
    iou= ((intersect+smooth)/(union+smooth)).mean()

    return iou.item()




