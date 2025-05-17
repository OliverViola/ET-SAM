import numpy as np
import torch
import json
import os
from loss import MixedEdgeWeightedLoss
from sam2.build_sam import build_sam2
from transform import morphological_open
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from dataset import SAMDataset
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
# import hydra

# # init hydra
# hydra.core.global_hydra.GlobalHydra.instance().clear()
# hydra.initialize_config_module('./sam2_configs', version_base='1.2')

# Hyper Parameters
NUM_EPOCHS = 210
LR = 5e-4
WD = 5e-5
DEVICE="cuda"
BS = 12
SEED = 42 # universe secret
DATASET = SAMDataset("/root/lung-segment/train.csv")
OPEN_KERNEL_SIZE = 14

def load_model(ckpt='/root/sam2/checkpoints/sam2.1_hiera_base_plus.pt', cfg='configs/sam2.1/sam2.1_hiera_b+.yaml'):
    model = build_sam2(cfg, ckpt, device='cuda')
    predictor = SAM2ImagePredictor(model)
    # set training parameters
    pretrained_model = torch.load(ckpt, map_location="cpu", weights_only=True)['model']
    old_param_names = pretrained_model.keys()
    for name, param in model.named_parameters():
        if name in old_param_names and 'mask_tokens' not in name and 'iou_token' not in name  and 'not_a_point_embed' not in name and 'point_embeddings' not in name: # 
            param.requires_grad=False
        else:
            param.requires_grad=True
    return predictor


def k_fold_train(k):
    """生成普通五折划分方案"""
    # 初始化TensorBoard Writer
    writer = SummaryWriter(log_dir="/root/tf-logs")

    kf = KFold(k, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(DATASET)))):
        print(f"\n========= 训练第 {fold+1} 折 =========")
        train_subset = Subset(DATASET, train_idx)
        val_subset = Subset(DATASET, val_idx)

        train_loader = DataLoader(train_subset, BS, shuffle=True, num_workers=10)
        val_loader = DataLoader(val_subset, BS*2, num_workers=10)

        predictor = load_model()

        optimizer=torch.optim.AdamW(params=predictor.model.parameters(), lr=LR, weight_decay=WD)
        scaler = torch.amp.GradScaler('cuda') # mixed precision
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=30,           # 初始周期长度（例如50个epoch）
            T_mult=2,         # 周期长度倍增因子（每个周期后T=T*T_mult）
            eta_min=1e-6
        )
        criterion = MixedEdgeWeightedLoss()
        train(fold, predictor, train_loader, val_loader, optimizer, scaler, scheduler, writer, criterion)


    # 关闭Writer
    writer.close()

def predict_mask(predictor:SAM2ImagePredictor, points, labels, boxes, mask):
    # prompt encoding
    if points is not None:
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(points, labels), boxes=boxes,masks=mask)
    else:
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=None, boxes=boxes,masks=mask)

    #print(predictor._features["image_embed"].shape)
    # mask decoder
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"],image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=False,repeat_image=False,high_res_features=high_res_features,)
    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution
    # print(low_res_masks.min(), low_res_masks.max(), torch.median(low_res_masks), low_res_masks.shape)
    prd_mask = torch.sigmoid(prd_masks[:, 0]) # Turn logit map to probability map
    return low_res_masks, prd_mask, prd_scores

def save_training_data(folder_path, fold, dices, ious, loss, train:bool):
    base = os.path.join(folder_path, f'fold_{fold}')
    if not os.path.exists(os.path.join(base)):
        os.makedirs(os.path.join(folder_path, f'fold_{fold}'))
    
    if train:
        method = 'train'
    else:
        method = 'validation'

    with open(os.path.join(base, f'{method}_loss.json'), 'w') as f:
        json.dump(loss, f)

    with open(os.path.join(base, f'{method}_iou.json'), 'w') as f:
        json.dump(ious, f)
    
    with open(os.path.join(base, f'{method}_dice.json'), 'w') as f:
        json.dump(dices, f)

def calculate_loss(gt_masks, prd_mask, prd_scores):
    # Segmentation Loss calculation
    seg_loss = (-gt_masks * torch.log(prd_mask + 1e-6) - (1 - gt_masks) * torch.log((1 - prd_mask) + 1e-6)).mean() # cross entropy loss
    print(torch.unique(gt_masks), gt_masks.dtype, gt_masks.shape, torch.unique(seg_loss), seg_loss.dtype)
    # Score loss calculation (intersection over union) IOU
    # print(gt_mask.sum(1).shape, prd_mask.sum(1).sum(1).shape)
    inter = (gt_masks * (prd_mask > 0.5)).sum(1).sum(1)
    iou = (inter / (gt_masks.sum((1, 2)) + (prd_mask > 0.5).sum((1, 2)) - inter)).mean()
    iou_loss = (1 - iou)
    dice = ((2 * iou) / (1 + iou)).mean()
    dice_loss = (1 - dice)
    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
    loss=seg_loss + (score_loss + iou_loss + dice_loss) * 0.05 # mix losses
    return loss, iou, dice

def train(fold, predictor:SAM2ImagePredictor, train_loader, val_loader, optimizer, scaler, scheduler, writer, criterion):
    # training iteration
    max_iou = 0
    max_dice = 0
    train_step = 0
    val_step = 0
    train_loss = []
    epoch_train_dice = []
    epoch_train_iou = []
    validation_loss = []
    epoch_validation_dice = []
    epoch_validation_iou = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        ious = []
        dices = []
        train_losses = []
        print(f"epoch {epoch} is running")
        predictor.model.train()
        for batch in train_loader:
            with torch.amp.autocast('cuda'): # cast to mix precision
                images, masks = batch
                images, gt_masks = images, masks.cuda()
                images = [
                    images[i].numpy()  # 移除梯度+转移到CPU+转numpy
                    for i in range(images.shape[0])   # 按第一个维度遍历
                ]
                predictor.set_image_batch(images) # apply SAM image encoder to the image

                # one-stage
                low_res_masks, prd_mask, prd_scores = predict_mask(predictor, None, None, None, None)
                
                # print(prd_mask.shape)
                if epoch >= 5:
                    prompts = DATASET.find_entities(prd_mask)
                    if prompts != False:
                        points, labels, boxes = prompts
                        points, labels, boxes = points.cuda(), labels.cuda(), boxes.cuda()
                        # two-stage
                        low_res_masks, prd_mask, prd_scores = predict_mask(predictor, points, labels, boxes, low_res_masks)

                # print(gt_masks.shape, )
                loss, iou, dice = criterion(gt_masks, prd_mask, prd_scores)
                train_losses.append(loss.item())
                
                # apply back propogation
                predictor.model.zero_grad() # empty gradient
                scaler.scale(loss).backward()  # Backpropogate
                scaler.step(optimizer)
                scaler.update() # Mix precision

                writer.add_scalar(f"Fold_{fold}/train_loss", loss.item(), train_step)
                train_step += 1
                train_loss.append(loss.item())

                # Display results
                iou = iou.mean().cpu().detach().numpy()
                dice = dice.mean().cpu().detach().numpy()
                # print(f'train mean iou {np.mean(iou)}; train mean dice {np.mean(dice)}')
                ious.append(float(np.mean(iou)))
                dices.append(float(np.mean(dice)))

        scheduler.step()
        print(f"Epoch {epoch} Training Mean Loss {np.mean(train_losses)} Training Mean IoU: {np.mean(ious)}; Training Mean Dice: {np.mean(dices)}, lr={optimizer.param_groups[0]['lr']}")
        writer.add_scalar(f'Fold_{fold}/Train Mean IoU', np.mean(ious), epoch)
        writer.add_scalar(f'Fold_{fold}/Train Mean Dice', np.mean(dices), epoch)
        epoch_train_dice.append(float(np.mean(dices)))
        epoch_train_iou.append(float(np.mean(ious)))

        ious = []
        dices = []
        val_losses = []
        predictor.model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images, gt_masks = images, masks.cuda()
                images = [
                    images[i].numpy()  # 移除梯度+转移到CPU+转numpy
                    for i in range(images.shape[0])   # 按第一个维度遍历
                ]
                predictor.set_image_batch(images) # apply SAM image encoder to the image

                # one-stage
                low_res_masks, prd_mask, prd_scores = predict_mask(predictor, None, None, None, None)
                if epoch >= 5:
                    prompts = DATASET.find_entities(prd_mask)
                    if prompts != False:
                        points, labels, boxes = prompts
                        points, labels, boxes = points.cuda(), labels.cuda(), boxes.cuda()
                        # two-stage
                        low_res_masks, prd_mask, prd_scores = predict_mask(predictor, points, labels, boxes, low_res_masks)

                # prd_mask = morphological_open(prd_mask.unsqueeze(1), OPEN_KERNEL_SIZE).squeeze(1)

                loss, iou, dice = criterion(gt_masks, prd_mask, prd_scores)
                val_losses.append(loss.item())
                
                writer.add_scalar(f"Fold_{fold}/validation_loss", loss.item(), val_step)
                val_step += 1
                validation_loss.append(loss.item())

                iou = iou.mean().cpu().detach().numpy()
                dice = dice.mean().cpu().detach().numpy()
                # print(f'validation mean iou {np.mean(iou)}; validation mean dice {np.mean(dice)}')
                ious.append(float(np.mean(iou)))
                dices.append(float(np.mean(dice)))

        print(f"Epoch {epoch} Validation Mean Loss: {np.mean(val_losses)} Validation Mean IoU: {np.mean(ious)}; Validation Mean Dice: {np.mean(dices)}")
        writer.add_scalar(f'Fold_{fold}/Validation Mean IoU', np.mean(ious), epoch)
        writer.add_scalar(f'Fold_{fold}/Validation Mean Dice', np.mean(dices), epoch)

        epoch_validation_dice.append(float(np.mean(dices)))
        epoch_validation_iou.append(float(np.mean(ious)))

        if np.mean(dices) > max_dice:
            # 修改为符合 build_sam2 要求的格式
            checkpoint = {
                "model": predictor.model.state_dict(), # 将参数存入 'model' 键
            }
            # 保存检查点
            max_dice = np.mean(dices)
            torch.save(checkpoint, f"/root/sam2/CXR/fold_{fold}_finetuned_sam2.1.pt")
        
        max_iou = max(max_iou, np.mean(ious))
    
    save_training_data('/root/sam2/CXR/two_stage_md_mp', fold, epoch_train_dice, epoch_train_iou, train_loss, True)
    save_training_data('/root/sam2/CXR/one_stage_md_mp', fold, epoch_validation_dice, epoch_validation_iou, validation_loss, False)
    print(f"FOLD {fold} finished! Max Validation dice is {max_dice}, max Validation IoU is {max_iou}")

k_fold_train(5)

