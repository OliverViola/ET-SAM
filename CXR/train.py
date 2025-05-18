import numpy as np
import torch
import json
import os
from loss import MixedEdgeWeightedLoss
from sam2.build_sam import build_sam2
from utils import select_few_shot_samples
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
from dataset import SAMDataset
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random

# Hyper Parameters
NUM_EPOCHS = 210
LR = 5e-4
WD = 1e-5
DEVICE="cuda"
BS = 5  # Reduced batch size for smaller training set
SEED = 42 # universe secret
DATASET = SAMDataset("/root/lung-segment/train.csv")
OPEN_KERNEL_SIZE = 14
NUM_TRAIN = 10  # Number of images for few-shot learning

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_model(ckpt='/root/sam2/checkpoints/sam2.1_hiera_small.pt', cfg='configs/sam2.1/sam2.1_hiera_s.yaml'):
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

def extract_features(predictor, dataset):
    """Extract image features using SAM2's encoder"""
    features = []
    predictor.model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Extracting features"):
            image, _ = dataset[idx]
            # Convert to numpy and create batch of 1
            image_np = image
            # Set image and get features
            predictor.set_image(image_np)
            # Get image embedding from predictor's features
            feature = predictor._features["image_embed"]
            features.append(feature)
            # print(features[0].device)
    return features

def few_shot_train():
    """Train with few-shot learning setup using similarity-based sample selection"""
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir="/root/tf-logs")

    # Load model first to extract features
    predictor = load_model()

    print("Extracting features for sample selection...")
    # Extract features for all samples
    features = extract_features(predictor, DATASET)
    
    # Select samples using similarity-based approach
    print("Selecting samples using similarity-based approach...")
    support_indices, query_groups = select_few_shot_samples(features, k=NUM_TRAIN)
    
    # Flatten query groups to get validation indices
    val_idx = []
    for group in query_groups:
        val_idx.extend(group)
    
    print(f"Training with {len(support_indices)} images, validating with {len(val_idx)} images")
    print(f"Selected training indices: {support_indices}")

    train_subset = Subset(DATASET, support_indices)
    val_subset = Subset(DATASET, val_idx)

    train_loader = DataLoader(train_subset, BS, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_subset, BS*5, num_workers=10)

    # Reset model gradients for training
    predictor = load_model()  # Reload model to reset state

    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda')  # mixed precision
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,           # Initial period length
        T_mult=2,         # Period length multiplier
        eta_min=1e-6
    )
    criterion = MixedEdgeWeightedLoss()
    
    # Train with the few-shot setup
    train_model(predictor, train_loader, val_loader, optimizer, scaler, scheduler, writer, criterion)

    # Close Writer
    writer.close()

def train_model(predictor:SAM2ImagePredictor, train_loader, val_loader, optimizer, scaler, scheduler, writer, criterion):
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

                writer.add_scalar("Few_Shot/train_loss", loss.item(), train_step)
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
        writer.add_scalar('Few_Shot/Train Mean IoU', np.mean(ious), epoch)
        writer.add_scalar('Few_Shot/Train Mean Dice', np.mean(dices), epoch)
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

                loss, iou, dice = criterion(gt_masks, prd_mask, prd_scores)
                val_losses.append(loss.item())
                
                writer.add_scalar("Few_Shot/validation_loss", loss.item(), val_step)
                val_step += 1
                validation_loss.append(loss.item())

                iou = iou.mean().cpu().detach().numpy()
                dice = dice.mean().cpu().detach().numpy()
                # print(f'validation mean iou {np.mean(iou)}; validation mean dice {np.mean(dice)}')
                ious.append(float(np.mean(iou)))
                dices.append(float(np.mean(dice)))

        print(f"Epoch {epoch} Validation Mean Loss: {np.mean(val_losses)} Validation Mean IoU: {np.mean(ious)}; Validation Mean Dice: {np.mean(dices)}")
        writer.add_scalar('Few_Shot/Validation Mean IoU', np.mean(ious), epoch)
        writer.add_scalar('Few_Shot/Validation Mean Dice', np.mean(dices), epoch)

        epoch_validation_dice.append(float(np.mean(dices)))
        epoch_validation_iou.append(float(np.mean(ious)))

        if np.mean(dices) > max_dice:
            # 修改为符合 build_sam2 要求的格式
            checkpoint = {
                "model": predictor.model.state_dict(),
            }
            # 保存检查点
            max_dice = np.mean(dices)
            torch.save(checkpoint, f"/root/sam2/CXR/few_shot_finetuned_sam2.1.pt")
        
        max_iou = max(max_iou, np.mean(ious))
    
    save_training_data('/root/sam2/CXR/few_shot_k_center', 0, epoch_train_dice, epoch_train_iou, train_loss, True)
    save_training_data('/root/sam2/CXR/few_shot_k_center', 0, epoch_validation_dice, epoch_validation_iou, validation_loss, False)
    print(f"Few-shot training finished! Max Validation dice is {max_dice}, max Validation IoU is {max_iou}")

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

# Call the few-shot training function
few_shot_train()

