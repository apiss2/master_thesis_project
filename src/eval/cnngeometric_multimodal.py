# 汎用ライブラリ
import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp
# torch関連
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# 自作ファイル
from custom_dataset import Dataset_fromDir, PairImageDataset
from models import CNNGeometric
from transform_util import CenterTransform

if __name__ == '__main__':
    #各種定数の設定
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    base = args.base
    size = args.size

    print('-------------------------------------')
    print('モデルのロード中...')
    ch = args.fe_output_channel
    s = args.fe_output_size
    output_dim_1 = geometric_choice(args.geometric_1)
    model_1 = CNNGeometric(select_model=args.select_model,
                output_dim=output_dim_1, fe_output_shape=(ch,s,s),
                fr_channels=[ch,128,64], use_cuda=use_cuda,
                pretrain_path=args.FE_model)
    model_1.FeatureRegression.load_state_dict(torch.load(args.model_1))
    model_1.eval()
    if args.two_stage:
        output_dim_2 = geometric_choice(args.geometric_2)
        model_2 = CNNGeometric(select_model=args.select_model,
                    output_dim=output_dim_2, fe_output_shape=(ch,s,s),
                    fr_channels=[ch,128,64], use_cuda=use_cuda,
                    pretrain_path=args.FE_model)
        model_2.FeatureRegression.load_state_dict(torch.load(args.model_2))
        model_2.eval()
    # ここまでは同じ

    print('-------------------------------------')
    print('dataloaderの作成中...')
    dataset = PairImageDataset(image_path_1=args.src_image_path,
                               image_path_2=args.tgt_image_path,
                               label_path_1=args.src_label_path,
                               label_path_2=args.tgt_label_path,
                               center_crop=size)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)
    center_transform = CenterTransform(use_cuda=use_cuda, geometric=args.geometric_1,\
                        base=base, size=size)
    if args.two_stage:
        center_transform_2 = CenterTransform(use_cuda=use_cuda, geometric=args.geometric_2,\
                            base=base, size=size)
    criterion = smp.utils.losses.DiceLoss()
    loss=list()

    print('-------------------------------------')
    print('評価中...')
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            # gpu(cpu)に移動
            src_image = batch['image_1'].to(device)
            src_label = batch['label_1'].to(device)
            tgt_image = batch['image_2'].to(device)
            tgt_label = batch['label_2'].to(device)
            raw_image_1 = batch['raw_image_1'].to(device)

            # モデルの予測
            pred_theta_1 = model_1(src_image, tgt_image)
            # 予測によって画像を変形
            pred_label = center_transform.transform(src_label, pred_theta_1)
            #2段階予測の場合
            if args.two_stage:
                pred_image = center_transform.transform(src_image, pred_theta_1)
                # モデルの予測
                pred_theta_2 = model_2(pred_image, tgt_image)
                # 予測によって画像を変形
                pred_label = center_transform_2.transform(pred_label, pred_theta_2)
            # Dice係数を計算
            loss.append(1-criterion(tgt_label, pred_label).item())

            if args.save:
                pred_image = center_transform.transform(raw_image_1, pred_theta_1)
                if args.two_stage:
                    pred_image = center_transform_2.transform(pred_image, pred_theta_2)
                image = pred_image.cpu().numpy()[0][0]
                cv2.imwrite('./save/images/{}.png'.format(i), image*255)
    df = pd.DataFrame(loss)
    df.to_csv('dice_values.csv')

    print('-------------------------------------')
    print("Dice: %5f\n"%(np.array(loss).mean()))
    print(df.describe())