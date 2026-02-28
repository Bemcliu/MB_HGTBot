import argparse
import os
import torch
from torch import nn
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

# 导入您的模块 (请确保模块名与您的项目结构一致)
from model_topics import MB_HGTBot
from data_preprocess.abortion import Abortion
# from data_preprocess.lgbtq import Lgbtq
# from data_preprocess.trumpAttack import TrumpAttacked
from utils import accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Test MB_HGTBot from a saved checkpoint.")
    # 默认路径可以修改为您最常使用的路径
    parser.add_argument('--checkpoint', type=str, default='1212-返稿数据/ours_0_5/abortion/MB-HGTBot/20251215-192906/20251215-192906_MHCLBot.pth',
                        help='Path to the saved model checkpoint (.pth file)')

    # parser.add_argument('--checkpoint', type=str,
    #                     default='1212-返稿数据/ours_0_5/lgbtq/MB-HGTBot/20251215-204816/20251215-204816_MHCLBot.pth',
    #                     help='Path to the saved model checkpoint (.pth file)')

    # parser.add_argument('--checkpoint', type=str,
    #                     default='1212-返稿数据/ours_0_5/trump_attacked/MB-HGTBot/20251215-162333/20251215-162333_MHCLBot.pth',
    #                     help='Path to the saved model checkpoint (.pth file)')


    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 1. 基础设置与超参数 (必须与 main_interaction.py 训练时保持绝对一致)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    (
        des_size, tweet_size, num_prop_size, comments_prop_size, embedding_dimension,
        dropout, num_heads, mlp_hidden_size, num_layers, temperature
    ) = (
        768, 768, 6, 6, 192, 0.3, 2, 192, 6, 0.3
    )

    # 2. 加载数据集
    print("Loading data...")
    dataset = Abortion(device=device, process=False, save=False)
    # dataset = Lgbtq(device=device, process=False, save=False)
    # dataset = TrumpAttacked(device=device, process=False, save=False)

    (
        tweets_tensor, num_prop, comments_prop, edge_index, edge_type,
        labels, train_idx, val_idx, test_idx
    ) = dataset.dataloader()

    # 3. 初始化模型并加载权重
    print("Initializing model...")
    model = MB_HGTBot(
        tweet_size=tweet_size,
        num_prop_size=num_prop_size,
        comments_prop_size=comments_prop_size,
        embedding_dimension=embedding_dimension,
        dropout=dropout,
        num_heads=num_heads,
        mlp_hidden_size=mlp_hidden_size,
        num_layers=num_layers,
        temperature=temperature
    ).to(device)

    print(f"Loading checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    loss_fn = nn.CrossEntropyLoss()

    # 4. 评估测试集
    print("Starting evaluation on test set...")
    model.eval()  # 开启评估模式，关闭 Dropout 等机制

    with torch.no_grad():  # 禁用梯度计算以节省显存和加速
        output, contrastive_loss = model(
            tweets_tensor, num_prop, comments_prop,
            edge_index, edge_type, labels, train_idx=len(train_idx)
        )

        # 计算测试集损失和准确率
        loss_test = loss_fn(output[test_idx], labels[test_idx]) + contrastive_loss
        acc_test = accuracy(output[test_idx], labels[test_idx])

        # 获取预测标签与真实标签，计算 F1 和 MCC
        output_test = output[test_idx].max(1)[1].to('cpu').detach().numpy()
        label_test = labels[test_idx].to('cpu').detach().numpy()

        f1_test = f1_score(label_test, output_test)
        mcc_test = matthews_corrcoef(label_test, output_test)

    # 5. 打印最终结果
    print("\n" + "=" * 50)
    print("Testing Results:")
    print("=" * 50)
    print(f"Accuracy : {acc_test.item():.4f}")
    print(f"F1 Score : {f1_test:.4f}")
    print(f"MCC      : {mcc_test:.4f}")
    print(f"Loss     : {loss_test.item():.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()