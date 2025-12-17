import gc
import logging
import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, matthews_corrcoef
from model_interaction import MB_HGTBot
from dataLoder import Twibot22_subnet_split


from utils import accuracy, init_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(
    des_size, tweet_size, num_prop_size, cat_prop_size, embedding_dimension,
    dropout, num_heads, mlp_hidden_siz, num_layers, temperature,
    lr, weight_decay
) = (
    768, 768, 5, 3, 192, 0.3, 2, 192, 6, 0.3, 1e-3, 5e-3
)

current_time = time.strftime("%Y%m%d-%H%M%S")
log_dir = f'log/MB_HGTBot/Reply/{current_time}'
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)
log_file = os.path.join(log_dir, f'{current_time}_MHCLBot.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

dataset = Twibot22_subnet_split(device=device, process=False, save=False)

(
    des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type,
    labels, train_idx, val_idx, test_idx
) = dataset.dataloader()

saved_model_dir = 'saved_models/MB_HGTBot/Reply/'
os.makedirs(saved_model_dir, exist_ok=True)


def train(epoch):
    model.train()
    output, contrastive_loss = model(
        des_tensor, tweets_tensor, num_prop, category_prop,
        edge_index, edge_type, labels, train_idx=len(train_idx)
    )

    loss_train = loss_fn(output[train_idx], labels[train_idx]) + contrastive_loss
    acc_train = accuracy(output[train_idx], labels[train_idx])

    loss_val = loss_fn(output[val_idx], labels[val_idx]) + contrastive_loss
    acc_val = accuracy(output[val_idx], labels[val_idx])

    output_train = output[train_idx].max(1)[1].to('cpu').detach().numpy()
    label_train = labels[train_idx].to('cpu').detach().numpy()
    f1_train = f1_score(label_train, output_train)
    mcc_train = matthews_corrcoef(label_train, output_train)

    output_val = output[val_idx].max(1)[1].to('cpu').detach().numpy()
    label_val = labels[val_idx].to('cpu').detach().numpy()
    f1_val = f1_score(label_val, output_val)
    mcc_val = matthews_corrcoef(label_val, output_val)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    writer.add_scalar('Loss/train', loss_train.item(), epoch)
    writer.add_scalar('Accuracy/train', acc_train.item(), epoch)
    writer.add_scalar('F1/train', f1_train, epoch)
    writer.add_scalar('MCC/train', mcc_train, epoch)

    writer.add_scalar('Loss/val', loss_val.item(), epoch)
    writer.add_scalar('Accuracy/val', acc_val.item(), epoch)
    writer.add_scalar('F1/val', f1_val, epoch)
    writer.add_scalar('MCC/val', mcc_val, epoch)

    logging.info(
        f'Epoch: {epoch + 1}, '
        f'Train Loss: {loss_train.item():.4f}, Train Acc: {acc_train.item():.4f}, '
        f'Train F1: {f1_train:.4f}, Train MCC: {mcc_train:.4f}, '
        f'Val Loss: {loss_val.item():.4f}, Val Acc: {acc_val.item():.4f}, '
        f'Val F1: {f1_val:.4f}, Val MCC: {mcc_val:.4f}'
    )

    print(
        f'Epoch: {epoch + 1}, '
        f'Train Loss: {loss_train.item():.4f}, Train Acc: {acc_train.item():.4f}, '
        f'Train F1: {f1_train:.4f}, Train MCC: {mcc_train:.4f}, '
        f'Val Loss: {loss_val.item():.4f}, Val Acc: {acc_val.item():.4f}, '
        f'Val F1: {f1_val:.4f}, Val MCC: {mcc_val:.4f}'
    )

    return acc_train, acc_val, loss_train, f1_train, mcc_train


def test():
    model.eval()
    output, contrastive_loss = model(
        des_tensor, tweets_tensor, num_prop, category_prop,
        edge_index, edge_type, labels, train_idx=len(train_idx)
    )

    loss_test = loss_fn(output[test_idx], labels[test_idx]) + contrastive_loss
    acc_test = accuracy(output[test_idx], labels[test_idx])

    output_test = output[test_idx].max(1)[1].to('cpu').detach().numpy()
    label_test = labels[test_idx].to('cpu').detach().numpy()
    f1_test = f1_score(label_test, output_test)
    mcc_test = matthews_corrcoef(label_test, output_test)

    writer.add_scalar('Loss/test', loss_test.item())
    writer.add_scalar('Accuracy/test', acc_test.item())
    writer.add_scalar('F1/test', f1_test)
    writer.add_scalar('MCC/test', mcc_test)

    logging.info(
        f'Test set results: Loss: {loss_test.item():.4f}, '
        f'Accuracy: {acc_test.item():.4f}, F1 Score: {f1_test:.4f}, MCC: {mcc_test:.4f}'
    )

    print(
        f'Test set results: Loss: {loss_test.item():.4f}, '
        f'Accuracy: {acc_test.item():.4f}, F1 Score: {f1_test:.4f}, MCC: {mcc_test:.4f}'
    )

    if acc_test.item() > 0.81:
        model_save_path = os.path.join(saved_model_dir, f'{current_time}_MHCLBot.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
        logging.info(f'Model saved to {model_save_path}')

    return acc_test, loss_test, f1_test, mcc_test


test_acc_list, test_loss_list, f1_list, mcc_list = [], [], [], []

for run in range(1):
    epochs = 400
    patience = 20
    print(f'Run {run + 1}/5')
    logging.info(f'Run {run + 1}/5')

    model = MB_HGTBot(
        des_size=des_size,
        tweet_size=tweet_size,
        num_prop_size=num_prop_size,
        cat_prop_size=cat_prop_size,
        embedding_dimension=embedding_dimension,
        dropout=dropout,
        num_heads=num_heads,
        mlp_hidden_size=mlp_hidden_siz,
        num_layers=num_layers,
        temperature=temperature
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.apply(init_weights)

    best_acc_val = 0
    counter = 0
    for epoch in range(epochs):
        _, acc_val, _, _, _ = train(epoch)

        if acc_val - best_acc_val > 0.01:
            best_acc_val = acc_val
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    acc_test, loss_test, f1, mcc = test()

    test_acc_list.append(acc_test.item())
    test_loss_list.append(loss_test.item())
    f1_list.append(f1)
    mcc_list.append(mcc)

    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()

test_acc_mean, test_acc_std = np.mean(test_acc_list), np.std(test_acc_list)
test_loss_mean, test_loss_std = np.mean(test_loss_list), np.std(test_loss_list)
f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
mcc_mean, mcc_std = np.mean(mcc_list), np.std(mcc_list)

print(f'Final results over 5 runs:')
print(f'Accuracy: {test_acc_mean:.4f} ± {test_acc_std:.4f}')
print(f'Loss: {test_loss_mean:.4f} ± {test_loss_std:.4f}')
print(f'F1 Score: {f1_mean:.4f} ± {f1_std:.4f}')
print(f'MCC: {mcc_mean:.4f} ± {mcc_std:.4f}')

logging.info(f'Final results over 5 runs:')
logging.info(f'Accuracy: {test_acc_mean:.4f} ± {test_acc_std:.4f}')
logging.info(f'Loss: {test_loss_mean:.4f} ± {test_loss_std:.4f}')
logging.info(f'F1 Score: {f1_mean:.4f} ± {f1_std:.4f}')
logging.info(f'MCC: {mcc_mean:.4f} ± {mcc_std:.4f}')

writer.close()