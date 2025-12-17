import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime as dt

path = '../'

user = pd.read_json(path + 'user.json')
edge = pd.read_csv(path + 'edges.csv')

user_idx = user['id']

uid_index = {uid: index for index, uid in enumerate(user_idx.values)}

user_index_to_uid = list(user.id)

uid_to_user_index = {x: i for i, x in enumerate(user_index_to_uid)}

# split = pd.read_csv(path + 'split.csv')
label = pd.read_csv(path + 'label.csv')

uid_label = {uid: label for uid, label in zip(label['id'].values, label['label'].values)}

# uid_split = {uid: split for uid, split in zip(split['num'].values, split['split'].values)}

label_new = []
# train_idx = []
# test_idx = []
# val_idx = []

for i, uid in enumerate(tqdm(user_idx.values)):
    single_label = uid_label[uid]
    # single_split = uid_split[uid]
    if single_label == 'human':
        label_new.append(0)
    else:
        label_new.append(1)
    # if single_split == 'train':
    #     train_idx.append(i)
    # elif single_split == 'test':
    #     test_idx.append(i)
    # else:
    #     val_idx.append(i)

labels = torch.LongTensor(label_new)
# train_mask = torch.LongTensor(train_idx)
# valid_mask = torch.LongTensor(val_idx)
# test_mask = torch.LongTensor(test_idx)
# torch.save(train_mask, "../twibot_20/BotRGCN-main/Twibot-22_original/train_idx.pt")
# torch.save(valid_mask, "../twibot_20/BotRGCN-main/Twibot-22_original/val_idx.pt")
# torch.save(test_mask, "../twibot_20/BotRGCN-main/Twibot-22_original/test_idx.pt")
# torch.save(labels, '../twibot_20/BotRGCN-main/Twibot-22_original/label.pt')
# torch.save(train_mask, "./train_idx.pt")
# torch.save(valid_mask, "./val_idx.pt")
# torch.save(test_mask, "./test_idx.pt")
torch.save(labels, './label.pt')

edge_index = []
edge_type = []
for i in tqdm(range(len(edge))):
    sid = edge['source_id'][i]
    tid = edge['target_id'][i]
    if edge['relation'][i] == 'replied_to':
        try:
            edge_index.append([uid_index[sid], uid_index[tid]])
            edge_type.append(0)
        except KeyError:
            continue
    elif edge['relation'][i] == 'ed_replied_to':
        try:
            edge_index.append([uid_index[sid], uid_index[tid]])
            edge_type.append(1)
        except KeyError:
            continue

torch.save(torch.LongTensor(edge_index).t(), "./edge_index.pt")
torch.save(torch.LongTensor(edge_type), "./edge_type.pt")

following_count = []
for i, each in enumerate(user['public_metrics']):
    if i == len(user):
        break
    if each is not None and isinstance(each, dict):
        if each['following_count'] is not None:
            following_count.append(each['following_count'])
        else:
            following_count.append(0)
    else:
        following_count.append(0)

statues = []
for i, each in enumerate(user['public_metrics']):
    if i == len(user):
        break
    if each is not None and isinstance(each, dict):
        if each['tweet_count'] is not None:
            statues.append(each['tweet_count'])
        else:
            statues.append(0)
    else:
        statues.append(0)

followers_count = []
for each in user['public_metrics']:
    if each is not None and each['followers_count'] is not None:
        followers_count.append(int(each['followers_count']))
    else:
        followers_count.append(0)

num_username = []
for each in user['username']:
    if each is not None:
        num_username.append(len(each))
    else:
        num_username.append(int(0))

created_at = user['created_at']
created_at = pd.to_datetime(created_at, unit='s')

followers_count = pd.DataFrame(followers_count)
followers_count = (followers_count - followers_count.mean()) / followers_count.std()
followers_count = torch.tensor(np.array(followers_count), dtype=torch.float32)

date0 = dt.strptime('Tue Sep 5 00:00:00 +0000 2020 ', '%a %b %d %X %z %Y ')
active_days = []
for each in created_at:
    active_days.append((date0 - each).days)

active_days = pd.DataFrame(active_days)
active_days = active_days.fillna(int(1)).astype(np.float32)

screen_name_length = []
for each in user['name']:
    if each is not None:
        screen_name_length.append(len(each))
    else:
        screen_name_length.append(int(0))

followers_count = (followers_count - followers_count.mean()) / followers_count.std()
followers_count = torch.tensor(np.array(followers_count), dtype=torch.float32)

active_days = pd.DataFrame(active_days)
active_days.fillna(int(0))
active_days = active_days.fillna(int(0)).astype(np.float32)

active_days = (active_days - active_days.mean()) / active_days.std()
active_days = torch.tensor(np.array(active_days), dtype=torch.float32)

screen_name_length = pd.DataFrame(screen_name_length)
screen_name_length = (screen_name_length - screen_name_length.mean()) / screen_name_length.std()
screen_name_length = torch.tensor(np.array(screen_name_length), dtype=torch.float32)

following_count = pd.DataFrame(following_count)
following_count = (following_count - following_count.mean()) / following_count.std()
following_count = torch.tensor(np.array(following_count), dtype=torch.float32)

statues = pd.DataFrame(statues)
statues = (statues - statues.mean()) / statues.std()
statues = torch.tensor(np.array(statues), dtype=torch.float32)

num_properties_tensor = torch.cat(
    [followers_count, active_days, screen_name_length, following_count, statues],
    dim=1
)

pd.DataFrame(num_properties_tensor.detach().numpy()).isna().value_counts()

protected = user['protected']
verified = user['verified']

protected_list = []
for each in protected:
    if each == True:
        protected_list.append(1)
    else:
        protected_list.append(0)

verified_list = []
for each in verified:
    if each == True:
        verified_list.append(1)
    else:
        verified_list.append(0)

default_profile_image = []
for each in user['profile_image_url']:
    if each is not None:
        if each == 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png':
            default_profile_image.append(int(1))
        elif each == '':
            default_profile_image.append(int(1))
        else:
            default_profile_image.append(int(0))
    else:
        default_profile_image.append(int(1))

protected_tensor = torch.tensor(protected_list, dtype=torch.float)
verified_tensor = torch.tensor(verified_list, dtype=torch.float)
default_profile_image_tensor = torch.tensor(default_profile_image, dtype=torch.float)

cat_properties_tensor = torch.cat(
    [
        protected_tensor.reshape([32209, 1]),
        verified_tensor.reshape([32209, 1]),
        default_profile_image_tensor.reshape([32209, 1])
    ],
    dim=1
)

torch.save(num_properties_tensor, './num_properties_tensor.pt')
torch.save(cat_properties_tensor, './cat_properties_tensor.pt')

# print("extracting each_user's tweets")
# id_tweet = {i: [] for i in range(len(user_idx))}
# for i in range(9):
#     name = 'tweet_' + str(i) + '.json'
#     user_tweets = json.load(open("../../datasets/Twibot-22/" + name, 'r'))
#     for each in user_tweets:
#         uid = 'u' + str(each['author_id'])
#         text = each['text']
#         try:
#             index = uid_index[uid]
#             id_tweet[index].append(text)
#         except KeyError:
#             continue
# json.dump(id_tweet, open('./Twibot-22_original/id_tweet.json', 'w'))