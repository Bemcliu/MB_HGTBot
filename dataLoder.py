import torch
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline


class Twibot22_subnet_split(Dataset):
    def __init__(self, root='./data_preprocess/abortion_50/', device='cpu', process=False, save=False):
        self.root = root
        self.device = device
        self.process = process

        if process:
            df_train = pd.read_json('../datasets/abortion_1216/random/train.json')
            df_test = pd.read_json('../datasets/abortion_1216/random/test.json')
            df_val = pd.read_json('../datasets/abortion_1216/random/val.json')
            df_support = pd.read_json('../datasets/abortion_1216/random/support.json')

            df_train = df_train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
            df_test = df_test.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
            df_val = df_val.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
            df_support = df_support.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

            self.df_data_labeled = pd.concat([df_train, df_val, df_test], ignore_index=True)
            self.df_data = pd.concat([df_train, df_val, df_test, df_support], ignore_index=True)
            self.df_data = self.df_data
            self.df_data_labeled = self.df_data_labeled
            self.save = save

    def load_labels(self):
        path = self.root + 'label.pt'
        if not os.path.exists(path):
            labels = torch.LongTensor(self.df_data_labeled['label']).to(self.device)
            if self.save:
                torch.save(labels, 'abortion_1216/label.pt')
        else:
            labels = torch.load(self.root + "label.pt", weights_only=True).to(self.device)
        return labels

    def tweets_preprocess(self):
        path = self.root + 'tweets.npy'

        if not os.path.exists(path):
            tweets = []
            for i in tqdm(range(self.df_data.shape[0])):
                one_usr_tweets = []
                user_comments = self.df_data['comments'][i]

                if user_comments is None or len(user_comments) == 0:
                    one_usr_tweets.append('')
                else:
                    for comment in user_comments:
                        one_usr_tweets.append(comment.get('content', ''))

                tweets.append(one_usr_tweets)

            tweets = np.array(tweets, dtype=object)
            if self.save:
                np.save(path, tweets)
        else:
            tweets = np.load(path, allow_pickle=True)

        return tweets

    def tweets_embedding(self):
        global total_word_tensor, total_each_person_tweets

        path = self.root + "tweets_tensor.pt"
        if not os.path.exists(path):
            tweets = np.load("abortion_1216/tweets.npy", allow_pickle=True)
            feature_extract = pipeline(
                'feature-extraction',
                model='roberta-base',
                tokenizer='roberta-base',
                device=0,
                padding=True,
                truncation=True,
                max_length=500,
                add_special_tokens=True
            )

            tweets_list = []

            for each_person_tweets in tqdm(tweets):
                for j, each_tweet in enumerate(each_person_tweets):
                    each_tweet_tensor = torch.tensor(feature_extract(each_tweet))

                    for k, each_word_tensor in enumerate(each_tweet_tensor[0]):
                        if k == 0:
                            total_word_tensor = each_word_tensor
                        else:
                            total_word_tensor += each_word_tensor

                    total_word_tensor /= each_tweet_tensor.shape[1]

                    if j == 0:
                        total_each_person_tweets = total_word_tensor
                    else:
                        total_each_person_tweets += total_word_tensor

                total_each_person_tweets /= len(each_person_tweets)
                tweets_list.append(total_each_person_tweets)

            tweets_tensor = torch.stack(tweets_list).to(self.device)
            if self.save:
                torch.save(tweets_tensor, path)
        else:
            tweets_tensor = torch.load(self.root + "tweets_tensor.pt", weights_only=True).to(self.device)

        return tweets_tensor

    def num_prop_preprocess(self):
        path0 = self.root + 'num_properties_tensor.pt'
        if not os.path.exists(path0):
            path = self.root
            if not os.path.exists(path + "followers_count.pt"):
                followers_count = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['followers_count'][i] is None:
                        followers_count.append(0)
                    else:
                        followers_count.append(self.df_data['followers_count'][i])
                followers_count = torch.tensor(np.array(followers_count, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(followers_count, path + "followers_count.pt")

                friends_count = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['friends_count'][i] is None:
                        friends_count.append(0)
                    else:
                        friends_count.append(self.df_data['friends_count'][i])
                friends_count = torch.tensor(np.array(friends_count, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(friends_count, path + 'friends_count.pt')

                screen_name_length = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['screen_name'][i] is None:
                        screen_name_length.append(0)
                    else:
                        screen_name_length.append(len(self.df_data['screen_name'][i]))
                screen_name_length = torch.tensor(np.array(screen_name_length, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(screen_name_length, path + 'screen_name_length.pt')

                favourites_count = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['favourites_count'][i] is None:
                        favourites_count.append(0)
                    else:
                        favourites_count.append(self.df_data['favourites_count'][i])
                favourites_count = torch.tensor(np.array(favourites_count, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(favourites_count, path + 'favourites_count.pt')

                media_count = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['media_count'][i] is None:
                        media_count.append(0)
                    else:
                        media_count.append(self.df_data['media_count'][i])
                media_count = torch.tensor(np.array(media_count, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(media_count, path + 'media_count.pt')

                location = []
                for i in range(self.df_data.shape[0]):
                    if self.df_data['location'][i] == "":
                        location.append(0)
                    else:
                        location.append(1)
                location = torch.tensor(np.array(location, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(location, path + 'location.pt')

            else:
                screen_name_length = torch.load(path + "screen_name_length.pt")
                favourites_count = torch.load(path + "favourites_count.pt")
                followers_count = torch.load(path + "followers_count.pt")
                friends_count = torch.load(path + "friends_count.pt")
                media_count = torch.load(path + "media_count.pt")
                location = torch.load((path + "location.pt"))

            screen_name_length = pd.Series(screen_name_length.to('cpu').detach().numpy())
            screen_name_length_days = (screen_name_length - screen_name_length.mean()) / screen_name_length.std()
            screen_name_length_days = torch.tensor(np.array(screen_name_length_days))

            favourites_count = pd.Series(favourites_count.to('cpu').detach().numpy())
            favourites_count = (favourites_count - favourites_count.mean()) / favourites_count.std()
            favourites_count = torch.tensor(np.array(favourites_count))

            followers_count = pd.Series(followers_count.to('cpu').detach().numpy())
            followers_count = (followers_count - followers_count.mean()) / followers_count.std()
            followers_count = torch.tensor(np.array(followers_count))

            friends_count = pd.Series(friends_count.to('cpu').detach().numpy())
            friends_count = (friends_count - friends_count.mean()) / friends_count.std()
            friends_count = torch.tensor(np.array(friends_count))

            media_count = pd.Series(media_count.to('cpu').detach().numpy())
            media_count = (media_count - media_count.mean()) / media_count.std()
            media_count = torch.tensor(np.array(media_count))

            num_prop = torch.cat(
                (
                    followers_count.reshape([13650, 1]),
                    friends_count.reshape([13650, 1]),
                    favourites_count.reshape([13650, 1]),
                    media_count.reshape([13650, 1]),
                    screen_name_length_days.reshape([13650, 1]),
                    location.reshape([13650, 1])
                ),
                1
            ).to(self.device)

            if self.save:
                torch.save(num_prop, "abortion_1216/num_properties_tensor.pt")
        else:
            num_prop = torch.load(self.root + "num_properties_tensor.pt", weights_only=True).to(self.device)

        return num_prop

    def flatten_and_average(self, matrix):
        result = []
        for sublist in matrix:
            if len(sublist) > 0:
                avg_value = sum(sublist) // len(sublist)
                result.append(avg_value)
            else:
                result.append(0)
        return result

    def convert_str_to_int(self, matrix):
        result = []
        for sublist in matrix:
            int_sublist = [int(item) if item is not None else 0 for item in sublist]
            result.append(int_sublist)
        return result

    def comments_prop_preprocess(self):
        path0 = self.root + 'com_properties_tensor.pt'
        if not os.path.exists(path0):
            path = self.root
            if not os.path.exists(path + "bookmark_count.pt"):
                bookmark_count = []
                favorite_count = []
                quote_count = []
                reply_count = []
                retweet_count = []
                view_count = []
                # time_span=[]

                for i in tqdm(range(self.df_data.shape[0])):
                    bookmark_count_oneuser = []
                    favorite_count_oneuser = []
                    quote_count_oneuser = []
                    reply_count_oneuser = []
                    retweet_count_oneuser = []
                    view_count_oneuser = []
                    # time_span_oneuser = []

                    user_comments = self.df_data['comments'][i]

                    if user_comments is None or len(user_comments) == 0:
                        bookmark_count_oneuser.append('')
                        favorite_count_oneuser.append('')
                        quote_count_oneuser.append('')
                        reply_count_oneuser.append('')
                        retweet_count_oneuser.append('')
                        view_count_oneuser.append('')
                        # time_span_oneuser.append('')
                    else:
                        for comment in user_comments:
                            bookmark_count_oneuser.append(comment.get('bookmark_count', ''))
                            favorite_count_oneuser.append(comment.get('favorite_count', ''))
                            quote_count_oneuser.append(comment.get('quote_count', ''))
                            reply_count_oneuser.append(comment.get('reply_count', ''))
                            retweet_count_oneuser.append(comment.get('retweet_count', ''))
                            view_count_oneuser.append(comment.get('view_count', ''))
                            # time_span_oneuser.append(comment.get('time_span', ''))

                    bookmark_count.append(bookmark_count_oneuser)
                    favorite_count.append(favorite_count_oneuser)
                    quote_count.append(quote_count_oneuser)
                    reply_count.append(reply_count_oneuser)
                    retweet_count.append(retweet_count_oneuser)
                    view_count.append(view_count_oneuser)
                    # time_span.append(time_span_oneuser)

                bookmark_count = self.flatten_and_average(bookmark_count)
                favorite_count = self.flatten_and_average(favorite_count)
                quote_count = self.flatten_and_average(quote_count)
                reply_count = self.flatten_and_average(reply_count)
                retweet_count = self.flatten_and_average(retweet_count)
                view_count = self.flatten_and_average(self.convert_str_to_int(view_count))
                # time_span = self.flatten_and_average(time_span)

                bookmark_count = torch.tensor(np.array(bookmark_count, dtype=np.float32)).to(self.device)
                favorite_count = torch.tensor(np.array(favorite_count, dtype=np.float32)).to(self.device)
                quote_count = torch.tensor(np.array(quote_count, dtype=np.float32)).to(self.device)
                reply_count = torch.tensor(np.array(reply_count, dtype=np.float32)).to(self.device)
                retweet_count = torch.tensor(np.array(retweet_count, dtype=np.float32)).to(self.device)
                view_count = torch.tensor(np.array(view_count, dtype=np.float32)).to(self.device)
                # time_span = torch.tensor(np.array(time_span, dtype=np.float32)).to(self.device)
                if self.save:
                    torch.save(bookmark_count, path + "bookmark_count.pt")
                    torch.save(favorite_count, path + "favorite_count.pt")
                    torch.save(quote_count, path + "quote_count.pt")
                    torch.save(reply_count, path + "reply_count.pt")
                    torch.save(retweet_count, path + "retweet_count.pt")
                    torch.save(view_count, path + "view_count.pt")
                    # torch.save(time_span, path + "time_span.pt")
            else:
                bookmark_count = torch.load(path + "bookmark_count.pt")
                favorite_count = torch.load(path + "favorite_count.pt")
                quote_count = torch.load(path + "quote_count.pt")
                reply_count = torch.load(path + "reply_count.pt")
                retweet_count = torch.load(path + "retweet_count.pt")
                view_count = torch.load((path + "view_count.pt"))
                # time_span = torch.load((path + "time_span.pt"))

            bookmark_count = pd.Series(bookmark_count.to('cpu').detach().numpy())
            bookmark_count = (bookmark_count - bookmark_count.mean()) / bookmark_count.std()
            bookmark_count = torch.tensor(np.array(bookmark_count))

            favorite_count = pd.Series(favorite_count.to('cpu').detach().numpy())
            favorite_count = (favorite_count - favorite_count.mean()) / favorite_count.std()
            favorite_count = torch.tensor(np.array(favorite_count))

            quote_count = pd.Series(quote_count.to('cpu').detach().numpy())
            quote_count = (quote_count - quote_count.mean()) / quote_count.std()
            quote_count = torch.tensor(np.array(quote_count))

            reply_count = pd.Series(reply_count.to('cpu').detach().numpy())
            reply_count = (reply_count - reply_count.mean()) / reply_count.std()
            reply_count = torch.tensor(np.array(reply_count))

            retweet_count = pd.Series(retweet_count.to('cpu').detach().numpy())
            retweet_count = (retweet_count - retweet_count.mean()) / retweet_count.std()
            retweet_count = torch.tensor(np.array(retweet_count))

            view_count = pd.Series(view_count.to('cpu').detach().numpy())
            view_count = (view_count - view_count.mean()) / view_count.std()
            view_count = torch.tensor(np.array(view_count))


            com_prop = torch.cat(
                (
                    bookmark_count.reshape([13650, 1]),
                    favorite_count.reshape([13650, 1]),
                    quote_count.reshape([13650, 1]),
                    reply_count.reshape([13650, 1]),
                    retweet_count.reshape([13650, 1]),
                    view_count.reshape([13650, 1])
                ),
                1
            ).to(self.device)

            if self.save:
                torch.save(com_prop, "abortion_1216/com_properties_tensor.pt")
        else:
            com_prop = torch.load(self.root + "com_properties_tensor.pt", weights_only=True).to(self.device)

        return com_prop

    def Build_Graph(self):
        path = self.root + 'edge_index.pt'
        if not os.path.exists(path):
            id2index_dict = {id: index for index, id in enumerate(self.df_data['userid'])}

            edge_index = []
            edge_type = []

            for i, relation in enumerate(self.df_data['interaction']):
                if relation is not None:
                    for each_id in relation['following_ids']:
                        try:
                            target_id = id2index_dict[int(each_id)]
                        except KeyError:
                            continue
                        else:
                            edge_index.append([i, target_id])
                        edge_type.append(0)
                    for each_id in relation['follower_ids']:
                        try:
                            target_id = id2index_dict[int(each_id)]
                        except KeyError:
                            continue
                        else:
                            edge_index.append([i, target_id])
                        edge_type.append(1)
                else:
                    continue

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
            edge_type = torch.tensor(edge_type, dtype=torch.long).to(self.device)
            if self.save:
                torch.save(edge_index, self.root + "edge_index.pt")
                torch.save(edge_type, self.root + "edge_type.pt")
        else:
            edge_index = torch.load(self.root + "edge_index.pt", weights_only=True).to(self.device)
            edge_type = torch.load(self.root + "edge_type.pt", weights_only=True).to(self.device)

        return edge_index, edge_type

    def train_val_test_mask(self):
        sample_number = len(self.load_labels())
        train_idx = range(int(0.7 * sample_number))
        val_idx = range(int(0.7 * sample_number), int(0.9 * sample_number))
        test_idx = range(int(0.9 * sample_number), int(sample_number))

        return train_idx, val_idx, test_idx

    def dataloader(self):
        labels = self.load_labels()
        if self.process:
            self.tweets_preprocess()

        tweets_tensor = self.tweets_embedding()
        num_prop = self.num_prop_preprocess()
        comments_prop = self.comments_prop_preprocess()

        edge_index, edge_type = self.Build_Graph()
        train_idx, val_idx, test_idx = self.train_val_test_mask()
        return tweets_tensor, num_prop, comments_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx