from data_process import *
import numpy as np
import csv
import ast
import re

def convert_to_list(caption_str):
    try:
        # 使用 ast.literal_eval 安全地将字符串转换为 Python 列表
        caption_list = ast.literal_eval(caption_str)

        # 确保转换结果是列表
        if isinstance(caption_list, list):
            # 去除列表元素中的标点符号
            clean_caption_list = []
            for sentence in caption_list:
                # 使用正则表达式去除标点符号
                clean_sentence = re.sub(
                    r"[^\w\s]", "", sentence
                )  # 去除所有非字母数字和空格的字符
                clean_caption_list.append(clean_sentence.strip())  # 去除多余空格
            return clean_caption_list
        else:
            raise ValueError("The string does not represent a list.")
    except Exception as e:
        print(f"Error converting string to list: {e}")
        return []


n_lstm_steps = 80
FEATURE_DIR = "dataset/features/"
train_csv = "dataset/train_data.csv"
test_cav = "dataset/test.csv"

train_video = {}
test_video = {}
train_video_id = []
test_video_id = []

with open(train_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        caption = row["captions"]
        caption = convert_to_list(caption)
        video = row["video_id"]
        train_video[video] = caption
        train_video_id.append(video)

with open(test_cav, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video = row["video_id"]
        test_video_id.append(video)

word_counts, unk_required = build_vocab(word_count_threshold=0)
word2id, id2word = word_to_ids(word_counts, unk_requried=unk_required)


# fetch features of train videos
def fetch_train_data(batch_size):
    video = np.random.choice(train_video_id, batch_size)
    cur_vid = np.array([np.load(FEATURE_DIR + vid + ".npy") for vid in video])
    feats_idx = np.linspace(0, 79, n_lstm_steps).astype(int)
    cur_vid = cur_vid[:, feats_idx, :]
    captions = [np.random.choice(train_video[vid], 1)[0] for vid in video]
    captions, cap_mask = convert_caption(captions, word2id, n_lstm_steps)
    return cur_vid, captions, cap_mask


def fetch_test_data(idx, batch_size):
    video = test_video_id[idx : idx + batch_size]
    cur_vid = np.array([np.load(FEATURE_DIR + vid + ".npy") for vid in video])
    feats_idx = np.linspace(0, 79, n_lstm_steps).astype(int)
    cur_vid = cur_vid[:, feats_idx, :]
    return cur_vid, video


# print captions
def captions_to_english(captions):
    captions_english = [[id2word[word] for word in caption] for caption in captions]
    real_captions = []
    for cap in captions_english:
        if "<EOS>" in cap:
            cap = cap[0 : cap.index("<EOS>")] # 截取到 <EOS> 之前的部分（去除 <EOS>）
        real_captions.append(" ".join(cap))
    return real_captions
