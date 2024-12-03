import numpy as np
import csv
import string

# 构建词汇表
def build_vocab(word_count_threshold, unk_required=False):
    # 假设你的CSV文件路径如下
    train_file = "dataset/train_data.csv"

    all_captions = []
    word_counts = {}
    
    # 定义一个翻译表，去掉所有标点符号
    table = str.maketrans("", "", string.punctuation)
    
    # 读取训练集文件
    with open(train_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            caption = row["captions"]
            caption = caption.translate(table)  # 去除标点符号
            caption = "<BOS> " + caption + " <EOS>"
            all_captions.append(caption)
            for word in caption.split(" "):
                word_counts[word] = word_counts.get(word, 0) + 1

    # 根据词频阈值过滤词汇
    for word in list(word_counts):
        if word_counts[word] < word_count_threshold:
            del word_counts[word]
            unk_required = True

    return word_counts, unk_required



def word_to_ids(word_counts, unk_requried):
    word_to_id = {}
    id_to_word = {}
    count = 0
    if unk_requried:
        word_to_id["<UNK>"] = count
        id_to_word[count] = "<UNK>"
        count += 1
        print("<UNK> True")
    for word in word_counts:
        word_to_id[word] = count
        id_to_word[count] = word
        count += 1
    return word_to_id, id_to_word


# convert each word of captions to the index
def convert_caption(captions, word_to_id, max_length):
    '''将文本描述（captions）转换为对应的数字索引，并为每个句子生成一个掩码（cap_mask）'''
    if type(captions) == "str": 
        captions = [captions]
    caps, cap_mask = [], []
    for cap in captions:
        nWord = len(cap.split(" "))
        # 如果句子长度不足 max_length，添加 <EOS> 填充
        cap = cap + " <EOS>" * (max_length - nWord)
        # 创建掩码：实际单词是 1，填充部分是 0
        cap_mask.append([1.0] * nWord + [0.0] * (max_length - nWord))
        cap_ids = []
        # 对句子中的每个单词进行处理
        for word in cap.split(" "):
            # 如果单词在词汇表中，获取其对应的索引
            if word in word_to_id:
                cap_ids.append(word_to_id[word])
            # 如果单词不在词汇表中，则使用 <UNK> 索引    
            else:
                cap_ids.append(word_to_id["<UNK>"])
        caps.append(cap_ids)
    return np.array(caps), np.array(cap_mask)
