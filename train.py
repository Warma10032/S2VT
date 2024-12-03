from module import *
import torch
from torch import nn
from utils import *


EPOCH = 30
nIter = 50
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
vocab_size = len(word_counts)
pkl_file = None

# save training log
def write_txt(epoch, iteration, loss):
    with open("log/train_log.txt", 'a+') as f:
        f.write("Epoch:[ %d ]\t Iteration:[ %d ]\t loss:[ %f ]\n" % (epoch, iteration, loss))


if __name__ == "__main__":
    # 初始化S2VT模型
    s2vt = S2VT(vocab_size=vocab_size, batch_size=BATCH_SIZE)
    if pkl_file:
        s2vt.load_state_dict(torch.load(pkl_file))
    s2vt = s2vt.cuda()
    # 初始化损失函数和优化器
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(s2vt.parameters(), lr=LEARNING_RATE)

    # 开始训练
    for epoch in range(EPOCH):
        for i in range(nIter):
            video, caption, cap_mask = fetch_train_data(BATCH_SIZE)
            video, caption, cap_mask = torch.FloatTensor(video).cuda(), torch.LongTensor(caption).cuda(), \
                                       torch.FloatTensor(cap_mask).cuda()

            cap_out = s2vt(video, caption)
            cap_labels = caption[:, 1:].contiguous().view(-1)       # size [batch_size, 79]
            cap_mask = cap_mask[:, 1:].contiguous().view(-1)        # size [batch_size, 79]

            # 计算损失
            logit_loss = loss_func(cap_out, cap_labels)
            masked_loss = logit_loss*cap_mask
            loss = torch.sum(masked_loss)/torch.sum(cap_mask)

            # 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%20 == 0:
                # print("Epoch: %d  iteration: %d , loss: %f" % (epoch, i, loss))
                write_txt(epoch, i, loss)
                
        if (epoch+1)%1 == 0:
            torch.save(s2vt.state_dict(), f"checkpoints/s2vt_v{epoch}.pkl")
            print("Epoch: %d iter: save successed!" % (epoch))


