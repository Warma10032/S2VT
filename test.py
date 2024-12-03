from module import *
from utils import *

vovab_size = len(word_counts)
BATCH_SIZE = 10

if __name__ == "__main__":
    s2vt = S2VT(vocab_size=vovab_size, batch_size=BATCH_SIZE)
    s2vt = s2vt.cuda()
    s2vt.load_state_dict(
        torch.load("checkpoints/s2vt_v8.pkl")
    )
    s2vt.eval()
    test_captions = {}
    for idx in range(int(50/BATCH_SIZE)):
        video, video_id = fetch_test_data(idx * BATCH_SIZE, BATCH_SIZE)
        video = torch.FloatTensor(video).cuda()

        cap_out = s2vt(video)

        captions = []
        for tensor in cap_out:
            captions.append(tensor.tolist())
        # size of captions : [79, batch_size]

        # transform captions to [batch_size, 79]
        captions = [[row[i] for row in captions] for i in range(len(captions[0]))]

        predicted_caption = captions_to_english(captions)
        for i in range(BATCH_SIZE):
            test_captions[video_id[i]] = predicted_caption[i]
            
    print("............................\nPredicted Caption:\n")
    print(test_captions)
    
    output_file = "output/test_captions_v8.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = ['video_id', 'row_id','caption']  # 定义表头
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 写入每一行
        for i, (video_id, caption) in enumerate(test_captions.items()):
            writer.writerow({'video_id': video_id, 'row_id': i, 'caption': caption})
    
        
