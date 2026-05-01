import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn.functional as F
import re
from PIL import UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score,classification_report
from tqdm import tqdm
from modelbart import Similarity, DetectionModule,Attention_Encoder,Reason_Similarity,Aggregator,BertEncoder
from sklearn import metrics
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import re
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
# Configs
# DEVICE = "cuda:1"
NUM_WORKER = 20
BATCH_SIZE = 32
LR = 3e-5
L2 = 1e-5
NUM_EPOCH = 50

torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#print(torch.__version__)

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def text_preprocessing(text):
    """
    - 删除实体@符号(如。“@united”)
    — 纠正错误(如:'&amp;' '&')
    @参数 text (str):要处理的字符串
    @返回 text (Str):已处理的字符串
    """
   
    text = re.sub(r'(@.*?)[\s]', ' ', text)

 
    text = re.sub(r'&amp;', '&', text)

 
    text = re.sub(r'\s+', ' ', text).strip()
    

    return text

def extract_quoted_text(text):
    
    
    sentences_to_remove = [
        
        "agree reasoning",
        "disagree reasoning",
        "real news",
        "fake news",
        "fake News",
        "real News",
        "credibility score",
        "increase",
        "decrease"
    ]
    sentences_to_remove2 = [
        "Here's another that can the of the :",
        "Here's a that can the of the :",
        "Here is a that can the of the :",
        "Here's another attempt at generating a that can the of the :",
        "Here's a new that can the of the :",
        "Here's a revised that can the of the :",
        "Here's a rewritten that can the of the :",
        "Here's a possible that can the of the :"
    ]
    for sentence in sentences_to_remove:
        pattern = re.escape(sentence)+r'\s*'
        text = re.sub(pattern,'',text)

    for sentence2 in sentences_to_remove2:
        pattern = re.escape(sentence2)+r'\s*'

    print('处理后：',text)
    return text

def tokenize_and_numericalize_data(text, tokenizer):
  
    tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=100)

    return tokenized['input_ids']

class FakeNewsDataset(Dataset):

    def __init__(self, df, tokenizer, MAX_LEN):
        
        self.csv_data = df
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return self.csv_data.shape[0]

    def __getitem__(self, idx):
        try:

            text = self.csv_data['title'][idx]
            pos = self.csv_data['agree_reason'][idx]
            neg = self.csv_data['disagree_reason'][idx]
            

            content_input_id = tokenize_and_numericalize_data(text,self.tokenizer)
            pos_input_id = tokenize_and_numericalize_data(pos,self.tokenizer)
            neg_input_id = tokenize_and_numericalize_data(neg,self.tokenizer)

            
            agree_score = self.csv_data['agree_score'][idx]  
            disagree_score = self.csv_data['disagree_score'][idx]  
            

        except (ValueError):
           
            return None
        label = self.csv_data['label'][idx]

        label = int(label)

        label = torch.tensor(label)
        agree_soft_label = torch.tensor(float(self.csv_data['agree_score'][idx])/100, dtype=torch.float32)
        disagree_soft_label = torch.tensor(float(self.csv_data['disagree_score'][idx])/100, dtype=torch.float32)
        sample = {
            'content': torch.tensor(content_input_id),
            'pos_reason': torch.tensor(pos_input_id),
            'neg_reason': torch.tensor(neg_input_id),
            'label': label,
            'agree_soft_label': agree_soft_label,
            'disagree_soft_label': disagree_soft_label
        }

        return sample
    

# Import thêm hàm chia dataset từ sklearn
from sklearn.model_selection import train_test_split

# Đọc file dữ liệu tổng đã được sinh reasoning từ thuật toán SORG
# Lưu ý: Thay đổi tên file 'sorg_output.csv' cho khớp với tên file bạn đã lưu ở bước trước
df_total = pd.read_csv("sorg_gpt3.5_output.csv") 

# Thực hiện phân chia tỷ lệ 80% Train và 20% Test
df_train, df_test = train_test_split(df_total, test_size=0.2, random_state=42)

# Reset lại index của Dataframe để tránh lỗi KeyError khi DataLoader gọi các batch dữ liệu
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

MAX_LEN = 256
def collate_fn(batch):
    
        content = torch.stack([item['content'] for item in batch if item is not None])
        
        pos_reason = torch.stack([item['pos_reason'] for item in batch if item is not None])
        neg_reason = torch.stack([item['neg_reason'] for item in batch if item is not None])
        labels = torch.stack([item['label'] for item in batch if item is not None])
        
        agree_soft_labels = torch.stack([item['agree_soft_label'] for item in batch if item is not None])
        disagree_soft_labels = torch.stack([item['disagree_soft_label'] for item in batch if item is not None])
        return {'content': content, 'pos_reason': pos_reason, 'neg_reason': neg_reason, 'label': labels, 'agree_soft_label': agree_soft_labels,
        'disagree_soft_label': disagree_soft_labels}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
dataset_train = FakeNewsDataset(df_train, tokenizer, MAX_LEN)
dataset_val = FakeNewsDataset(df_test, tokenizer, MAX_LEN)
train_dataloader = DataLoader(dataset_train, batch_size=16,
                        shuffle=True, num_workers=4,collate_fn=collate_fn,drop_last=True)
val_dataloader = DataLoader(dataset_val, batch_size=16,
                        shuffle=True, num_workers=4,collate_fn=collate_fn,drop_last=True)

import csv
csv_file = 'S-teacher-bert.csv'
with open(csv_file,mode='w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch','loss','Accuracy','AUC','macro_f1','mic_f1','fake_f1','fake_precision','fake_recall','true_f1','true_precision','true_recall'])



def train():
   
    
    lr = LR 
    l2 = L2  
    num_epoch = NUM_EPOCH  
    
    bert = BertEncoder(256,False)
    bert.to(device)   
    optim_bert = torch.optim.Adam(bert.parameters(),lr=lr,weight_decay=l2)  
    bert2 = BertEncoder(256,False)
    bert2.to(device)  
    optim_bert2 = torch.optim.Adam(bert2.parameters(),lr=lr,weight_decay=l2)
    bert3 = BertEncoder(256,False)
    bert3.to(device)  
    optim_bert3 = torch.optim.Adam(bert3.parameters(),lr=lr,weight_decay=l2)
    attention = Attention_Encoder() 
    attention.to(device)
    optim_task_attention = torch.optim.Adam(
        attention.parameters(), lr=lr, weight_decay=l2
    )  
   
    R2T_usefulness = Similarity()  
    R2T_usefulness.to(device)
    optim_task_R2T = torch.optim.Adam(
        R2T_usefulness.parameters(), lr=lr, weight_decay=l2
    )
    T2R_usefulness = Similarity()  
    T2R_usefulness.to(device)
    optim_task_T2R = torch.optim.Adam(
        T2R_usefulness.parameters(), lr=lr, weight_decay=l2
    )  
    Reason_usefulness = Reason_Similarity()
    Reason_usefulness.to(device)
    optim_task_reason = torch.optim.Adam(
        Reason_usefulness.parameters(), lr=lr, weight_decay=l2
    )
    aggregator = Aggregator()
    aggregator.to(device)
    optim_task_aggregator = torch.optim.Adam(
        aggregator.parameters(), lr=lr, weight_decay=l2
    )
    detection_module = DetectionModule()  
    detection_module.to(device)
    loss_func_similarity = torch.nn.CosineEmbeddingLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()
    optim_task_detection = torch.optim.Adam(
        detection_module.parameters(), lr=lr, weight_decay=l2
    )  

    loss_R2T_list = []
    loss_T2R_list = []
    loss_reason_list = []
    loss_detection_list = []
    
    loss_R2T_total = 0
    loss_detection_total = 0
    best_acc = 0
    best_loss = 1000
    acc_best_train = 0
    
    save_path = 'best_teachermodel.pth'
    
    for epoch in range(num_epoch):
        attention.train()
        R2T_usefulness.train()
        T2R_usefulness.train()
        detection_module.train()
        Reason_usefulness.train()

        corrects_pre_R2T = 0
        corrects_pre_T2R = 0
        corrects_pre_reason = 0
        corrects_pre_detection = 0
        loss_R2T_total = 0
        loss_T2R_total = 0
        loss_R_total = 0
        loss_detection_total = 0
        R2T_count = 0
        T2R_count = 0
        R_count = 0
        detection_count = 0

        epoch_loss_R2T = 0.0
        epoch_loss_T2R = 0.0
        epoch_loss_reason = 0.0
        epoch_loss_detection = 0.0
        num_batches = 0
        train_outputs = []
        train_labels = []
        
        for batch in train_dataloader:
            
            news_content = batch['content'].to(device)  
            pos =  batch["pos_reason"].to(device)
            neg = batch['neg_reason'].to(device)
            label = batch['label'].to(device)
            agree_soft_label = batch['agree_soft_label'].to(device)
            disagree_soft_label = batch['disagree_soft_label'].to(device)
            
            content = bert(news_content)
            positive = bert2(pos)
            negative = bert3(neg)
            
            # --- cross attention ---
            pos_reason2text, pos_text2reason, positive,neg_reason2text, neg_text2reason, negative = attention(content, positive, negative)
            

            
            # --- task1 R2T  loss ---
            
            text_aligned_match1, R2T_match, pred_R2T_match = R2T_usefulness(content, pos_reason2text)
            text_aligned_unmatch1, R2T_unmatch, pred_R2T_unmatch = R2T_usefulness(content, neg_reason2text)
            R2T_pred = torch.cat([pred_R2T_match.argmax(1), pred_R2T_unmatch.argmax(1)], dim=0)
            R2Tlabel_0 = torch.cat([agree_soft_label, disagree_soft_label], dim=0)  
            R2Tlabel_1 = torch.cat([agree_soft_label, -1*disagree_soft_label], dim=0).to(device)
            text_aligned_4_task1 = torch.cat([text_aligned_match1, text_aligned_unmatch1], dim=0)
            R2T_aligned_4_task1 = torch.cat([R2T_match, R2T_unmatch], dim=0)
            loss_R2T = loss_func_similarity(text_aligned_4_task1, R2T_aligned_4_task1, R2Tlabel_1)
            loss_R2T = (loss_R2T*R2Tlabel_0).mean()

            # --- task2 T2R loss ---
            
            text_aligned_match2, T2R_match, pred_T2R_match = T2R_usefulness(content, pos_text2reason)
            text_aligned_unmatch2, T2R_unmatch, pred_T2R_unmatch = T2R_usefulness(content, neg_text2reason)
            T2R_pred = torch.cat([pred_T2R_match.argmax(1), pred_T2R_unmatch.argmax(1)], dim=0)
            T2R_label_0 = torch.cat([agree_soft_label, disagree_soft_label], dim=0)
            T2R_label_1 = torch.cat([agree_soft_label, -1*disagree_soft_label], dim=0)
            text_aligned_4_task2 = torch.cat([text_aligned_match2, text_aligned_unmatch2], dim=0)
            T2R_aligned_4_task2 = torch.cat([T2R_match, T2R_unmatch], dim=0)
            loss_T2R = loss_func_similarity(text_aligned_4_task2, T2R_aligned_4_task2, T2R_label_1)
            loss_T2R = (loss_T2R*T2R_label_0).mean()

            # --- task3 Reason loss ---
            text_aligned_match3, reason_match, pred_reason_match = Reason_usefulness(content, positive)
            text_aligned_unmatch3, reason_unmatch, pred_reason_unmatch = Reason_usefulness(content, negative)
            reason_pred = torch.cat([pred_reason_match.argmax(1), pred_reason_unmatch.argmax(1)], dim=0)
            reason_label_0 = torch.cat([agree_soft_label, disagree_soft_label], dim=0)
            reason_label_1 = torch.cat([agree_soft_label, -1*disagree_soft_label], dim=0)
            text_aligned_4_task3 = torch.cat([text_aligned_match3, text_aligned_unmatch3], dim=0)
            reason_aligned_4_task3 = torch.cat([reason_match, reason_unmatch], dim=0)
            reason_target = torch.where(reason_label_0>=0.5,1,-1).to(device)
            loss_reason = loss_func_similarity(text_aligned_4_task3, reason_aligned_4_task3, reason_label_1)
            loss_reason = (loss_reason*reason_label_0).mean()

            # --- TASK fake news Detection ---
            text_R2T_aligned_agr, R2T_aligned_agr, _ = R2T_usefulness(content, pos_reason2text)
            text_T2R_aligned_agr, T2R_aligned_agr, _ = T2R_usefulness(content, pos_text2reason)
            text_R_aligned_agr, R_aligned_agr, _ = Reason_usefulness(content, positive)
            text_R2T_aligned_dis, R2T_aligned_dis, _ = R2T_usefulness(content, neg_reason2text)
            text_T2R_aligned_dis, T2R_aligned_dis, _ = T2R_usefulness(content, neg_text2reason)
            text_R_aligned_dis, R_aligned_dis, _ = Reason_usefulness(content, negative)
            final_feature = aggregator(content,R2T_aligned_agr,T2R_aligned_agr,R_aligned_agr,R2T_aligned_dis,T2R_aligned_dis,R_aligned_dis)
            
            pre_detection = detection_module(final_feature)
            loss_detection = loss_func_detection(pre_detection, label)
            pre_label_detection = pre_detection.argmax(1)
            train_outputs.append(final_feature.detach().cpu().numpy())
            train_labels.append(label.detach().cpu().numpy())

          
            optim_task_attention.zero_grad()
            optim_task_R2T.zero_grad()
            optim_task_T2R.zero_grad()
            optim_task_reason.zero_grad()
            optim_task_aggregator.zero_grad()
            optim_task_detection.zero_grad()
            optim_bert.zero_grad()
            optim_bert2.zero_grad()
            optim_bert3.zero_grad()

            
            loss_R2T.backward(retain_graph=True)
            loss_T2R.backward(retain_graph=True)
            loss_reason.backward(retain_graph=True)
            loss_detection.backward()

            
            optim_task_attention.step()
            optim_task_R2T.step()
            optim_task_T2R.step()
            optim_task_reason.step()
            optim_task_detection.step()
            optim_task_aggregator.step()
            optim_bert3.step()
            optim_bert2.step()
            optim_bert.step()

            binary_R2T_pred = (R2T_pred < 0.5).long()
            binary_R2Tlabel = (R2Tlabel_0 < 0.5).long()
            corrects_pre_R2T += binary_R2T_pred.eq(binary_R2Tlabel).sum().item()


            binary_T2R_pred = (T2R_pred < 0.5).long()
            binary_T2Rlabel = (T2R_label_0 < 0.5).long()
            corrects_pre_T2R += binary_T2R_pred.eq(binary_T2Rlabel).sum().item()


            binary_reason_pred = (reason_pred < 0.5).long()
            binary_reason_label = (reason_label_0 < 0.5).long()
            corrects_pre_T2R += binary_reason_pred.eq(binary_reason_label).sum().item()

            corrects_pre_detection += pre_detection.argmax(1).eq(label.view_as(pre_detection.argmax(1))).sum().item()

            # ---  Record  ---
            loss_R2T_total += loss_R2T.item() * (2* content.shape[0])
            loss_T2R_total += loss_T2R.item() * (2*content.shape[0])
            loss_R_total += loss_reason.item() * (2* content.shape[0])
            loss_detection_total += loss_detection.item() * content.shape[0]
            R2T_count += (2*content.shape[0] *2)
            T2R_count += (2* content.shape[0]*2 )
            R_count += ( 2*content.shape[0] *2)
            detection_count += content.shape[0]

            epoch_loss_R2T += loss_R2T.item()
            epoch_loss_T2R += loss_T2R.item()
            epoch_loss_reason += loss_reason.item()
            epoch_loss_detection += loss_detection.item()
            num_batches += 1

        train_labels = np.concatenate(train_labels,axis=0)
        train_outputs = np.concatenate(train_outputs,axis=0)

        loss_R2T_train = loss_R2T_total / R2T_count
        loss_T2R_train = loss_T2R_total / T2R_count
        loss_R_train = loss_R_total / R_count
        loss_detection_train = loss_detection_total / detection_count
        acc_R2T_train = corrects_pre_R2T / R2T_count
        acc_T2R_train = corrects_pre_T2R / T2R_count
        acc_detection_train = corrects_pre_detection / detection_count

        avg_loss_R2T = epoch_loss_R2T / num_batches
        avg_loss_T2R = epoch_loss_T2R / num_batches
        avg_loss_reason = epoch_loss_reason / num_batches
        avg_loss_detection = epoch_loss_detection / num_batches

        loss_R2T_list.append(avg_loss_R2T)
        loss_T2R_list.append(avg_loss_T2R)
        loss_reason_list.append(avg_loss_reason)
        loss_detection_list.append(avg_loss_detection)

        # ---  Test  ---
        
        acc_R2T_test, acc_detection_test, loss_R2T_test, loss_detection_test, cm_similarity, cm_detection,loss_T2R_test,acc_T2R_test,test_outputs,test_labels= test(bert,bert2,bert3,Reason_usefulness, attention,R2T_usefulness, detection_module,T2R_usefulness, aggregator,val_dataloader,epoch,avg_loss_detection)

        
        
        if acc_detection_test >= best_acc and loss_detection_test <= best_loss:
            best_acc = acc_detection_test
            state_dicts = {
                'bert':bert.state_dict(),
                'bert2':bert2.state_dict(),
                'bert3':bert3.state_dict(),
                'attention': attention.state_dict(),
                'R2T_usefulness': R2T_usefulness.state_dict(),
                'T2R_usefulness': T2R_usefulness.state_dict(),
                'Reason_usefulness': Reason_usefulness.state_dict(),
                'aggregator':aggregator.state_dict(),
                'detection_module': detection_module.state_dict(),
            }
            torch.save(state_dicts, save_path)

            print(f'Best model parameters saved to {save_path}')
        
        



def test(bert,bert2,bert3,Reason_usefulness, attention,R2T_usefulness, detection_module,T2R_usefulness, aggregator,test_dataloader,epoch,avg_loss):

    
    bert.eval()
    bert2.eval()
    bert3.eval()
    Reason_usefulness.eval()
    detection_module.eval()
    R2T_usefulness.eval()
    T2R_usefulness.eval()
    attention.eval()
    aggregator.eval()

 
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_func_similarity = torch.nn.CosineEmbeddingLoss()

    R2T_count = 0
    T2R_count = 0
    detection_count = 0
    loss_R2T_total = 0
    loss_T2R_total = 0
    loss_detection_total = 0
    R2Tlabel_all = []
    T2R_label_all = []
    detection_label_all = []
    R2T_pre_label_all = []
    T2R_pre_label_all = []
    detection_pre_label_all = []
    pre_detection_pre_probs_all = []

    all_outputs = []
    all_labels = []

    with torch.no_grad():
     
        for batch in test_dataloader:

            if all(item is None for item in batch) :
                print('error2',batch)
                continue
            news_content = batch['content'].to(device)
            news_content = news_content.long()
            pos =  batch["pos_reason"].to(device).long()
            neg = batch['neg_reason'].to(device).long()
            label = batch['label'].to(device)
            agree_soft_label = batch['agree_soft_label'].to(device)
            disagree_soft_label = batch['disagree_soft_label'].to(device)

            content = bert(news_content)
            positive = bert2(pos)
            negative = bert3(neg)

            # --- cross attention ---
            pos_reason2text, pos_text2reason, positive, neg_reason2text, neg_text2reason, negative = attention(content, positive, negative)

            # --- task1 R2T  loss ---
            text_aligned_match1, R2T_match, pred_R2T_match = R2T_usefulness(content, pos_reason2text)
            text_aligned_unmatch1, R2T_unmatch, pred_R2T_unmatch = R2T_usefulness(content, neg_reason2text)
            R2T_pred = torch.cat([pred_R2T_match.argmax(1),  pred_R2T_unmatch.argmax(1)], dim=0)
            R2Tlabel_0 = torch.cat([agree_soft_label, disagree_soft_label], dim=0)  
            R2Tlabel_1 = torch.cat([agree_soft_label, -1*disagree_soft_label], dim=0)
            text_aligned_4_task1 = torch.cat([text_aligned_match1, text_aligned_unmatch1], dim=0)
            R2T_aligned_4_task1 = torch.cat([R2T_match, R2T_unmatch], dim=0)
            loss_R2T = loss_func_similarity(text_aligned_4_task1, R2T_aligned_4_task1, R2Tlabel_1)
            loss_R2T = (loss_R2T*R2Tlabel_0).mean()

            # --- task2 T2R loss ---
            text_aligned_match2, T2R_match, pred_T2R_match = T2R_usefulness(content, pos_text2reason)
            text_aligned_unmatch2, T2R_unmatch, pred_T2R_unmatch = T2R_usefulness(content, neg_text2reason)
            T2R_pred = torch.cat([pred_T2R_match.argmax(1), pred_T2R_unmatch.argmax(1)], dim=0)
            T2R_label_0 = torch.cat([agree_soft_label, disagree_soft_label], dim=0)
            T2R_label_1 = torch.cat([agree_soft_label, -1*disagree_soft_label], dim=0)
            text_aligned_4_task2 = torch.cat([text_aligned_match2, text_aligned_unmatch2], dim=0)
            T2R_aligned_4_task2 = torch.cat([T2R_match, T2R_unmatch], dim=0)
            loss_T2R = loss_func_similarity(text_aligned_4_task2, T2R_aligned_4_task2, T2R_label_1)
            loss_T2R = (loss_T2R*T2R_label_0).mean()

            # --- task3 Reason loss ---
            text_aligned_match3, reason_match, pred_reason_match = Reason_usefulness(content, positive)
            text_aligned_unmatch3, reason_unmatch, pred_reason_unmatch = Reason_usefulness(content, negative)
            reason_pred = torch.cat([pred_reason_match.argmax(1), pred_reason_unmatch.argmax(1)], dim=0)
            reason_label_0 = torch.cat([agree_soft_label, disagree_soft_label], dim=0)
            reason_label_1 = torch.cat([agree_soft_label, -1*disagree_soft_label], dim=0)
            text_aligned_4_task3 = torch.cat([text_aligned_match3, text_aligned_unmatch3], dim=0)
            reason_aligned_4_task3 = torch.cat([reason_match, reason_unmatch], dim=0)
            loss_reason = loss_func_similarity(text_aligned_4_task3, reason_aligned_4_task3, reason_label_1)
            loss_reason = (loss_reason*reason_label_0).mean()

            # --- TASK fake news Detection ---
            text_R2T_aligned_agr, R2T_aligned_agr, _ = R2T_usefulness(content, pos_reason2text)
            text_T2R_aligned_agr, T2R_aligned_agr, _ = T2R_usefulness(content, pos_text2reason)
            text_R_aligned_agr, R_aligned_agr, _ = Reason_usefulness(content, positive)
            text_R2T_aligned_dis, R2T_aligned_dis, _ = R2T_usefulness(content, neg_reason2text)
            text_T2R_aligned_dis, T2R_aligned_dis, _ = T2R_usefulness(content, neg_text2reason)
            text_R_aligned_dis, R_aligned_dis, _ = Reason_usefulness(content, negative)
            final_feature = aggregator(content,R2T_aligned_agr,T2R_aligned_agr,R_aligned_agr,R2T_aligned_dis,T2R_aligned_dis,R_aligned_dis)
            pre_detection = detection_module(final_feature)
            pre_detection_pre_probs = torch.sigmoid(pre_detection).detach().cpu().numpy()
            pre_detection_pre_probs = pre_detection_pre_probs[:,0]
            loss_detection = loss_func_detection(pre_detection, label)
            pre_label_detection = pre_detection.argmax(1)
            # ---  Record  ---
            all_outputs.append(final_feature.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())

            loss_R2T_total += loss_R2T.item() * (2*content.shape[0])
            loss_T2R_total += loss_T2R.item() * (2*content.shape[0])
            loss_detection_total += loss_detection.item() * content.shape[0]
            R2T_count += (2*content.shape[0] *2)
            T2R_count += (2*content.shape[0] *2)
            detection_count += content.shape[0]

            R2T_pre_label_all.append(R2T_pred.detach().cpu().numpy())
            T2R_pre_label_all.append(T2R_pred.detach().cpu().numpy())
            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            pre_detection_pre_probs_all.append(pre_detection_pre_probs)
            R2Tlabel_all.append(R2Tlabel_0.detach().cpu().numpy())
            T2R_label_all.append(T2R_label_0.detach().cpu().numpy())
            detection_label_all.append(label.detach().cpu().numpy())

        pre_detection_pre_probs_all=np.concatenate(pre_detection_pre_probs_all,axis=0)
        test_labels = np.concatenate(all_labels,axis=0)
        test_outputs = np.concatenate(all_outputs,axis=0)
        print('R2T_count:',R2T_count)
        loss_R2T_test = loss_R2T_total / R2T_count
        loss_T2R_test = loss_T2R_total / T2R_count
        loss_detection_test = loss_detection_total / detection_count

        R2T_pre_label_all = np.concatenate(R2T_pre_label_all, 0)
        T2R_pre_label_all = np.concatenate(T2R_pre_label_all, 0)
        R2Tlabel_all = np.concatenate(R2Tlabel_all, 0)
        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)

        T2R_label_all = np.concatenate(T2R_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)

        R2T_pre_label_all=(R2T_pre_label_all<0.5).astype(int)
        R2Tlabel_all=(R2Tlabel_all<0.5).astype(int)
        T2R_pre_label_all=(T2R_pre_label_all<0.5).astype(int)
        T2R_label_all=(T2R_label_all<0.5).astype(int)


        
        detection_pre_label_all=(detection_pre_label_all<=0.5).astype(int)
        detection_label_all=(detection_label_all<=0.5).astype(int)


        acc_R2T_test = accuracy_score(R2T_pre_label_all, R2Tlabel_all)
        print(acc_R2T_test)
        acc_T2R_test = accuracy_score(T2R_pre_label_all, T2R_label_all)
        acc_detection_test = accuracy_score(detection_pre_label_all, detection_label_all)
        cm_similarity = confusion_matrix(R2T_pre_label_all, R2Tlabel_all)
        cm_detection = confusion_matrix(detection_pre_label_all, detection_label_all)
        acc_detection_test = accuracy_score(detection_pre_label_all, detection_label_all)
        print(f"Test Accuracy: {acc_detection_test}")

        precision_true = precision_score(detection_label_all, detection_pre_label_all, pos_label=1)
        recall_true = recall_score(detection_label_all, detection_pre_label_all, pos_label=1)
        f1_true = f1_score(detection_label_all, detection_pre_label_all, pos_label=1)

        precision_false = precision_score(detection_label_all, detection_pre_label_all, pos_label=0)
        recall_false = recall_score(detection_label_all, detection_pre_label_all, pos_label=0)
        f1_false = f1_score(detection_label_all, detection_pre_label_all, pos_label=0)

        print(f"Precision (True): {precision_true}")
        print(f"Recall (True): {recall_true}")
        print(f"F1 Score (True): {f1_true}")

        print(f"Precision (False): {precision_false}")
        print(f"Recall (False): {recall_false}")
        print(f"F1 Score (False): {f1_false}")

        macro_f1 = f1_score(detection_label_all, detection_pre_label_all, average='macro')
        micro_f1 = f1_score(detection_label_all, detection_pre_label_all, average='micro')

        print(f"Macro F1 Score: {macro_f1}")
        print(f"Micro F1 Score: {micro_f1}")

        
        auc_score = roc_auc_score(detection_label_all, pre_detection_pre_probs_all)
        print(f"AUC Score: {auc_score}")


        report = classification_report(detection_label_all, detection_pre_label_all)
        print(f"Classification Report:\n{report}")
        print(metrics.classification_report(detection_label_all, detection_pre_label_all, digits=4))

        with open(csv_file,mode='a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch,avg_loss,acc_detection_test,auc_score,macro_f1,micro_f1,f1_false,precision_false,recall_false,f1_true,precision_true,recall_true])

    return acc_R2T_test, acc_detection_test, loss_R2T_test, loss_detection_test, cm_similarity, cm_detection,loss_T2R_test,acc_T2R_test,test_outputs,test_labels


if __name__ == "__main__":
    train()

