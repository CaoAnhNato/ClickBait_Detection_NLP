import math
import random
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from transformers import BertModel
from fractions import Fraction
import torch.nn.functional as F

random.seed(825)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

class CrossAttention(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.text_to_image_query = nn.Linear(text_dim, hidden_dim)
        self.text_to_image_key = nn.Linear(image_dim, hidden_dim)
        self.text_to_image_value = nn.Linear(image_dim, hidden_dim)
        
        self.image_to_text_query = nn.Linear(image_dim, hidden_dim)
        self.image_to_text_key = nn.Linear(text_dim, hidden_dim)
        self.image_to_text_value = nn.Linear(text_dim, hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text_emb, img_emb):

        # Text to Image Attention
        text_query = self.text_to_image_query(text_emb)  # [batch_size, text_seq_len, hidden_dim]
        image_key = self.text_to_image_key(img_emb)      # [batch_size, img_seq_len, hidden_dim]
        image_value = self.text_to_image_value(img_emb)  # [batch_size, img_seq_len, hidden_dim]

        attention_scores_text_to_img = torch.bmm(text_query, image_key.transpose(1, 2))  # [batch_size, text_seq_len, img_seq_len]
        attention_weights_text_to_img = F.softmax(attention_scores_text_to_img, dim=-1)  # [batch_size, text_seq_len, img_seq_len]

        text_to_img_attention = torch.bmm(attention_weights_text_to_img, image_value)    # [batch_size, text_seq_len, hidden_dim]

        # Image to Text Attention
        image_query = self.image_to_text_query(img_emb)  # [batch_size, img_seq_len, hidden_dim]
        text_key = self.image_to_text_key(text_emb)      # [batch_size, text_seq_len, hidden_dim]
        text_value = self.image_to_text_value(text_emb)  # [batch_size, text_seq_len, hidden_dim]

        attention_scores_img_to_text = torch.bmm(image_query, text_key.transpose(1, 2))  # [batch_size, img_seq_len, text_seq_len]
        attention_weights_img_to_text = F.softmax(attention_scores_img_to_text, dim=-1)  # [batch_size, img_seq_len, text_seq_len]

        img_to_text_attention = torch.bmm(attention_weights_img_to_text, text_value)     # [batch_size, img_seq_len, hidden_dim]

        return text_to_img_attention,img_to_text_attention

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.size()
        query = self.query_proj(x)  # [batch_size, seq_len, hidden_dim]
        key = self.key_proj(x)      # [batch_size, seq_len, hidden_dim]
        value = self.value_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # Compute attention scores
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)   # [batch_size, seq_len, seq_len]
        attention_output = torch.bmm(attention_weights, value)    # [batch_size, seq_len, hidden_dim]
        attention_output = self.output_proj(attention_output)     # [batch_size, seq_len, input_dim]

        output = torch.sum(attention_output, dim=1)  # [batch_size, input_dim]
        
        return output

class EncodingPart(nn.Module):
    def __init__(self,input_dim,shared_text_dim=128
    ):
        super(EncodingPart, self).__init__()
        self.shared_content_linear = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        self.shared_reason_linear = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )

    def forward(self, content, reason):
        text_shared = self.shared_content_linear(content)
        reason_shared = self.shared_reason_linear(reason)
        return text_shared, reason_shared

class A_Encoding_Part(nn.Module):
    def __init__(
        self,
        cnn_channel=32,
        cnn_kernel_size=(1, 2, 4, 8),
        shared_reason_dim=128,
        shared_text_dim=128
    ):
        super(A_Encoding_Part, self).__init__()
        self.shared_content_linear = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.ReLU()
        )
        self.shared_reason_linear = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.ReLU()
        )

    def forward(self, content, reason):

        text_shared = self.shared_content_linear(content)
        reason_shared = self.shared_reason_linear(reason)
        return text_shared, reason_shared

class Similarity(nn.Module):
    def __init__(self, shared_dim=128, sim_dim=64):
        super(Similarity, self).__init__()
        self.encoding = EncodingPart(256,128)   #128
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.reason_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.selfattention = SelfAttention(768,256)
        self.linear = nn.Linear(768,256)

    def forward(self, content, reason):#256
        content = self.linear(self.selfattention(content))
        text_encoding, reason_encoding = self.encoding(content,reason) #hidden_dim:128
        text_aligned = self.text_aligner(text_encoding)
        reason_aligned = self.reason_aligner(reason_encoding)
        sim_feature = torch.cat([text_aligned, reason_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, reason_aligned, pred_similarity

class  Reason_Similarity(nn.Module):
    def __init__(self, shared_dim=128, sim_dim=64):
        super( Reason_Similarity, self).__init__()
        self.reason_attention = SelfAttention(768,128)
        self.encoding = EncodingPart(256,128)   #128
        self.linear = nn.Linear(768, 256)
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.reason_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, content, reason):#[batch_size, text_seq_len, hidden_dim]

        reason = self.linear(reason)
        content = self.reason_attention(content)
        content = self.linear(content)
        text_encoding, reason_encoding = self.encoding(content,reason) #hidden_dim:128
        text_aligned = self.text_aligner(text_encoding)
        reason_aligned = self.reason_aligner(reason_encoding)
        sim_feature = torch.cat([text_aligned, reason_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature)
        return text_aligned, reason_aligned, pred_similarity

class Attention_Encoder(nn.Module):
    def __init__(self):
        super(Attention_Encoder, self).__init__()
        
        self.news_attention = SelfAttention(256,128) #text_dim,hidden_dim
        self.reason_attention = SelfAttention(256,128) #text_dim,hidden_dim
        self.attention = SelfAttention(768,256)
        self.linear = nn.Linear(768,256)
        self.cross_attention = CrossAttention(768, 768, 256)  #text_dim, image_dim, hidden_dim

        

    def forward(self, content, positive, negative):#[batch_size, text_seq_len, hidden_dim]
        
        pos_text2reason,pos_reason2text = self.cross_attention(content,positive) #3维
        pos_text2reason = self.news_attention(pos_text2reason) #2维
        pos_reason2text = self.reason_attention(pos_reason2text)
        # pos_reason2text = self.news_attention(pos_reason2text)
        positive = self.attention(positive)

        neg_text2reason,neg_reason2text = self.cross_attention(content,negative) #3维
        neg_text2reason = self.news_attention(neg_text2reason) #2维
        neg_reason2text = self.reason_attention(neg_reason2text)
        negative = self.attention(negative)
   
        return pos_reason2text,pos_text2reason,positive, neg_reason2text,neg_text2reason,negative  #256


class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, z_dim * 2),
        )
        # self.apply(self.init_weights)

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the input
        # print(x)
        params = self.net(x)
        # print("params:   ",params)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = torch.nn.functional.softplus(sigma, beta=1, threshold=20) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        self.encoding = EncodingPart()
        self.encoder_text = Encoder()
        self.encoder_reason = Encoder()

    def forward(self, content,reason):
        # text_encoding, reason_encoding = self.encoding(content, reason)
        p_z1_given_text = self.encoder_text(content)
        p_z2_given_reason = self.encoder_reason(reason)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_reason.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1+1e-8) - p_z2_given_reason.log_prob(z1+1e-8)
        kl_2_1 = p_z2_given_reason.log_prob(z2+1e-8) - p_z1_given_text.log_prob(z2+1e-8)
        skl = (kl_1_2 + kl_2_1)/ 2.
        skl = nn.functional.sigmoid(skl)
        return skl


class UnimodalDetection(nn.Module):
        def __init__(self, shared_dim=128, prime_dim = 16):
            super(UnimodalDetection, self).__init__()
            self.text_uni = nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU()
            )
            self.reason_uni = nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU()
            )

        def forward(self, text_encoding, reason_encoding):
            text_prime = self.text_uni(text_encoding)
            reason_prime = self.reason_uni(reason_encoding)
            return text_prime, reason_prime


class CrossModule4Batch(nn.Module):
    def __init__(self, text_in_dim=64, reason_in_dim=64, corre_out_dim=64):
        super(CrossModule4Batch, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, reason):
        text_in = text.unsqueeze(2)
        reason_in = reason.unsqueeze(1)
        corre_dim = text.shape[1]
        similarity = torch.matmul(text_in, reason_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        correlation_out = self.c_specific_2(correlation_p)
        return correlation_out

class Aggregator(nn.Module):
    def __init__(self, mode='full', out_dim=64):
        super(Aggregator, self).__init__()
        if mode == 'full':
            text_in_dim = 7 * 64
        elif mode == 'no_tf':
            text_in_dim = 5 * 64
        elif mode == 'no_ta':
            text_in_dim = 3 * 64
        else:
            raise ValueError(f'Unsupported aggregator mode: {mode}')

        self.mode = mode
        self.linear = nn.Sequential(
            nn.Linear(text_in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.reason_attention = SelfAttention(768,128)
        self.linr = nn.Linear(768,64)

    def forward(
        self,
        content,
        R2T_aligned_agr=None,
        T2R_aligned_agr=None,
        R_aligned_agr=None,
        R2T_aligned_dis=None,
        T2R_aligned_dis=None,
        R_aligned_dis=None
    ):
        content = self.linr(self.reason_attention(content))
        if self.mode == 'full':
            final_corre = torch.cat(
                [
                    content,
                    R2T_aligned_agr,
                    T2R_aligned_agr,
                    R_aligned_agr,
                    R2T_aligned_dis,
                    T2R_aligned_dis,
                    R_aligned_dis,
                ],
                1,
            )
        elif self.mode == 'no_tf':
            final_corre = torch.cat(
                [
                    content,
                    R2T_aligned_agr,
                    T2R_aligned_agr,
                    R2T_aligned_dis,
                    T2R_aligned_dis,
                ],
                1,
            )
        else:  # no_ta
            final_corre = torch.cat([content, R_aligned_agr, R_aligned_dis], 1)
        correlation_out = self.linear(final_corre)
        return correlation_out

class DetectionModule(nn.Module):
    def __init__(self, feature_dim=448, h_dim=64):
        super(DetectionModule, self).__init__()
        self.uni_repre = UnimodalDetection()
        self.cross_module = CrossModule4Batch()
        self.classifier_corre = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.Linear(h_dim, 32),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2)
        )
        

    def forward(self,feature):
        pre_label = self.classifier_corre(feature)
        pre_label = torch.sigmoid(pre_label)
        # print(pre_label)
        return pre_label

class BertEncoder(nn.Module):
    def __init__(self, output_dim, freeze):
        super().__init__()
        self.bert = BertModel.from_pretrained(
                    'bert-base-uncased',
                    output_hidden_states=True,
                    output_attentions = True, 
                    return_dict=True)
        hidden_dim = 768
    
        self.fc = nn.Linear(hidden_dim, output_dim)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

    def forward(self, ids):
        # print(ids)
        output = self.bert(ids, output_attentions=True)
        hidden = output.last_hidden_state
        return hidden

