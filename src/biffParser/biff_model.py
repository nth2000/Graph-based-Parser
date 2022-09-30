import torch.nn as nn
import torch.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class ReLUMLP(nn.Module):
    def __init__(self,input_size:int,output_size:int,dropout:float):
        super().__init__()
        self.linear_proj =  nn.Linear(in_features = input_size,out_features = output_size,bias = True)
        self.RELU = nn.ReLU()
        self.drop_out = nn.Dropout(dropout)
    def forward(self,x):

        x = self.linear_proj(x)
        x = self.RELU(x)
        x = self.drop_out(x)
        return x



class biaffineparser(nn.Module):

    def __init__(self,embedding_dim:int,
                 drop_out:float,
                 lstm_hidden_size:int,
                 arc_mlp_size:int,
                 label_mlp_size:int,
                 lstm_depth:int,
                 vocabulary_size:int,
                 padding_idx:int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.drop_out_rate = drop_out
        self.lstm_hidden_size = lstm_hidden_size
        self.arc_mlp_size = arc_mlp_size
        self.label_mlp_size = label_mlp_size
        self.lstm_depth = lstm_depth
        self.drop_out = nn.Dropout(drop_out)
        self.LSTM = nn.LSTM(input_size = embedding_dim*2,hidden_size=lstm_hidden_size,num_layers = lstm_depth,bias = True,batch_first = True,dropout = drop_out,bidirectional=True)
        self.label_dep_MLP = ReLUMLP(input_size = 2*lstm_hidden_size,output_size = label_mlp_size,dropout = drop_out)
        self.label_head_MLP = ReLUMLP(input_size=2 * lstm_hidden_size, output_size=label_mlp_size, dropout=drop_out)
        self.arc_dep_MLP = ReLUMLP(input_size = 2*lstm_hidden_size,output_size = arc_mlp_size,dropout = drop_out)
        self.arc_head_MLP = ReLUMLP(input_size = 2*lstm_hidden_size,output_size = arc_mlp_size,dropout = drop_out)
        self.embedding = torch.nn.Embedding(num_embeddings=vocabulary_size,
                                            embedding_dim=embedding_dim,
                                            padding_idx=padding_idx)
        self.u_2 = nn.Parameter(data = torch.empty((arc_mlp_size,1)))
        self.U_1 = nn.Parameter(data = torch.empty((arc_mlp_size,arc_mlp_size)))
        self.U_1_label = nn.Parameter(data = torch.empty((label_mlp_size,label_mlp_size)))
#        self.U_2_label = nn.Parameter(data = torch.empty())
    def forward(self,sentence: torch.LongTensor,
                pos: torch.LongTensor,
                length: torch.LongTensor,
                true_head = None):
        """

        Args:
            sentence:一个batch中的句子，维度为[bsz,maxlength]
            pos:一个batch中句子中每个词对应的pos tagging,维度为[bsz,maxlength]
            length:一个batch中每个句子的长度，维度为[bsz,]
            true_head:一个batch中每个句子中每个词作为dependent时true head对应的下标,维度为[bsz,maxlength]

        Returns:

        """
        bsz = sentence.shape[0]
        embed_word = self.embedding(sentence)
        embed_pos = self.embedding(pos)
        input = torch.cat((embed_word,embed_pos),dim = -1)
        input = pack_padded_sequence(input,length,True,False)

        R,_ = self.LSTM(input)
        R_,length = pad_packed_sequence(R,True)

        H_arc_dep = self.arc_dep_MLP(R_)
        H_arc_head = self.arc_head_MLP(R_)
        H_label_dep = self.label_dep_MLP(R_)

        U_ = self.U_1.squeeze(0).repeat(bsz,1,1)
        # S_arc = torch.bmm(torch.bmm(H_arc_head,U_),H_arc_dep.transpose(1,2)) +  H_arc_head @ self.u_2

        max_len = max(length)
        S_arc = torch.bmm(torch.bmm(H_arc_head,U_),H_arc_dep.transpose(1,2))
        mask = torch.zeros(size = (bsz,max_len,max_len))

        for id,l in enumerate(length):
            mask[id,l:,:]=1
        if mask is not None:      #mask padded token
            S_arc.data.masked_fill_(mask.bool(),-float('inf'))

        #计算每个head classifier的得分，S_arc的维度为(bsz,max_seq,max_seq)
        pred_head = torch.argmax(S_arc,dim = 1).detach() #每个词作为depent对应的head
        # one_hot_m = F.one_hot(pred_head,num_classes = max_len)
        # head_hidden = torch.bmm(one_hot_m,R)  #(bsz,max_seq,2*hidden_size) seq中每个单词作为dependent时其head在LSTM中的输出隐藏状态
        # H_label_head = self.label_head_MLP(head_hidden)

        return S_arc,pred_head

