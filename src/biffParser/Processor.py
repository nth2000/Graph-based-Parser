from data import *
import torch.optim as optim
import torch.nn as nn
from biff_model import biaffineparser
from  torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch
import logging
import time
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)




class Processor(object):



    def __init__(self,batch_size:int,
                 vocalulary:Vocabulary
                 ):
        self.bsz = batch_size
        self.vocabulary = vocalulary
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self,
              epchos:int,
              alpha:float,
              beta1:float,
              beta2:float,
              training_data:DataManager,
              test_data:DataManager,
              eval_intern:int,
              model:biaffineparser,
              model_save_path
              ):
        """

        Args:
            epchos:训练最大迭代轮数
            alpha:学习率
            beta1:adam算法中的参数beta1
            beta2:adam算法中的参数beta2
            training_data:训练数据
            test_data:测试数据
            eval_intern:多少epcho进行一次测试
            model:模型实例

        Returns:

        """
        loss = nn.CrossEntropyLoss(reduction = 'none')
        optimizer = optim.AdamW(model.parameters(),lr = alpha,betas = (beta1,beta2))
        lambda_annealing = lambda t:0.75**(t/5000)
        scheduler = LambdaLR(optimizer,lr_lambda=lambda_annealing)
        training_data = training_data.package(batch_size=self.bsz,shuffle = True)
        model = model.to(self.device)
        best_acc = 0
        for j in tqdm(range(epchos)):
            training_correct = 0
            total_length = 0
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            total_loss = torch.tensor([0]).float()
            for sentence,pos,dependent in training_data:
                score = torch.tensor([0]).float()
                sentence_ = self.to_tensor(sentence) #转变为tensor后的sentence_
                pos_ = self.to_tensor(pos)   #转变为tensor后的pos_
                dependent_ = self.to_tensor(dependent)
                assert dependent_[0][0]==-1
                dependent_[:,:1]=0
                length = torch.LongTensor([len(x) for x in sentence]).to(self.device) #真实长度
                S_arc,pred_head = model(sentence_,pos_,length)  #前向传播
                loss_  = loss(S_arc,dependent_)
                for i in range(len(loss_)):
                    a = torch.sum(loss_[i][1:length[i]])
                    b = (length[i] - 1).float()
                    score += a/b
                    training_correct += (pred_head[i,1:length[i]] == dependent_[i,1:length[i]]).sum()
                    total_length += length[i] - 1  #去除root结点
                score/=len(sentence)  #仅一个batch内的反向传播
                total_loss += score
                score.backward()
                optimizer.step()
            scheduler.step()

            if j % eval_intern == 0:
                logger.info(msg = "epcho = {},evaluating the model".format(j))
                logger.info(msg = "epcho = {},loss = {}".format(j,total_loss))
                logger.info(msg = "epcho = {},training accuracy = {}".format(j,training_correct/total_length))
                test_acc = self.evaluate(model,test_data,None)
                logger.info(msg = "epcho = {},testing accuracy = {}".format(j,test_acc))
                if test_acc > best_acc:
                    logger.info(msg = 'new powerful model detected! save the model!')
                    torch.save(model.state_dict(),model_save_path+"/" + str(time.time()) + ".pkl")
                    best_acc = test_acc


    def evaluate(self,
                 model:biaffineparser,
                 test_data:DataManager,
                 decoder):
        test_data = test_data.package(batch_size = self.bsz,shuffle = True)
        total_length = 0
        total_correct = 0
        for sentence, pos, dependent in test_data:  #遍历测试数据集
            sentence_ = self.to_tensor(sentence)  # 转变为tensor后的sentence_
            pos_ = self.to_tensor(pos)  # 转变为tensor后的pos_
            dependent_ = self.to_tensor(dependent)
            dependent_[:, 1:] = 0
            length = torch.LongTensor([len(x) for x in sentence]).to(self.device)  # 真实长度
            S_arc, pred_head = model(sentence_, pos_, length)  # 前向传播
            bsz = len(sentence)
            for i in range(bsz):
                total_correct  += (pred_head[i,1:length[i]] == dependent_[i,1:length[i]]).sum()
                total_length += length[i] - 1
        return total_correct / total_length




    def to_tensor(self,
                  x,
                  ):
        """

        Args:
            x:
        Returns:

        """
        max_len = len(max(x,key = len))
        #利用vocab中的填充token填充
        if isinstance(x[0][0],str):
            tensor = [[self.vocabulary[t] for t in s] for s in x]
        else:
            tensor = [[i for i in s] for s in x]
        for s in tensor:
            c = len(s)
            s+=[self.vocabulary['<PAD>'] if isinstance(x[0][0],str) else 0 for t in range(max_len - c)]

        return torch.LongTensor(tensor).to(self.device)
