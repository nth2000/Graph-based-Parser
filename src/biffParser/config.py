import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='train',
                    help="train model or predict result")
parser.add_argument('--dropout_rate',type=float,default=0.33,help  =  'dropout rate of the model')
parser.add_argument('--lstm_hidden_size',type = int,default = 500,help = 'BI-LSTM model hidden size')
parser.add_argument('--label_mlp_size',type = int,default=100,help = 'label mlp size')
parser.add_argument('--arc_mlp_size',type = int,default=100,help = 'arc mlp size')
parser.add_argument('--lstm_depth',type = int,default=3,help = 'lstm layer depth')
parser.add_argument('--training_data_path',type = str,default='./data/train.conll',help  ='training data path')
parser.add_argument('--test_data_path',type = str,default='./data/test.conll',help  ='test data path')
parser.add_argument('--model_save_path',type = str,default='./model',help = 'path to save the model')
parser.add_argument('--eval_intern',type = int,default=50,help = 'how many intern to eval the model')
parser.add_argument('--embedding_dim',type = int,default = 100,help = 'embeddng layer dim')
parser.add_argument('--max_epcho',type = int,default = 500,help = 'maximal training steps')
parser.add_argument('--max_line',type = int,default = 5,help = 'maxline to read in the sentence,use for debugging')
parser.add_argument('--batch_size',type = int,default = 2,help  = 'batch size of the training and evaluate')
parser.add_argument('--alpha',type = float,default=2e-3,help = 'learning rate of the adam algorithm')
parser.add_argument('--beta1',type = float,default=0.9,help  = 'beta1 of the adam algorithm')
parser.add_argument('--beta2',type = float,default=0.9,help  = 'beta2 of the adam algorithm')
parser.add_argument('--existing_model_path',type=str,default=None,help = 'path to the existing model if there is any')
