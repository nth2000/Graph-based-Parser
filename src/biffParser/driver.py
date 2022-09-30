import config
from data import *
from biff_model import *
from Processor import *
if __name__ == '__main__':
    args = config.parser.parse_args()
    train_data = DataManager(mode = args.mode)
    train_data,vocab =  train_data.load(path = args.training_data_path,mode = args.mode,max_line = args.max_line)
    test_data = DataManager(mode = 'test')
    test_data,_ = test_data.load(path = args.test_data_path,mode = 'test',max_line = args.max_line)

    model =  biaffineparser(
        embedding_dim = args.embedding_dim,
        drop_out = args.dropout_rate,
        lstm_hidden_size = args.lstm_hidden_size,
        arc_mlp_size=args.arc_mlp_size,
        label_mlp_size = args.label_mlp_size,
        lstm_depth=args.lstm_depth,
        vocabulary_size=len(vocab),
        padding_idx = vocab['<PAD>']
    )
    if args.existing_model_path:
        logger.info(msg = 'loading existing model')
        model.load_state_dict(torch.load(args.existing_model_path))
    else:
        logger.info(msg = 'training model from scratch')
    process = Processor(batch_size = args.batch_size,vocalulary=vocab)
    process.train(
        epchos = args.max_epcho,
        beta1 = args.beta1,
        beta2 = args.beta2,
        training_data = train_data,
        test_data = test_data,
        eval_intern = args.eval_intern,
        model = model,
        alpha = args.alpha,
        model_save_path = args.model_save_path
    )
