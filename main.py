import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color

from ncgbt import NCGBT, Bert
from trainer import NCGBTTrainer
from utils import *
import numpy as np


from transformers import BertModel
from datasets import load_dataset
from transformers import BertTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def collate_fn(data):
    sents = [i for i in data]
    token = BertTokenizer.from_pretrained('bert-large-cased')
    # labels = [int(i[2]) for i in data]
    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=505,
                                   return_tensors='pt',
                                   return_length=True)

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    # labels = torch.LongTensor(labels).to(device)

    # print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids


def run_single_model(args, collate_fn):
    # configurations initialization
    config = Config(
        model=NCGBT,
        dataset=args.dataset, 
        config_file_list=args.config_file_list
    )
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    # item_id = torch.tensor(dataset.field2id_token['item_id'][1:].astype(np.int))
    # item_id = item_id.to(device)

    dataset_Bert = Dataset('train', args.dataset)
    loader = torch.utils.data.DataLoader(dataset=dataset_Bert,
                                         batch_size=39,
                                         collate_fn=collate_fn,
                                         shuffle=False,
                                         drop_last=False)

    # 加载预训练模型
    pretrained = BertModel.from_pretrained('bert-large-cased')
    # 需要移动到cuda上
    pretrained.to(device)
    for param in pretrained.parameters():
        param.requires_grad_(False)
    model = Bert(pretrained)
    # 同样要移动到cuda
    model.to(device)
    model.eval()
    out_all = torch.tensor([])
    out_all = out_all.to(device)
    for i, (input_ids, attention_mask, token_type_ids) in enumerate(loader):
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        out_all = torch.cat((out_all, torch.tensor(out.last_hidden_state[:, 0])), dim=0)
        print(i, torch.tensor(out.last_hidden_state[:, 0]).size(), out_all.size())
    # out_all = sort_with_idx(out_all, item_id)
    pad_emb = torch.tensor(np.zeros([1, 1024]))
    pad_emb = pad_emb.to(device)
    out_all = torch.cat((pad_emb, out_all), dim=0)
    out_all = out_all.float()
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = NCGBT(config, train_data.dataset, out_all).to(device)
    logger.info(model)

    # trainer loading and initialization
    trainer = NCGBTTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='eclipse', help='The datasets can be: mozilla, office, eclipse')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/overall.yaml',
        'properties/NCGBT.yaml'
    ]
    if args.dataset in ['eclipse']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config is not '':
        args.config_file_list.append(args.config)

    run_single_model(args, collate_fn)

# 'ml-1m', 'yelp', 'amazon-books', 'gowalla-merged', 'alibaba'