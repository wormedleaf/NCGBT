import torch
import random
import datasets
from transformers import BertModel
from datasets import load_dataset
from datasets import Features
from datasets import load_from_disk
from transformers import BertTokenizer
from transformers import AdamW
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sort_with_idx(source,idx):
    #根据索引对tensor进行排序
    "source: b c h w ,  idx: b c 1 1"
    b,_ = source.size()
    after_sort = torch.zeros_like(source)
    for i in range(b):
            after_sort[i][:] = source[idx[i]][:]
    return after_sort

class Dataset(torch.utils.data.Dataset):

    def __init__(self, split, dataset_name):
        #self.dataset = load_dataset(path='seamew/ChnSentiCorp', split=split)
        data_files = {"train": "./datas/" + dataset_name + "/" + dataset_name + "_my.clean.txt"}
        self.dataset = load_dataset(path='text', data_files=data_files, split=split)

        # def f(data):
        #     return len(data) > 500
        #
        # self.dataset = self.dataset.filter(f)
        # new_features = self.dataset.features.copy()
        # new_features['priority'] = datasets.ClassLabel(names=['1', '2', '3', '4', '5'])
        # new_features['my_method'] = datasets.Value('large_string')
        # new_features['description'] = datasets.Value('large_string')
        # self.dataset = self.dataset.cast(new_features)
        # self.dataset = load_dataset(path='json',data_files='F:\project-graduate\Bert\dataset\/test\eclipse_my_test.json',split='test')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]
        # label = self.dataset[i]['priority']
        # sentence1 = text[:250]
        # sentence2 = text[250:500]
        # label = 0

        # if random.randint(0, 1) == 0:
        #     j = random.randint(0, len(self.dataset) - 1)
        #     sentence2 = self.dataset[j][250:500]
        #     label = 1

        return text['text']
        # return sentence1, sentence2, label

# dataset = Dataset('train')
# token = BertTokenizer.from_pretrained('bert-large-cased')
#
# # def collate_fn(data):
# #     sents = [i for i in data]
# #     # labels = [int(i[2]) for i in data]
# #     #编码
# #     data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
# #                                    truncation=True,
# #                                    padding='max_length',
# #                                    max_length=505,
# #                                    return_tensors='pt',
# #                                    return_length=True)
# #
# #     #input_ids:编码之后的数字
# #     #attention_mask:是补零的位置是0,其他位置是1
# #     input_ids = data['input_ids'].to(device)
# #     attention_mask = data['attention_mask'].to(device)
# #     token_type_ids = data['token_type_ids'].to(device)
# #     # labels = torch.LongTensor(labels).to(device)
# #
# #     #print(data['length'], data['length'].max())
# #
# #     return input_ids, attention_mask, token_type_ids
#
# loader = torch.utils.data.DataLoader(dataset=dataset,
#                                      batch_size=13,
#                                      collate_fn=collate_fn,
#                                      shuffle=False,
#                                      drop_last=False)
#
#
#
# #加载预训练模型
# pretrained = BertModel.from_pretrained('bert-large-cased')
# #需要移动到cuda上
# pretrained.to(device)
#
# for param in pretrained.parameters():
#     param.requires_grad_(False)
#
# class Model(torch.nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input_ids, attention_mask, token_type_ids):
#         with torch.no_grad():
#             out = pretrained(input_ids=input_ids,
#                              attention_mask=attention_mask,
#                              token_type_ids=token_type_ids)
#         return out
#
# model = Model()
# #同样要移动到cuda
# model.to(device)
#
# model.eval()
# for i, (input_ids, attention_mask, token_type_ids) in enumerate(loader):
#     out = model(input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids)
#     print(i, torch.tensor(out.last_hidden_state[:, 0]).size())
# print(out)