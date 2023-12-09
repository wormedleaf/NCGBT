# -*- coding: utf-8 -*- as np
import numpy as np
import scipy.sparse as sp
import torch
from torch_multi_head_attention import MultiHeadAttention as SelfAttention
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

class Bert(torch.nn.Module):

    def __init__(self, pretrained):
        super().__init__()
        self.pretrained = pretrained
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        return out

class GCBT(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, bug_embedding):
        super(GCBT, self).__init__(config, dataset)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.latent_dim = config['embedding_size']  # int type: the embedding size of the base model
        self.n_layers = config['n_layers']  # int type: the layer num of the base model
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization

        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.hyper_layers = config['hyper_layers']
        self.hidden_size = config['embedding_size']
        self.heads_num = 4

        self.alpha = config['alpha']

        self.proto_reg = config['proto_reg']
        self.k = config['num_clusters']

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.user_embedding = torch.nn.Embedding.from_pretrained(bug_embedding)

        self.key_layer = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.query_layer = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.value_layer = torch.nn.Linear(self.latent_dim, self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat = self.get_norm_adj_mat(self.interaction_matrix).to(self.device)

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_norm_adj_mat(self, inter_matrix):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        # print("A", self.n_users * self.n_items)

        inter_M = inter_matrix
        # print("inter_m:\n", inter_M.shape, type(inter_M))
        inter_M_t = inter_M.transpose()

        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        # print("inter_M.row:", type(inter_M.row))
        # print("inter_M.col:", type(inter_M.col))
        # print("inter.nnz:", len(([1] * inter_M.nnz)))
        # print("data_dict", len(data_dict))
        # N
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        # print("data_dict", len(data_dict))
        # 2N
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)
        # print("sumArr", sumArr.shape)
        # (n_u+n_v, 1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        index = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(index, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        """Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, norm_adj_mat):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
            all_embeddings = torch.sparse.mm(norm_adj_mat, all_embeddings)
            embeddings_list.append(all_embeddings)

        # spatial convolution
        spatial_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
        spatial_embeddings = torch.mean(spatial_embeddings, dim=1)

        # temporal convolution
        # self_attention = SelfAttention(self.latent_dim, self.heads_num)
        # self_attention = self_attention.to(self.device)
        # temporal_all_embeddings, _ = self_attention(
        #     self.query_layer(spatial_embeddings),
        #     self.key_layer(spatial_embeddings),
        #     self.value_layer(spatial_embeddings))

        user_all_embeddings, item_all_embeddings = torch.split(spatial_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def calculate_loss(self, interaction):
        # print("interaction", interaction)
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        # print("pos_item:", pos_item.shape, pos_item)
        neg_item = interaction[self.NEG_ITEM_ID]
        # print("neg_item:", neg_item.shape, neg_item)

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward(self.norm_adj_mat)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward(self.norm_adj_mat)

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.forward(self.norm_adj_mat)
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]
        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores.view(-1)
