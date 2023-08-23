# -*- coding: utf-8 -*-
r"""
NCGBT
################################################
Reference:
    Zihan Lin*, Changxin Tian*, Yupeng Hou*, Wayne Xin Zhao. "Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning." in WWW 2022.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import faiss
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

class NCGBT(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, bug_embedding):
        super(NCGBT, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type: the embedding size of the base model
        self.n_layers = config['n_layers']          # int type: the layer num of the base model
        self.reg_weight = config['reg_weight']      # float32 type: the weight decay for l2 normalization

        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.hyper_layers = config['hyper_layers']

        self.alpha = config['alpha']

        self.rep_reg = config['proto_reg']
        self.k = config['num_clusters']

        # define layers and loss
        self.bug_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.developer_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.bug_embedding = torch.nn.Embedding.from_pretrained(bug_embedding)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_bug_e = None
        self.restore_developer_e = None

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_bug_e', 'restore_developer_e']

        self.bug_centroids = None
        self.bug_2cluster = None
        self.developer_centroids = None
        self.developer_2cluster = None

    def e_step(self):
        bug_embeddings = self.bug_embedding.weight.detach().cpu().numpy()
        developer_embeddings = self.developer_embedding.weight.detach().cpu().numpy()
        self.bug_centroids, self.bug_2cluster = self.run_kmeans(bug_embeddings)
        self.developer_centroids, self.developer_2cluster = self.run_kmeans(developer_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of bugs and developers.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of bugs and developers and combine to an embedding matrix.
        """
        bug_embeddings = self.bug_embedding.weight
        developer_embeddings = self.developer_embedding.weight
        ego_embeddings = torch.cat([bug_embeddings, developer_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers*2)):
            all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers+1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        bug_all_embeddings, developer_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return bug_all_embeddings, developer_all_embeddings, embeddings_list

    def repNCE_loss(self, node_embedding, bug, developer):
        bug_embeddings_all, developer_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        bug_embeddings = bug_embeddings_all[bug]     # [B, e]
        norm_bug_embeddings = F.normalize(bug_embeddings)

        bug2cluster = self.bug_2cluster[bug]     # [B,]
        bug2centroids = self.bug_centroids[bug2cluster]   # [B, e]
        pos_score_bug = torch.mul(norm_bug_embeddings, bug2centroids).sum(dim=1)
        pos_score_bug = torch.exp(pos_score_bug / self.ssl_temp)
        ttl_score_bug = torch.matmul(norm_bug_embeddings, self.bug_centroids.transpose(0, 1))
        ttl_score_bug = torch.exp(ttl_score_bug / self.ssl_temp).sum(dim=1)

        rep_nce_loss_bug = -torch.log(pos_score_bug / ttl_score_bug).sum()

        developer_embeddings = developer_embeddings_all[developer]
        norm_developer_embeddings = F.normalize(developer_embeddings)

        developer2cluster = self.developer_2cluster[developer]  # [B, ]
        developer2centroids = self.developer_centroids[developer2cluster]  # [B, e]
        pos_score_developer = torch.mul(norm_developer_embeddings, developer2centroids).sum(dim=1)
        pos_score_developer = torch.exp(pos_score_developer / self.ssl_temp)
        ttl_score_developer = torch.matmul(norm_developer_embeddings, self.developer_centroids.transpose(0, 1))
        ttl_score_developer = torch.exp(ttl_score_developer / self.ssl_temp).sum(dim=1)
        rep_nce_loss_developer = -torch.log(pos_score_developer / ttl_score_developer).sum()

        rep_nce_loss = self.rep_reg * (rep_nce_loss_bug + rep_nce_loss_developer)
        return rep_nce_loss

    def ssl_layer_loss(self, current_embedding, previous_embedding, bug, developer):
        current_bug_embeddings, current_developer_embeddings = torch.split(current_embedding, [self.n_users, self.n_items])
        previous_bug_embeddings_all, previous_developer_embeddings_all = torch.split(previous_embedding, [self.n_users, self.n_items])

        current_bug_embeddings = current_bug_embeddings[bug]
        previous_bug_embeddings = previous_bug_embeddings_all[bug]
        norm_bug_emb1 = F.normalize(current_bug_embeddings)
        norm_bug_emb2 = F.normalize(previous_bug_embeddings)
        norm_all_bug_emb = F.normalize(previous_bug_embeddings_all)
        pos_score_bug = torch.mul(norm_bug_emb1, norm_bug_emb2).sum(dim=1)
        ttl_score_bug = torch.matmul(norm_bug_emb1, norm_all_bug_emb.transpose(0, 1))
        pos_score_bug = torch.exp(pos_score_bug / self.ssl_temp)
        ttl_score_bug = torch.exp(ttl_score_bug / self.ssl_temp).sum(dim=1)

        ssl_loss_bug = -torch.log(pos_score_bug / ttl_score_bug).sum()

        current_developer_embeddings = current_developer_embeddings[developer]
        previous_developer_embeddings = previous_developer_embeddings_all[developer]
        norm_developer_emb1 = F.normalize(current_developer_embeddings)
        norm_developer_emb2 = F.normalize(previous_developer_embeddings)
        norm_all_developer_emb = F.normalize(previous_developer_embeddings_all)
        pos_score_developer = torch.mul(norm_developer_emb1, norm_developer_emb2).sum(dim=1)
        ttl_score_developer = torch.matmul(norm_developer_emb1, norm_all_developer_emb.transpose(0, 1))
        pos_score_developer = torch.exp(pos_score_developer / self.ssl_temp)
        ttl_score_developer = torch.exp(ttl_score_developer / self.ssl_temp).sum(dim=1)

        ssl_loss_developer = -torch.log(pos_score_developer / ttl_score_developer).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_bug + self.alpha * ssl_loss_developer)
        return ssl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_bug_e is not None or self.restore_developer_e is not None:
            self.restore_bug_e, self.restore_developer_e = None, None

        bug = interaction[self.USER_ID]
        pos_developer = interaction[self.ITEM_ID]
        neg_developer = interaction[self.NEG_ITEM_ID]

        bug_all_embeddings, developer_all_embeddings, embeddings_list = self.forward()

        center_embedding = embeddings_list[0]
        context_embedding = embeddings_list[self.hyper_layers * 2]

        ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, bug, pos_developer)
        rep_loss = self.repNCE_loss(center_embedding, bug, pos_developer)

        u_embeddings = bug_all_embeddings[bug]
        pos_embeddings = developer_all_embeddings[pos_developer]
        neg_embeddings = developer_all_embeddings[neg_developer]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.bug_embedding(bug)
        pos_ego_embeddings = self.developer_embedding(pos_developer)
        neg_ego_embeddings = self.developer_embedding(neg_developer)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss, ssl_loss, rep_loss



    def predict(self, interaction):
        bug = interaction[self.USER_ID]
        developer = interaction[self.ITEM_ID]

        bug_all_embeddings, developer_all_embeddings, embeddings_list = self.forward()

        u_embeddings = bug_all_embeddings[bug]
        i_embeddings = developer_all_embeddings[developer]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        bug = interaction[self.USER_ID]
        if self.restore_bug_e is None or self.restore_developer_e is None:
            self.restore_bug_e, self.restore_developer_e, embedding_list = self.forward()
        # get bug embedding from storage variable
        u_embeddings = self.restore_bug_e[bug]

        # dot with all developer embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_developer_e.transpose(0, 1))

        return scores.view(-1)
