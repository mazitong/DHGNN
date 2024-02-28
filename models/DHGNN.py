from torch import nn
import torch
import torch.nn.functional as F


class DHGNN(nn.Module):
    def __init__(self,nums,norm_L,args):
        super(DHGNN,self).__init__()
        self.nums=nums
        self.norm_L=norm_L
        self.device = args.device
        self.emb_size = args.embed_size

        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout

        self.layers = eval(args.layer_size)
        self.weight_dict,self.embedding_dict = self.init_weight()
        self.norm_L = torch.tensor(norm_L.todense(),dtype=torch.float32).to(self.device)
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(norm_L).to(self.device)


    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'emb': nn.Parameter(initializer(torch.empty(self.nums,
                                                 self.emb_size)))
        })
        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return  weight_dict,embedding_dict
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def rating(self, i_embeddings, pos_i_embeddings):
        return torch.matmul(i_embeddings, pos_i_embeddings.t())

    def forward(self, drop_flag=True):

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = self.embedding_dict['emb']

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)  # L* embddding

            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                             + self.weight_dict['b_gc_%d' % k]

            ego_embeddings = sum_embeddings

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]  # embedding

        all_embeddings = torch.cat(all_embeddings, 1)

        i_g_embeddings = all_embeddings

        return i_g_embeddings
