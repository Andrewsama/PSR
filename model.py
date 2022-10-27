import world
import torch
import torch.nn as nn
import torch.nn.functional as F

class PureBPR(nn.Module):
    def __init__(self, config, dataset):
        super(PureBPR, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        print("using Normal distribution initializer")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss


class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self._init_weight()

    def _init_weight(self):
        self.num_users = self.dataset.n_users  # user
        self.num_items = self.dataset.m_items  # item
        # self.beta = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.latent_dim = self.config['latent_dim_rec']   # 64
        self.n_layers = self.config['layer']   # 3
        self.embedding_user = torch.nn.Embedding(    # user embedding
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(    # item embedding
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)  
        nn.init.normal_(self.embedding_item.weight, std=0.1)  
        '''1'''
        # self.embedding_user.weight = F.normalize(self.embedding_user.weight,p=2,dim=1)
        # self.embedding_item.weight = F.normalize(self.embedding_item.weight,p=2,dim=1)

        self.f = nn.Sigmoid()
        self.interactionGraph = self.dataset.getInteractionGraph()
        print(f"{world.model_name} is already to go")

    
    # def computer(self):
    #     """
    #     propagate methods for lightGCN
    #     """
    #     users_emb = self.embedding_user.weight
    #     items_emb = self.embedding_item.weight
    #     all_emb = torch.cat([users_emb, items_emb])
    #     #   torch.split(all_emb , [self.num_users, self.num_items])
    #     embs = [all_emb]
    #     G = self.interactionGraph
    #
    #     for layer in range(self.n_layers):
    #         all_emb = torch.sparse.mm(G, all_emb)  
    #         embs.append(all_emb)
    #     embs = torch.stack(embs, dim=1)
    #     # print(embs.size())
    #     light_out = torch.mean(embs, dim=1)       
    #     
    #     users, items = torch.split(light_out, [self.num_users, self.num_items])
    #     self.final_user, self.final_item = users, items
    #     return users, items

  
    def getUsersRating(self, users):
        all_users, all_items = self.final_user, self.final_item
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
  
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]    # user
        pos_emb = all_items[pos_items]  # pos item
        neg_emb = all_items[neg_items]  # neg item
        users_emb_ego = self.embedding_user(users)  # user
        pos_emb_ego = self.embedding_item(pos_items)# pos item
        neg_emb_ego = self.embedding_item(neg_items)# neg item
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

 
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        # print('pos_emb_shape:',pos_emb.shape)
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        
        pos_scores = torch.mul(users_emb, pos_emb) 
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        '''GOR?'''
        # left = F.normalize(users_emb, p=2, dim=1)
        # right = F.normalize(neg_emb, p=2, dim=1)
        # M_ = torch.mul(left, right)

        M_ = torch.mul(users_emb, neg_emb)
        M1 = torch.pow(torch.mean(M_), 2)
        M2 = torch.mean(torch.pow(M_, 2)) - torch.tensor(1/float(self.latent_dim))
        attr_loss = M1 + F.softplus(M2)

     
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # print('loss',loss)
        # print('reg_loss',reg_loss)

        # return loss, reg_loss
        '''change'''
        return loss, reg_loss, attr_loss# * self.beta


class PSR(LightGCN):
    def _init_weight(self):
        super(PSR, self)._init_weight()
        self.socialGraph = self.dataset.getSocialGraph()
        # self.user2item = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        # self.item2user = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        # self.social_s = []
        # self.social_u = []
        # self.social_c = []
        # self.social_i = []
        # for layer in range(self.n_layers):
        #     self.social_s.append(nn.Linear(self.latent_dim * 2, self.latent_dim, bias=False).cuda())
        #     self.social_u.append(nn.Linear(self.latent_dim, self.latent_dim, bias=False).cuda())
        #     self.social_c.append(nn.Linear(self.latent_dim, self.latent_dim, bias=False).cuda())
        #     self.social_i.append(nn.Linear(self.latent_dim, self.latent_dim, bias=False).cuda())

        self.social_s = nn.Linear(self.latent_dim * 2, self.latent_dim, bias=False)
        self.social_u = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.social_c = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.social_i = nn.Linear(self.latent_dim, self.latent_dim, bias=False)

        # self.Graph_Comb = []
        # for layer in range(self.n_layers):
        #     self.Graph_Comb.append(Graph_Comb(self.latent_dim).cuda())
        self.Graph_Comb = Graph_Comb(self.latent_dim)
        # self.item_comb = item_comb(self.latent_dim)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # social_emb = self.embedding_social.weight
        all_emb = torch.cat([users_emb, items_emb])
        A = self.interactionGraph  # item * user 
        S = self.socialGraph       # user * user
        embs = [all_emb]
        # all_social = [social_emb]
        for layer in range(self.n_layers):
            # # 1.
            # all_emb_interaction = torch.sparse.mm(A, all_emb)
            # users_emb_interaction, items_emb_next = torch.split(all_emb_interaction, [self.num_users, self.num_items])
            # # users_emb_next = torch.tanh(self.social_i[layer](users_emb_interaction))
            # users_emb_next = torch.tanh(self.social_i(users_emb_interaction))
            #
            # all_emb = torch.cat([users_emb_next, items_emb_next])
            #
            # # 2.
            # users_emb_social = torch.sparse.mm(S, users_emb)
            # # users_emb = torch.tanh(self.social_u[layer](users_emb_social))
            # users_emb = torch.tanh(self.social_u(users_emb_social))
            #
            # # 3.
            # users = self.social_s(torch.cat([users_emb_next, users_emb],dim=1))
            # #
            # # users = self.social_c(users_emb_next + users_emb)
            #
            # users = users / users.norm(2)
            # embs.append(torch.cat([users, items_emb_next]))
            '''ablation'''
            # embedding from last layer
            users_emb, items_emb = torch.split(all_emb, [self.num_users, self.num_items])
            # social network propagation(user embedding)
            users_emb_social = torch.sparse.mm(S, users_emb)
            # user-item bi-network propagation(user and item embedding)
            all_emb_interaction = torch.sparse.mm(A, all_emb)
            # get users_emb_interaction
            users_emb_interaction, items_emb_next = torch.split(all_emb_interaction, [self.num_users, self.num_items])
            # graph fusion model
            users_emb_next = self.Graph_Comb(users_emb_social, users_emb_interaction)
            all_emb = torch.cat([users_emb_next, items_emb_next])
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        final_embs = torch.mean(embs, dim=1)
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items


class Graph_Comb(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))
        h2 = torch.tanh(self.att_y(y))
        output = self.comb(torch.cat((h1, h2), dim=1))
        output = output / output.norm(2)
        # output = F.normalize(output, p=2, dim=1)
        return output
