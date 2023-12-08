import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import kaiming_uniform_
import torch.nn.functional as F

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def get_param(*args):
    param = Parameter(torch.empty(args[0], args[1]))
    kaiming_uniform_(param.data, mode='fan_in', nonlinearity='relu')
    return param

# Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=num_heads, batch_first=True
            ),
            num_layers=num_layers,
        )

    def forward(self, x):
        x = self.transformer_encoder(x) # batch_size * len * num_feature
        x = x.mean(dim=1)  # batch_size * num_feature
        return x

# LSTM
class LSTMModel(nn.Module):
    def __init__(self, hidden_size, num_layers, input_feature, out_feature):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
        self.lstm = nn.LSTM(input_size=input_feature, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, out_features=out_feature)

    def forward(self, x): 
        output, (h, c) = self.lstm(x)
        h_output = self.out(h[-1, :,:])
        return h_output


# sequential graph convolutiion
class GraphAttentionLayer(nn.Module):
    def __init__(self, total_activities_num):
        super(GraphAttentionLayer, self).__init__()
        self.activity_num = total_activities_num
        self.W = get_param(self.activity_num, self.activity_num)
        
     # attention normalization
    def _normalize(self, x, epsilon=1e-12):
        norm = torch.sum(x, dim=1).reshape(-1, 1)
        norm_x = x / (norm + epsilon)
        return norm_x
    
    def _prepare_attentional_input(self, currents, targets, activities_features, cases_features):
        att_input = torch.zeros((self.activity_num, self.activity_num), device=activities_features.device)
        
        # attention calculation
        for head_idx, case_emb, tail_idx in zip(currents, cases_features, targets):
            h_emb = activities_features[head_idx]
            t_emb = activities_features[tail_idx]

            dist = torch.norm(h_emb + case_emb - t_emb)
            att_input[tail_idx][head_idx] += torch.exp(-dist)
                
        attention = self._normalize(att_input)
        return attention

    def forward(self, currents, targets, activities_features, cases_features):
        attention = self._prepare_attentional_input(
            currents, targets, activities_features, cases_features)
        
        # feature update
        activities_features = torch.mm(self.W, activities_features) 
        h = torch.mm(attention, activities_features) + activities_features
        return h 


# SGAP
class SGAP(nn.Module):
    def __init__(
        self,
        activity_num,
        layer_num,
        hidden_size,
        feature_num,
        dropout,
        encoder_type,
        encoder_layer,
        num_heads=1,
    ):
        super(SGAP, self).__init__()
        self.activity_num = activity_num
        self.layer_num = layer_num
        self.feature_num = feature_num
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.encoder_layer = encoder_layer
        
        self.att = nn.ModuleList([GraphAttentionLayer(self.activity_num) for i in range(layer_num)])
        self.activities_features = nn.Embedding(activity_num, feature_num)
        self._generate_encoder_layer(encoder_type, hidden_size, num_heads)

    # init sequence encoding model
    def _generate_encoder_layer(self, encoder_type, hidden_size, num_heads):
        if encoder_type == 'transformer':
            self.case_encoder = TransformerModel(self.feature_num, self.encoder_layer, num_heads)
        elif encoder_type == 'lstm':
            self.case_encoder = LSTMModel(hidden_size, self.encoder_layer, self.feature_num, self.feature_num)
    
    def feature_conv(self, data, activities_features):
        # temporal encoding
        self.case_features = self.case_encoder(self.activities_features(data[:, :-1]))
        
        # sequential graph convolution
        for i in range(self.layer_num):
            if i == 0: x = F.relu(self.att[0](data[:, -2], data[:, -1], activities_features, self.case_features))
            else: x = F.relu(self.att[i](data[:, -2], data[:, -1], x, self.case_features))
        
        return F.dropout(x, self.dropout, training=self.training)
        
    def forward(self, conv_data, data=None):   
        if data == None:
            data = conv_data
            
        # init activity feature matrix
        activities_features = self.activities_features(torch.arange(self.activity_num, device=conv_data.device))
        
        # attention graph network
        conv_activity_feature = self.feature_conv(conv_data, activities_features) 
        
        # next-activity prediction
        pre_head = conv_activity_feature[data[:, -1]]  
        pre_relation = self.case_encoder(conv_activity_feature[data[:, :-2]])          
        pre_emb = pre_head + pre_relation

        pre_emb = torch.repeat_interleave(pre_emb, self.activity_num, 0)
        activity_emb = activities_features.repeat(len(data[:, -1]), 1)
        
        logits = - torch.norm((pre_emb - activity_emb), dim=1, keepdim=True)
        logits = logits.view(-1, self.activity_num)

        return logits
