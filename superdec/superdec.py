import torch
import torch.nn as nn

from superdec.models.decoder import TransformerDecoder
from superdec.models.decoder_layer import DecoderLayer
from superdec.models.point_encoder import StackedPVConv
from superdec.models.heads import SuperDecHead
from superdec.lm_optimization.lm_optimizer import LMOptimizer

class SuperDec(nn.Module):
    def __init__(self, ctx):
        super(SuperDec, self).__init__()
        self.n_layers = ctx.decoder.n_layers
        self.n_heads = ctx.decoder.n_heads
        self.n_queries = ctx.decoder.n_queries
        self.deep_supervision = ctx.decoder.deep_supervision
        self.pos_encoding_type = ctx.decoder.pos_encoding_type
        self.dim_feedforward = ctx.decoder.dim_feedforward
        self.emb_dims = ctx.point_encoder.l3.out_channels # output dimension of pvcnn
        self.lm_optimization = False
        if self.lm_optimization:
            self.lm_optimizer = LMOptimizer()

        self.point_encoder = StackedPVConv(ctx.point_encoder)

        # After testing concat, reduce depth back to point encoder output channels
        self.cat_llm = ctx.llm_cat
        self.out_channels = ctx.point_encoder.l3.out_channels
        self.post_concat_proj = nn.Conv1d(self.out_channels*2, self.out_channels, kernel_size=1)

        decoder_layer = DecoderLayer(d_model=self.emb_dims, nhead=self.n_heads, dim_feedforward=self.dim_feedforward,
                                               batch_first=True, swapped_attention=ctx.decoder.swapped_attention)
        self.layers = TransformerDecoder(decoder_layer=decoder_layer, n_layers=self.n_layers,
                                         max_len=self.n_queries, pos_encoding_type=self.pos_encoding_type,
                                         masked_attention=ctx.decoder.masked_attention)

        self.layers.project_queries = nn.Sequential(
            nn.Linear(self.emb_dims, self.emb_dims),
            nn.ReLU(),
            nn.Linear(self.emb_dims, self.emb_dims),
        )
        self.heads = SuperDecHead(emb_dims=self.emb_dims)
        init_queries = torch.zeros(self.n_queries + 1, self.emb_dims)
        self.register_buffer('init_queries', init_queries) # TODO double check -> new codebase

    def forward(self, x):
        point_features = self.point_encoder(x) # [bs, n_points, C]

        if self.cat_llm:
            # Concatenate test_llm_embedding with all point features along channel dim
            bs, n_points, c = point_features.shape
            test_llm_embedding = torch.rand(size=(bs, self.out_channels), device=point_features.device, dtype=point_features.dtype)
            test_llm_embedding = test_llm_embedding[:, None, :].expand(bs, n_points, self.out_channels)
            point_features = torch.cat([point_features, test_llm_embedding], dim=-1)  # [bs, n_points, C+self.out_channels]

            # Reduce depth back to ctx.pointencoder.out_channel via 1x1 conv
            point_features = self.post_concat_proj(point_features.transpose(1, 2)).transpose(1, 2)

        refined_queries_list, assign_matrices = self.layers(self.init_queries, point_features)
        outdict_list = []

        thred = 24
        for i, q in enumerate(refined_queries_list):
            outdict_list += [self.heads(q[:,:-1,...])]
            assign_matrix = assign_matrices[i]
            assign_matrix = torch.softmax(assign_matrix, dim=2)
            outdict_list[i]['assign_matrix'] = assign_matrix
            # outdict_list[i]['exist'] = (assign_matrix.sum(1) > thred).to(torch.float32).detach()[...,None]

        if self.lm_optimization:
            outdict_list[-1] = self.lm_optimizer(outdict_list[-1], x)

        return outdict_list[-1]
