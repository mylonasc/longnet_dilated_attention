
import unittest

device = 'cpu'

class TestModules(unittest.TestCase):

    def test_single_dilated_attention(self):
        from src.dilated_attention import SingleHeadDilatedSelfAttention
        import torch
        import numpy as np
        batch_size, emb_size, seq_size = ( 2, 128, 512)
        x = torch.tensor(np.random.randn(batch_size, seq_size, emb_size).astype('float32')).to(device)/np.sqrt(emb_size)

        model = SingleHeadDilatedSelfAttention(32,32,dilation = 1, segment_count= 4)
        res = model(x)

        # return also the attention layer:
        res = model.forward(x,  return_interm_comps=True)

    def test_mhsa_construction(self):
        """
        This tests the construction strategies for a dilated MHSA 
        This is the same MHSA implemented in the LongNet paper.

        i.e., each head contains a list of dilations and segments, and the aggregation is 
        done by using the softmax denominator scaling (eq 10 in longnet paper).

        """
        from src.dilated_attention import MultiHeadDilatedAttention
        import torch

        n_heads = 3
        d_v = per_head_out_size = 12
        d_k = 23
        dilation_schedule = [1,2]
        segm_schedule = [64,128]

        ## test concatenation-head-aggregation transformer (the classic one):
        mha = MultiHeadDilatedAttention(d_k, 
                                        d_v = per_head_out_size, 
                                        d_model  = per_head_out_size * n_heads,
                                        n_heads = n_heads,
                                        segment_schedule  = segm_schedule,
                                        dilation_schedule = dilation_schedule,
                                        pos_emb_scaling=  [1.] * n_heads,
                                        device = device)
        
        t = torch.randn(10,2048, d_v * n_heads).to(device)
        att = mha.forward(t)


    def test_mhsa_constructionV1(self):
        """
        This tests the construction strategies for a dilated MHSA 
        initially implemented - that is not the same as the one described in the LongNet paper.

        In this flavor of attention each "head" has its own dilation and segment size.
        """
        from src.dilated_attention import MultiHeadDilatedAttentionV1
        import torch
        d_k, d_v = 256,654
        dilation_schedule = [1,2]
        segm_schedule = [64,64]
        pos_emb_scaling = 1.

        ## test concatenation-head-aggregation transformer (the classic one):
        mha = MultiHeadDilatedAttentionV1(
            d_k,
            d_v, 
            dilation_schedule,
            segm_schedule,
            pos_emb_scaling, 
            device=device, 
            mha_agg_strategy = 'concat'
        )
        t = torch.randn(10,2048, 123).to(device)
        att, rest = mha.forward(t)

        # test softmax denom.-weighted aggregation
        # transformer (the longnet-type - eq 10):
        # (note this is not the same as LongNet!)
        mha = MultiHeadDilatedAttentionV1(
            d_k,
            d_v, 
            dilation_schedule,
            segm_schedule,
            pos_emb_scaling, 
            device=device, 
            mha_agg_strategy = 'softmax_denom'
        )
        t = torch.randn(10,2048, 123).to(device)
        att, rest = mha.forward(t)

    def test_longnet_construction(self):
        from src.longnet import LongNetDecoder
        SEGMENT_LENGTH = 1024
        BATCH_LEN = 4
        N_LAYERS = 8
        dilation_schedule = [1, 2, 4  ,  8, 16 , 32  , 64 ]*2
        segment_sizes     = [4, 8, 16 , 32, 64 , 128 , 256]*2
        segment_schedule =  [SEGMENT_LENGTH//k for k in segment_sizes]

        num_heads = len(dilation_schedule)
        emb_per_head = 16
        emb_dim = num_heads * emb_per_head

        # See ALiBI paper "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (https://arxiv.org/abs/2108.12409)
        s = num_heads/len(dilation_schedule)
        pos_emb_scalings = [1/2**(h/s) for h in range(1,num_heads+1)]

        model = LongNetDecoder(
            d_k = 32,
            n_layers = N_LAYERS,
            num_heads = num_heads,
            dilation_schedule = dilation_schedule,
            segment_schedule = segment_schedule,
            segment_length=SEGMENT_LENGTH,
            emb_dic_len = 1234,
            emb_dim=emb_dim,
            pos_emb_scaling = pos_emb_scalings,
            device = device,
            build_lazy=False,
        )


if __name__ == '__main__':
    unittest.main()