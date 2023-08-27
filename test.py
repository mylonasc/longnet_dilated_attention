
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
        res, smKQ = model.forward(x,return_smKQ=True)

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