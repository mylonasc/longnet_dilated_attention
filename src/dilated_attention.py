from typing import Union, List
import torch
import numpy as np

BIG_NEGATIVE_SCALAR = -10000.

def _make_causal_bool(att_length, device = None):
    c = torch.tril(torch.ones(att_length,att_length,dtype=torch.bool)).to(device)
    return c

def _single_head_diagonal_block_self_attention(x_sliced, Wk, Wq,Wv, dilation = 1, norm_fact = 1., causal_mask = None):
    """
    This function computes self attention for strided/sliced inputs
    """
    
    Q = torch.einsum('ijkl,mk -> ijlm', x_sliced[:,:,:,::dilation],Wq)
    K = torch.einsum('ijkl,mk -> ijlm', x_sliced[:,:,:,::dilation],Wk)
    V = torch.einsum('ijkl,mk -> ijlm', x_sliced[:,:,:,::dilation],Wv)
    KQ = torch.einsum('ijkp, ijop -> ijko',K, Q)
    
    if causal_mask is not None:
        KQ = KQ * causal_mask + (~causal_mask * BIG_NEGATIVE_SCALAR)

    smKQ = torch.nn.functional.softmax(KQ * norm_fact, dim = 2)
    att = torch.einsum('ijld, ijlm-> ijdm',smKQ, V)
    return att ,(K, Q, smKQ)
    
class SingleHeadDilatedSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_k = 32,
        d_v = None,
        d_model = None,
        dilation = 1,
        segment_size = 16,
        device = 'cpu',
        padding = 'same',
        is_causal = True,
        use_layer_norm = False
    ):
        super(SingleHeadDilatedSelfAttention, self).__init__()
        """Breaks a sequence to segments and computes dilated single-head attention. 

        This module returns tensors of smaller size than the ones entered, except if dilation==1

        Restrictions:
        -------------
         * [segment_size] x [dilation_rate] must be smaller than the input provided.
         * The input must be an integer multiple of [segment_size]*[dilation_rate]. 
           (this restriction can be removed by appropriate padding).
        
        Does some dangerous tricks with reshaping and einsums.
        
        Args:
            d_k : the size of the projection 
            d_model : the model size (K, Q == d_k x d_model)
            dilation : how many 
            output_padding : ('same','valid', 'return_orig_output') with "same" padding, the length of the output 
                             is shaped to have the same size as the original input length.
                             Convenient for adding multiple heads.
        """

        self.d_k = d_k
        if d_v is None:
            d_v = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.dilation = dilation
        self.segment_size = segment_size
        self.device = device 
        self._is_built = False
        self.norm_fact = torch.scalar_tensor(1./np.sqrt(self.d_k)).to(device)
        self.padding = padding
        self.is_causal = is_causal
        self.causal_mask = None
        self.use_layer_norm = use_layer_norm

    def build_from_in_shape(
        self,
        in_shape
    ):
        
        """
        Builds the necessary weights if not already available.
        Assuming B x T x E tensor ( (batch dim) x (time dim) x (embedding dim) )
        """
        assert(len(in_shape) == 3)
        # if (in_shape[1] % (self.segment_size * self.dilation)) != 0:
        #     raise Exception('Currently, input length and S = [segment_size]*[dilation_rate] are supported only when they are integer multiples. \n ' + \
        #                     'Moreover, the input tensor should be at least 1 x S.\n' + \
        #                     'The provided input dimension is %i whereas S = %i x %i = %i'%(
        #                             in_shape[1],
        #                             self.segment_size, 
        #                             self.dilation,self.segment_size *self.dilation
        #                         )
        #                     )
            
        if self.d_model is None:
            self.d_model = in_shape[2]

        self.Wk = torch.Tensor(np.random.randn(self.d_k,self.d_model).astype(np.float32)/np.sqrt(self.d_k), ).to(self.device)
        self.Wq = torch.Tensor(np.random.randn(self.d_k,self.d_model).astype(np.float32)/np.sqrt(self.d_k), ).to(self.device)
        self.Wv = torch.Tensor(np.random.randn(self.d_v,self.d_model).astype(np.float32)/np.sqrt(self.d_k), ).to(self.device)
        self._is_built = True
        
        if self.is_causal:
            self.causal_mask = _make_causal_bool(in_shape[1]//self.segment_size, device =self.device)[::self.dilation, ::self.dilation]

        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm([self.d_v], device = self.device)
        

    def build(self, x_in):
        self.build_from_in_shape(x_in.shape)
            
    def get_head_weights(self):
        return self.Wk, self.Wq, self.Wv
    
    def output_shape(
        self,
        x_in = None
    ):
        if self._is_built:
            dim1 = None
            dim2 = None
            if x_in is not None:
                dim1 = x_in.shape[0]
                dim2 = x_in.shape[1]
            
            if self.padding == 'same':
                return (dim1, dim2, self.d_v)
                
            if self.padding =='no_padding':
                return (dim1, dim2//self.segment_size, self.d_v)
                
        raise Exception("Not implemented!")
        
    
    def forward(
        self,
        x_in
    ):
        if not self._is_built:
            self.build(x_in)
        
        x_view = x_in.view(x_in.shape[0],x_in.shape[1]//self.segment_size, self.segment_size, x_in.shape[2]).permute(0,2,3,1)

        att, _ = _single_head_diagonal_block_self_attention(
            x_view, 
            self.Wk,
            self.Wq,
            self.Wv,
            dilation = self.dilation,
            norm_fact= self.norm_fact,
            causal_mask = self.causal_mask
        )
        
        if self.padding == 'same':
            res = torch.zeros(x_in.shape[0], x_in.shape[1], self.d_v, device = self.device)
            res[:,::self.dilation,:] = att.reshape(att.shape[0], att.shape[1]*att.shape[2], att.shape[3])
            
        if self.padding == 'no_padding':
            res = att.reshape(att.shape[0], att.shape[1]*att.shape[2], att.shape[3])
            
        if self.padding == 'return_orig_output':
            res = att            
        if self.use_layer_norm:
            res = self.layer_norm(res)
        return res

class MultiHeadDilatedAttention(torch.nn.Module):
    def __init__(
            self,
            d_k = 256,
            d_v = None,
            dilation_schedule = [1,2,4,8],
            segment_schedule : Union[List, int] = 128, 
            device = 'cpu',
            is_causal = True,
            linear_out_size = None
    ):
        super(MultiHeadDilatedAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.dilation_schedule = dilation_schedule
        if not isinstance(segment_schedule, list):
            segment_schedule = [segment_schedule] * len(dilation_schedule)
            
        self.segment_schedule = segment_schedule
        self._is_built = False
        self.device = device 
        self.is_causal = is_causal
        self.linear_out_size = linear_out_size
        
    def build(
        self,
        x_in
    ):
        self.heads = torch.nn.ModuleList()
        
        for dilation, segment_size in zip(self.dilation_schedule, self.segment_schedule):

            head = SingleHeadDilatedSelfAttention(
                d_k = self.d_k,
                d_v = self.d_v,
                dilation= dilation,
                segment_size= segment_size,
                padding='same',
                device = self.device,
                is_causal = self.is_causal,
                use_layer_norm = False
            )


            head.build(x_in)
            # out_shape = head.output_shape(x_in)
            self.heads.append(head)
            
        if self.linear_out_size:
            # not actually used.
            self.dense_out = torch.nn.Linear(self.heads[-1].d_model, self.linear_out_size, bias=True, device = self.device)
        
        self._is_built = True

    def forward(
        self, 
        x_in
    ):
        if not self._is_built:
            self.build(x_in)
            
        head_outputs = torch.cat([head_module(x_in) for head_module in self.heads], axis = -1)

        if self.linear_out_size is not None:
            # not used
            head_outputs = torch.nn.functional.gelu(self.dense_out(head_outputs))

        return head_outputs
    

class DilatedTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_k = 32,
        num_heads = 16, 
        dilation_schedule = [1,2,4,8],
        segment_schedule = [1024,1024,512,512],
        device = 'cpu',
    ):
        super(DilatedTransformerBlock, self).__init__()
        """

        The heads' last dimension is concatenated to be added again back to the input tensor.
        Therefore we need this to be compatible with the input (residual connection - avoiding linear projection)

        In order to achieve that, the number of heads for each MultiheadDilatedAttention layer should be the same
        as [embedding_size] // [num_heads] (exactly divided).

        Dilation schedule and segment schedules are repeated in case they are smaller in length (i.e., )
        
        For the created MultiheadDilatedAttention blocks, we have:
            d_model = [embedding size] // [num_heads]

        Args:
          num_layers : the number of MultiHeadAttentionBlocks MHDA
          num_heads  : the number of heads per MHDA.
          dilation_schedule : the order of dilations for the heads to be created (1 dilation = 1 head). 
          segment_schedule  : the order of segment lengths (1 length == 1 head)

        """
        super(DilatedTransformerBlock, self).__init__()
        self.device = device 
        assert(len(dilation_schedule) == len(segment_schedule))

        if num_heads > len(dilation_schedule):
            segment_schedule = [segment_schedule[i % len(segment_schedule)] for i in range(num_heads)]
            dilation_schedule = [dilation_schedule[i % len(dilation_schedule)] for i in range(num_heads)]
        self.d_k = d_k
        self.num_heads = num_heads
        self.segment_schedule = segment_schedule
        self.dilation_schedule = dilation_schedule
        self._is_built = False

    def _build(self, x_in):
        """
        Builds the layers given input of a speciffic size
        """
        self.emb_dimension = x_in.shape[-1]
        # self.d_model = self.emb_dimension // self.num_heads
        assert(self.emb_dimension % self.num_heads == 0)
        if (x_in.shape[2] != self.emb_dimension):
            print(x_in.shape, self.emb_dimension)
            raise Exception('The input size does not seem to be correct! in: %i expected (emb_dimension): %i '%(x_in.shape[2], self.emb_dimension))
        self.per_head_out_size = self.emb_dimension // self.num_heads
        mhsa = MultiHeadDilatedAttention(
            d_k = self.d_k,
            d_v = self.per_head_out_size,
            segment_schedule = self.segment_schedule,
            dilation_schedule = self.dilation_schedule,
            device = self.device,
            is_causal = True, 
            linear_out_size = None
        )
        mhsa.build(x_in)
        self.mhsa_module = mhsa
        self.layer_norm_a = torch.nn.LayerNorm(normalized_shape = [x_in.shape[1],self.emb_dimension],device = self.device)
        self.layer_norm_b = torch.nn.LayerNorm(normalized_shape = [x_in.shape[1],self.emb_dimension],device = self.device)
        self.dense = torch.nn.Linear(self.emb_dimension, self.emb_dimension, device = self.device, bias = True)
        self._is_built = True

    def forward(self,x_in):
        if not self._is_built:
            self._build(x_in)

        h = (self.mhsa_module(x_in) + x_in)
        h = self.layer_norm_a(h)
        return h
    