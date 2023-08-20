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
    def __init__(self,
                 d_k = 256,
                 d_v = None, 
                 dilation = 1,
                 segment_size = 16,
                 device = 'cpu',
                 padding = 'same',
                 is_causal = True,
                 normalize_output = True,
                 use_layer_norm = True
        ):
        super(SingleHeadDilatedSelfAttention, self).__init__()
        """Breaks a sequence to segments and computes dilated single-head attention. 

        This module returns tensors of smaller size than the ones entered, except if dilation==1
        
        Does some dangerous tricks with reshaping and einsums.
        
        Args:
            d_k : the size of the projection 
            d_model : the embedding size 
            dilation : how many 
            output_padding : ('same','valid', 'return_orig_output') with "same" padding, the length of the output 
                             is shaped to have the same size as the original input length.
                             Convenient for adding multiple heads.
        """
        self.d_k = d_k
        if d_v is None:
            d_v = d_k
        self.d_v = d_v
        self.dilation = dilation
        self.segment_size = segment_size
        self.device = device 
        self._is_built = False
        self.norm_fact = torch.scalar_tensor(1./np.sqrt(self.d_k)).to(device)
        self.padding = padding
        
        self.is_causal = is_causal
        self.causal_mask = None
        self.use_layer_norm = use_layer_norm

    def build(self, x_in):
        """
        Builds the necessary weights if not already available.
        """
        self.d_model = x_in.shape[2]
        self.Wk = torch.Tensor(np.random.randn(self.d_k,self.d_model).astype(np.float32)/np.sqrt(self.d_k), ).to(self.device)
        self.Wq = torch.Tensor(np.random.randn(self.d_k,self.d_model).astype(np.float32)/np.sqrt(self.d_k), ).to(self.device)
        self.Wv = torch.Tensor(np.random.randn(self.d_v,self.d_model).astype(np.float32)/np.sqrt(self.d_k), ).to(self.device)
        self._is_built = True
        
        if self.is_causal:
            self.causal_mask = _make_causal_bool(x_in.shape[1]//self.segment_size, device =self.device)[::self.dilation, ::self.dilation]

        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm([self.d_v], device = self.device)
            
    
    def output_shape(self, x_in = None):
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
        
    
    def forward(self, x_in):
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
            segment_sizes : Union[List, int] = 128, 
            device = 'cpu',
            is_causal = True,
            linear_out = True
    ):
        super(MultiHeadDilatedAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.dilation_schedule = dilation_schedule
        if not isinstance(segment_sizes, list):
            segment_sizes = [segment_sizes] * len(dilation_schedule)
            
        self.segment_sizes = segment_sizes
        
        self._is_built = False
        self.device = device 
        self.is_causal = is_causal
        self.linear_out = linear_out 
        
        
    def build(
        self,
        x_in
    ):
        self.heads = torch.nn.ModuleList()
        for dilation, segment_size in zip(self.dilation_schedule, self.segment_sizes):
            head = SingleHeadDilatedSelfAttention(
                d_k = self.d_k,
                d_v = self.d_v,
                dilation= dilation,
                segment_size= segment_size,
                padding='same',
                device = self.device,
                is_causal = self.is_causal
            )
            head.build(x_in)
            out_shape = head.output_shape(x_in)
            self.heads.append(head)
            
        if self.linear_out:
            self.dense_out = torch.nn.Linear(self.d_k, self.heads[-1].d_model, bias=True, device = self.device)
        # self._accum_cache = torch.zeros(out_shape).to(self.device)
        
        self._is_built = True

    def forward(
        self, 
        x_in
    ):
        if not self._is_built:
            self.build(x_in)
            
        head_outputs = torch.stack([head_module(x_in) for head_module in self.heads],axis = 0)
        res = torch.sum(head_outputs,0)

        if self.linear_out:
            res = torch.nn.functional.gelu(self.dense_out(res))

        return res
