from typing import Union, List
import torch
import numpy as np

BIG_NEGATIVE_SCALAR = -10000.


def _make_alibi(l, m = 1/512 * 0.5, is_causal = True, device = None):
    """
    make alibi embeddings of size "l"
    Args:
      l : the size of the self-attention matrix
      m : the scaling for the embeddings 
    """
    V = []
    for i in range(l):
        v = [-i+l for i in range(i+1+l-1,i+2, -1)]
        V.append(v)
        
    V = torch.from_numpy(np.array(V).astype('float32'), device = device)
    if is_causal:
        V = torch.tril(V)
    return V * m

def _make_causal_bool(att_length, device = None):
    c = torch.tril(torch.ones(att_length,att_length,dtype=torch.bool)).to(device)
    return c

def _single_head_diagonal_block_self_attention(x_sliced, Wk, Wq,Wv, P = None, dilation = 1, norm_fact = 1., causal_mask = None):
    """
    This function computes self attention for strided/sliced inputs
    """
    
    Q = torch.einsum('ijkl,mk -> ijlm', x_sliced[:,:,:,::dilation],Wq)
    K = torch.einsum('ijkl,mk -> ijlm', x_sliced[:,:,:,::dilation],Wk)
    V = torch.einsum('ijkl,mk -> ijlm', x_sliced[:,:,:,::dilation],Wv)
    KQ = torch.einsum('ijkp, ijop -> ijko',K, Q)

    if causal_mask is not None:
        KQ = KQ * causal_mask 
        KQ = KQ + (~causal_mask * BIG_NEGATIVE_SCALAR)

    if P is not None:
        # The alibi embeddings need to take into account both segment size and 
        # dilation! 
        p_s = P.shape[0] # P is square - does not matter which dimension to chose
        segm_count = x_sliced.shape[1] # see x_in.view... section
        # I'm applying the same transformation I applied to the "x_in" sequence dimension to 
        # both dimensions of "P" to get the correct values. Possible slight perf. improvement with
        # indexing-based computation.
        p_reshaped = P.view(p_s//segm_count, segm_count,p_s//segm_count, segm_count)[::dilation,:,::dilation,:]
        p_dilated_segmented = p_reshaped[:,0,:,0] # for each segment these are the same again. It does not matter which one to take. 
        KQ = KQ + p_dilated_segmented

    #if P is not None:
    #    print(P.shape)
    #    print(KQ.shape)
    #    KQ = KQ + P[::dilation, ::dilation]
    #print(KQ.shape)

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
        pos_emb_scaling = 1.,
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

            d_v : the "value" size (can be different from "key")

            d_model : the model size (K.shape == (d_k x d_model)) - i.e., the "linear layer" 
                      matrix that implements the non-linearities internal to this layer

            dilation : how many elements are skipped from the sequence (==1 no element is skipped, ==2, every 1 element)

            segment_size : the size of the segments that the input sequence is first split (before 
                       using the dilation). 

            device   : in which device to keep the layer

            padding : ('same','valid', 'return_orig_output') with "same" padding, 
                             the length of the output is shaped to have the same
                             size as the original input length. Convenient for adding 
                             multiple heads. 

            is_causal : (True/False) whether the layer is a causal (decoder) layer

            use_layer_norm : (False) whether to normalize the layer (not necessary - can be removed eventually)

                             
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
        self.pos_emb_scaling = pos_emb_scaling


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
        wk = torch.from_numpy(np.random.randn(self.d_k,self.d_model).astype(np.float32)/np.sqrt(self.d_k)).to(self.device)
        wq = torch.from_numpy(np.random.randn(self.d_k,self.d_model).astype(np.float32)/np.sqrt(self.d_k)).to(self.device)
        wv = torch.from_numpy(np.random.randn(self.d_v,self.d_model).astype(np.float32)/np.sqrt(self.d_k)).to(self.device)

        self.Wk = torch.nn.Parameter(wk)
        self.Wq = torch.nn.Parameter(wq)
        self.Wv = torch.nn.Parameter(wv)
        self.Wk.requires_grad = True
        self.Wq.requires_grad = True
        self.Wv.requires_grad = True

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
        x_in,
        positional_embeddings_KQ = None
    ):
        """
        optionally pass positional embeddings (the layer itself does not have them!)
        """
        if not self._is_built:
            self.build(x_in)
        
        x_view = x_in.view(x_in.shape[0],x_in.shape[1]//self.segment_size, self.segment_size, x_in.shape[2]).permute(0,2,3,1)
        if positional_embeddings_KQ is not None:
            positional_embeddings_KQ *= self.pos_emb_scaling

        att, _ = _single_head_diagonal_block_self_attention(
            x_view, 
            self.Wk,
            self.Wq,
            self.Wv,
            P = positional_embeddings_KQ,
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
            pos_emb_scaling = None,
            device = 'cpu',
            is_causal = True,
            linear_out_size = None,
    ):
        super(MultiHeadDilatedAttention, self).__init__()
        """
        The multi-head dilated attention layer (similar to the LongNet paper)

        See also: 
          `SingleHeadDilatedSelfAttention`
          
        Args:
          d_k : key size
          d_v : value size
          dilation_schedule : dilated attention relevant parameter (how many samples to skip)
          segment_schedule :  dilated attention relevant parameter: how many blocks to split (segments) the input sequence.
          pos_emb_scaling : the scaling to apply to the positional embedding. A single pos. emb. matrix is computed
             and shared with all heads (and blocks is there are several blocks). This is achieved by simply passing
             the positional embedding to the "forward" function but scaling it differently for each head (ass suggested)
             by the alibi paper). 
        """
        self.d_k = d_k
        self.d_v = d_v
        self.pos_emb_scaling = pos_emb_scaling
        self.dilation_schedule = dilation_schedule

        if not isinstance(segment_schedule, list):
            segment_schedule = [segment_schedule] * len(dilation_schedule)
            
        self.segment_schedule = segment_schedule
        self._is_built = False
        self.device = device 
        self.is_causal = is_causal
        self.linear_out_size = linear_out_size
        self.dropout = torch.nn.Dropout(0.2)
        
    def build(
        self,
        x_in
    ):
        self.heads = torch.nn.ModuleList()
        seg_length = max([d * s for d,s in zip(self.segment_schedule, self.dilation_schedule)])
        if self.pos_emb_scaling is None:
            self.pos_emb_scaling = [1. for i in range(len(self.dilation_schedule))]

        if isinstance(self.pos_emb_scaling,float):
            self.pos_emb_scaling = [self.pos_emb_scaling for i in range(len(self.dilation_schedule))]

        
        for dilation, segment_size, pos_emb_scaling in zip(self.dilation_schedule, self.segment_schedule, self.pos_emb_scaling):

            head = SingleHeadDilatedSelfAttention(
                d_k = self.d_k,
                d_v = self.d_v,
                dilation= dilation,
                segment_size= segment_size,
                padding='same',
                device = self.device,
                is_causal = self.is_causal,
                use_layer_norm = False,
                pos_emb_scaling = pos_emb_scaling
            )


            head.build(x_in)
            self.heads.append(head)
            
        if self.linear_out_size:
            # not actually used in the transformer - I have the linear layer in the block 
            # for better flexibility.
            self.dense_out = torch.nn.Linear(self.heads[-1].d_model, self.linear_out_size, bias=True, device = self.device)

        # Create the alibi embeddings for this layer
        self._is_built = True

    def forward(
        self, 
        x_in,
        positional_embeddings_KQ = None,
    ):
        if not self._is_built:
            self.build(x_in)
            

        head_outputs = torch.cat([head_module(x_in, positional_embeddings_KQ = positional_embeddings_KQ) for head_module in self.heads], axis = -1)

        if self.linear_out_size is not None:
            hh = self.dropout(head_outputs)
            hh = self.dense_out(hh)
            head_outputs = torch.nn.functional.gelu(hh)

        return head_outputs
    

class DilatedTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_k = 32,
        num_heads = 16, 
        dilation_schedule = [1,2,4,8],
        segment_schedule = [1024,1024,512,512],
        device = 'cpu',
        max_seq_length = None,
        is_causal = True,
        pos_emb_scaling : Union[float, List[float]] = 1.
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

          dilation_schedule : the order of dilations for the heads to be created 
                       (1 dilation = 1 head). 

          segment_schedule  : the order of segment lengths (1 length == 1 head)

          device : the device on where to constr. the model 

          max_seq_length : (None) the maximum sequence (used to create the alibi embeddings)

          is_causal : whether to causally mask or not

          pos_emb_scaling : the scaling for the embeddings (typically different scaling
                      is used for different heads)

        """
        super(DilatedTransformerBlock, self).__init__()
        self.device = device 
        assert(len(dilation_schedule) == len(segment_schedule))

        if num_heads > len(dilation_schedule):
            segment_schedule = [segment_schedule[i % len(segment_schedule)] for i in range(num_heads)]
            dilation_schedule = [dilation_schedule[i % len(dilation_schedule)] for i in range(num_heads)]
            pos_emb_scaling = [pos_emb_scaling[i % len(pos_emb_scaling)] for i in range(num_heads)]
        self.d_k = d_k
        self.num_heads = num_heads
        self.segment_schedule = segment_schedule
        self.dilation_schedule = dilation_schedule
        self._is_built = False
        if max_seq_length is None:
            max_seq_length = max([d * s for d,s in zip(self.segment_schedule, self.dilation_schedule)])

        ## Positional embeddings:
        self.max_seq_length = max_seq_length # for the construction of "Alibi" positional embeddings.
        # self.P = _make_alibi(self.max_seq_length,1/self.max_seq_length, is_causal = is_causal)
        self.pos_emb_scaling = pos_emb_scaling
        # self.register_buffer('P',P)

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
            pos_emb_scaling = self.pos_emb_scaling,
            device = self.device,
            is_causal = True, 
            linear_out_size = self.emb_dimension
        )

        mhsa.build(x_in)
        self.mhsa_module = mhsa
        self.layer_norm_a = torch.nn.LayerNorm(normalized_shape = [x_in.shape[1],self.emb_dimension],device = self.device)
        self.layer_norm_b = torch.nn.LayerNorm(normalized_shape = [x_in.shape[1],self.emb_dimension],device = self.device)
        self.dense = torch.nn.Linear(self.emb_dimension, self.emb_dimension, device = self.device, bias = True)
        self._is_built = True
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self,x_in, positional_embedding_KQ = None):
        """
        Args:
          x_in : a pytorch tensor (input) : shape (n_seq x n_emb)
          positional_embedding_KQ (optional) : A positional embedding to be applied (additively) to the KQ attention output.
        
        """
        if not self._is_built:
            self._build(x_in)

        h = (self.mhsa_module(x_in, positional_embedding_KQ) + x_in)
        hd = self.dropout(h)
        h = self.layer_norm_a(h)
        return h
    
