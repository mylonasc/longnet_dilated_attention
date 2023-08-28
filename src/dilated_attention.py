from typing import Union, List, Any
import torch
import numpy as np

DROPOUT_RATE = 0.2
DTYPE = torch.float32


@torch.compile()
def _softmax_dim3_get_normalization( x):
    """
    Computes softmax along dimension 3 and returns
    the denominator of the softmax (normalization)
    """
    maxes = torch.max(x, 3, keepdim=True)[0]
    x_exp = torch.exp(x-maxes)
    x_exp_sum = torch.sum(x_exp, 3, keepdim=True)
    return x_exp/x_exp_sum, x_exp_sum

def _make_causal_bool(att_length, device = None):
    c = torch.tril(torch.ones(att_length,att_length,dtype=torch.bool)).to(device)
    return c


def _single_head_dilated_segmented_self_attention(
        x_sliced,
        Wk,
        Wq,
        Wv,
        P = None,
        dilation = 1,
        norm_fact = 1.,
        causal_mask = None
    ):

    # Dimensions:
    # x    : [batch] [segment] [emb] [t]
    # Wk   : [emb]   [d]
    # K    : [batch] [segment] [d] [t] 


    Q  = torch.einsum('ijkl, mk   ->  ijlm', x_sliced[:,:,:,::dilation],Wq)
    K  = torch.einsum('ijkl, mk   ->  ijlm', x_sliced[:,:,:,::dilation],Wk)
    V  = torch.einsum('ijkl, mk   ->  ijlm', x_sliced[:,:,:,::dilation],Wv)

    KQ = torch.einsum('ijkl, ijml ->  ijkm',K, Q)
    if causal_mask is not None:
        KQ = KQ * causal_mask
        KQ[:,:,~causal_mask] = float('-inf')
    
    if P is not None:
        p_s = x_sliced.shape[-1]
        p_reshaped = P[:p_s,:p_s][::dilation,::dilation]
        KQ = KQ + p_reshaped
    
    # smKQ = torch.nn.functional.softmax(KQ * norm_fact, dim = 3)
    sm_norm = None
    smKQ, sm_norm = _softmax_dim3_get_normalization(KQ * norm_fact)
    
    att = torch.einsum('ijnm ,ijnk-> ijmk',smKQ, V)
    return att, ({'K' : K, 'Q' : Q, 'smKQ' : smKQ, 'sm_norm' : sm_norm})
    
class SingleHeadDilatedSelfAttention(torch.nn.Module):
    def __init__(
            self,
            d_k = None,
            d_v = None,
            d_model = None,
            dilation = 1,
            segment_count = 16,
            pos_emb_scaling = 1.,
            device = 'cpu',
            padding = 'same',
            is_causal = True,
            use_v2 = False
        ):
        super(SingleHeadDilatedSelfAttention, self).__init__()
        """Breaks a sequence to segments and computes dilated single-head attention. 

        This module returns tensors of smaller size than the ones entered, except if dilation==1
        
        Does some dangerous tricks with reshaping and einsums.
        
        Args:
            d_k : the size of the projection 

            d_v : the "value" size (can be different from "key")

            d_model : the model size (K.shape == (d_k x d_model)) - i.e., the "linear layer" 
                      matrix that implements the non-linearities internal to this layer

            dilation : how many elements are skipped from the sequence (==1 no element is skipped, ==2, every 1 element)

            segment_count : the size of the segments that the input sequence is first split (before 
                       using the dilation). 

            device   : in which device to keep the layer

            padding : ('same','valid', 'return_orig_output') with "same" padding, 
                             the length of the output is shaped to have the same
                             size as the original input length. Convenient for adding 
                             multiple heads. 

            is_causal : (True/False) whether the layer is a causal (decoder) layer

            use_v2 : a different version of the operation where the KQV are first computed and then "dilated". 
               the operation is not exactly equivallent, and it may also lead to different computational and
               fitting performance. 
                             
        """

        self.d_k = d_k
        if d_v is None:
            d_v = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.dilation = dilation
        self.segment_count = segment_count
        self.device = device 
        self._is_built = False
        self.norm_fact = torch.scalar_tensor(1./np.sqrt(self.d_k)).to(device)
        self.padding = padding
        self.is_causal = is_causal
        self.causal_mask = None
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
            
        if self.d_model is None:
            self.d_model = in_shape[2]

        def _get_param(k,m, dtype = DTYPE):
            v = torch.randn((k,m) , dtype = DTYPE, device = self.device) / torch.math.sqrt(k)
            p = torch.nn.Parameter(v)
            p.requires_grad = True
            return p
        
        self.Wk = _get_param(self.d_k, self.d_model)
        self.Wq = _get_param(self.d_k,self.d_model)
        self.Wv = _get_param(self.d_v,self.d_model)
        self._is_built = True
        if self.is_causal:
            self.causal_mask = _make_causal_bool(in_shape[1]//self.segment_count, device =self.device)[::self.dilation, ::self.dilation]
        self.dropout = torch.nn.Dropout(DROPOUT_RATE)

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
                return (dim1, dim2//self.segment_count, self.d_v)
                
        raise Exception("Not implemented!")
    
    def get_same_padded_dilated(self, x_in,att):
        res = torch.zeros(x_in.shape[0], x_in.shape[1], self.d_v, device = self.device)
        res[:,::self.dilation,:] = att.reshape(att.shape[0], att.shape[1]*att.shape[2], att.shape[3])
        return res
    

    def forward(
            self,
            x_in,
            positional_embeddings_KQ = None,
            return_interm_comps = False
        ):
        """
        optionally pass positional embeddings (the layer itself does not have them!)
        
        Args:
            x_in : the input to be computed
            positional_embeddings_KQ: 
            return_smKQ : (false) whether to return the normalized key-querry matrix. 
            return_sm_norm : (true) whether to return the denominator of the softmax (used for downstream scaling)
        """
        if not self._is_built:
            self.build(x_in)
        
        x_view = x_in.view(x_in.shape[0],x_in.shape[1]//self.segment_count, self.segment_count, x_in.shape[2]).permute(0,2,3,1)
        if positional_embeddings_KQ is not None:
            positional_embeddings_KQ *= self.pos_emb_scaling

        att, out_dict = _single_head_dilated_segmented_self_attention(
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
            res = self.get_same_padded_dilated(x_in, att)
            
        if self.padding == 'no_padding':
            res = att.reshape(att.shape[0], att.shape[1]*att.shape[2], att.shape[3])
            
        if self.padding == 'return_orig_output' or self.padding is None:
            res = att

        if return_interm_comps:
            return res, out_dict
        else:
            return res

class MultiHeadDilatedAttention(torch.nn.Module):
    def __init__(
                self,
                d_k = None,
                d_v = None,
                dilation_schedule = [1,2,4,8],
                segment_schedule : Union[List, int] = 128, 
                pos_emb_scaling = None,
                device = 'cpu',
                is_causal = True,
                linear_out = True,
                mha_agg_strategy = 'concat' # softmax_denom
        ):
        super(MultiHeadDilatedAttention, self).__init__()
        """
        The multi-head dilated attention layer (similar to the LongNet paper)

        See also: 
          `SingleHeadDilatedSelfAttention`
          
        Args:
          d_v : value size
          d_k : key size (can be ommitted if d_v is provided and num_heads is clear from the length of dilation_schedule)
          dilation_schedule : dilated attention relevant parameter (how many samples to skip)
          segment_schedule :  dilated attention relevant parameter: how many blocks to split (segments) the input sequence.
          pos_emb_scaling : the scaling to apply to the positional embedding. A single pos. emb. matrix is computed
             and shared with all heads (and blocks is there are several blocks). This is achieved by simply passing
             the positional embedding to the "forward" function but scaling it differently for each head (ass suggested)
             by the ALiBI paper). 
          mha_agg_strategy : ('concat','softmax_denom') hether to aggregate through concatenation or through the softmax  weighted 
             aggregation technique proposed in the longnet paper (eq 10). This also changes the default behavior with respect 
             to the determination of the output size d_v (if it is not explicitly provided). 
        """
        self.pos_emb_scaling = pos_emb_scaling
        self.dilation_schedule = dilation_schedule

        assert(mha_agg_strategy in ['concat', 'softmax_denom'])
        self.mha_agg_strategy = mha_agg_strategy
        if not isinstance(segment_schedule, list):
            print("!! setting the segment schedule to the \
                   same size as the dilation schedule \
                   provided. For more control on the \
                   architecture, please provide both \
                   dilation and segment schedules.")
            segment_schedule = [segment_schedule] * len(dilation_schedule)

        if len(segment_schedule) != len(dilation_schedule):
            raise Exception("Different segment and dilation schedule lists were provided! These must be the same. Aborting init. ")

        self.num_heads = len(self.dilation_schedule)

        self.d_k = d_k
        self.d_v = d_v
        if self.d_k is None: 
            # this is for the key and querry matrices.
            # In multi-head attention, d_v
            # is simply = self.d_k * self.num_heads.
            # This allows determining the d_k:
            if self.mha_agg_strategy == 'concat':
                self.d_v = self.d_k // self.num_heads

            if self.mha_agg_strategy == 'softmax_denom':
                self.d_v = self.d_k
            print("determining automatically key and query matrix sizes (internal transformer parameters) through d_v: %i, d_k = d_v/num_heads: %i"%(self.d_v, self.d_k))
                    
        self.segment_schedule = segment_schedule
        self._is_built = False
        self.device = device 
        self.is_causal = is_causal
        self.linear_out = linear_out
        self.dropout = torch.nn.Dropout(DROPOUT_RATE)
        
    def build(
            self,
            x_in
        ):
        self.heads = torch.nn.ModuleList()
        if self.pos_emb_scaling is None:
            print("setting automatically all pos embedding scalings to 1. ")
            self.pos_emb_scaling = [1. for i in range(len(self.dilation_schedule))]

        if isinstance(self.pos_emb_scaling,float):
            print("setting all pos embedding scalings to %2.3f "%(self.pos_emb_scaling))
            self.pos_emb_scaling = [self.pos_emb_scaling for i in range(len(self.dilation_schedule))]

        for dilation, segment_count, pos_emb_scaling in zip(self.dilation_schedule, self.segment_schedule, self.pos_emb_scaling):
            head = SingleHeadDilatedSelfAttention(
                d_k = self.d_k,
                d_v = self.d_v,
                dilation= dilation,
                segment_count= segment_count,
                padding='return_orig_output',
                device = self.device,
                is_causal = self.is_causal,
                pos_emb_scaling = pos_emb_scaling
            )
            head.build(x_in)
            self.heads.append(head)
            
        if self.linear_out:
            if self.mha_agg_strategy == 'concat':
                out_shape = sum([h.Wv.shape[0] for h in  self.heads])
            else:
                # assuming that all the heads have the same value size:
                out_shape = self.heads[0].Wv.shape[0]

            self.dense_out = torch.nn.Linear(
                out_shape, 
                x_in.shape[-1],
                bias=False, 
                device = self.device
            )
        # Create the alibi embeddings for this layer
        self._is_built = True

    def forward(
            self, 
            x_in,
            positional_embeddings_KQ = None,
            return_attention_outputs = False,
            use_sm_norm = True,
            return_att_out = True

        ):
        if not self._is_built:
            self.build(x_in)

        head_outputs_list = []
        
        rest_outputs = {}

        def _conditional_append_out_or_make_list(return_x_data : bool, key : str, dat : Any ):
            if return_x_data:
                if key not in rest_outputs:
                    rest_outputs[key] = []
                rest_outputs[key].append(dat)
        
        s_i_O = 0
        sum_s_i = 0
        # Take note of eqn 10 of the LongNet paper:
        # see https://arxiv.org/pdf/2307.02486v2.pdf (eq 10)
        # In longnet instead of concatenating the outputs of 
        # different heads, they are weighted proportionally 
        # according to the denom. of the softmax of the
        # used in the attention matrix.
        
        for head_module in self.heads:
            att_raw, rest_dict = head_module.forward(
                x_in,
                positional_embeddings_KQ = positional_embeddings_KQ,
                return_interm_comps = True,
                
            )
            _conditional_append_out_or_make_list(use_sm_norm, 'sm_norm', rest_dict['sm_norm'])
            _conditional_append_out_or_make_list(return_att_out, 'att', att_raw)
            att = head_module.get_same_padded_dilated(x_in, att_raw)
            if self.mha_agg_strategy == 'softmax_denom':
                s_i = head_module.get_same_padded_dilated(x_in, rest_dict['sm_norm'])
                s_i_O += att * s_i
                sum_s_i += s_i
            head_outputs_list.append(att)

        if self.mha_agg_strategy  == 'softmax_denom':
            a_i_O = s_i_O / sum_s_i
            head_outputs = a_i_O

        if self.mha_agg_strategy == 'concat':
            head_outputs = torch.cat(head_outputs_list, dim = -1)

        if self.linear_out:
            hh = self.dense_out(head_outputs)
            hh = self.dropout(hh)
        else:
            hh = self.dropout(head_outputs)
        
        if len(rest_outputs) > 0:
            return hh, rest_outputs
        return hh
    
class DilatedTransformerBlock(torch.nn.Module):
    def __init__(
            self,
            d_k = None,
            num_heads = 16, 
            dilation_schedule = [1,2,4,8],
            segment_schedule = [1024,1024,512,512],
            mha_agg_strategy = 'softmax_denom',
            max_seq_length = None,
            pos_emb_scaling : Union[float, List[float]] = 1.,
            use_dropout = True,
            out_linear_size = None,
            build_lazy = False,
            emb_dimension = None,
            segment_length = None,
            device = 'cpu',
            is_causal = True
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
            d_k : the internal size of the key and querry matrices. If it is not provided it is 
                  determined from d_v and num_heads.

            num_heads : The number of heads for the MHSA layers

            dilation_schedule : arrays that determine the dilation schedules.

            segment_schedule   :arrays that determine the segment size schedules.

            mha_agg_strategy : (concat/softmax_denom) - the aggregation for the MHA layer (concat or weighted add as in longnet paper.)

            max_seq_length  : the maximum length of the sequence (it helps with efficiency to keep this static)
            
            pos_emb_scaling : (same size as the heads) - scaling for the positional embeddings (can be different for each head as in ALiBI)

            use_dropout : (always... or else.)

            out_linear_size : (None) Final projection of outputs. in the original "Attention is all you need" paper, this is 4 x (V out size). 
                            The 4xemb size is also adopted here.
                            Note:
                            It may be possible to get away with smaller sizes (Note that this does not affect the embedding size output!)
                            There is literature replacing this FFNN with micture of experts - potentially check this out in the future. 
                            This is a very memory-heavy parameter.
            
            build_lazy : whether to actually build the layer or build it when the first input is encountered (laziness is good for experimentation)

            emb_dimension : The dimension of the (concatenated) embedding.

            segment_length : The (static) length of the segment processed. This is good for efficiency (and for automated JIT compiling later on).

            device  : where to keep this layer

            is_causal : whether to use causal masking or not.
        """
        super(DilatedTransformerBlock, self).__init__()

        self.d_k = d_k
        self.device = device 
        self.mha_agg_strategy = mha_agg_strategy

        assert(len(dilation_schedule) == len(segment_schedule))
                    
        self.out_linear_size = out_linear_size

        if num_heads > len(dilation_schedule):
            segment_schedule = [segment_schedule[i % len(segment_schedule)] for i in range(num_heads)]
            dilation_schedule = [dilation_schedule[i % len(dilation_schedule)] for i in range(num_heads)]
            pos_emb_scaling = [pos_emb_scaling[i % len(pos_emb_scaling)] for i in range(num_heads)]

        self.num_heads = num_heads
        self.segment_schedule = segment_schedule
        self.dilation_schedule = dilation_schedule
        self._is_built = False
        self.use_dropout = use_dropout
        self.is_causal = is_causal

        if max_seq_length is None:
            max_seq_length = max([d * s for d,s in zip(self.segment_schedule, self.dilation_schedule)])

        self.out_linear_size = out_linear_size

        ## Positional embeddings:
        self.max_seq_length = max_seq_length # for the construction of "Alibi" positional embeddings.
        self.pos_emb_scaling = pos_emb_scaling
        self.build_lazy = build_lazy
        self.emb_dimension = emb_dimension
        self.segment_length = segment_length

        if self.emb_dimension is not None and (~self.build_lazy):
            if self.segment_length is None:
                raise Exception("You also need to provide the segment_length parameter - it is None! (when not building lazily).")
            t = torch.rand(2, self.segment_length, self.emb_dimension, dtype= DTYPE).to(self.device)
            self._build(t)


    def _build(self, x_in):
        """
        Builds the layers given input of a speciffic size
        """
        if self.emb_dimension is None:
            self.emb_dimension = x_in.shape[-1]

        
         # self.d_model = self.emb_dimension // self.num_heads
        assert(self.emb_dimension % self.num_heads == 0)
        if (x_in.shape[2] != self.emb_dimension):
            raise Exception('The input size does not seem to be correct! in: %i expected (emb_dimension): %i '%(x_in.shape[2], self.emb_dimension))
        
        self.per_head_out_size = self.emb_dimension // self.num_heads

        # if d_k is None, the constructor of the following
        # will make a d_k same as the one implied by the size of the 
        # embedding and the number of heads (split equally among 
        # heads)
        mhsa = MultiHeadDilatedAttention(
            d_k = self.d_k,
            d_v = self.per_head_out_size,
            segment_schedule = self.segment_schedule,
            dilation_schedule = self.dilation_schedule,
            pos_emb_scaling = self.pos_emb_scaling,
            device = self.device,
            is_causal = self.is_causal,
            mha_agg_strategy=self.mha_agg_strategy
        )

        mhsa.build(x_in)
        self.mhsa_module = mhsa

        if self.out_linear_size is None:
            self.out_linear_size = self.emb_dimension * 4
        
        self.out_dense = torch.nn.Sequential(
            torch.nn.Linear( self.emb_dimension , self.out_linear_size, bias =True, device = self.device),
            torch.nn.ReLU(),
            torch.nn.Linear(self.out_linear_size, self.emb_dimension , bias = True, device = self.device),
            torch.nn.Dropout(DROPOUT_RATE)
        )

        # self.layer_norm_a = torch.nn.LayerNorm(normalized_shape = [x_in.shape[1],self.emb_dimension],device = self.device)
        self.layer_norm_a = torch.nn.LayerNorm(normalized_shape = [self.emb_dimension],device = self.device)
        self.layer_norm_b = torch.nn.LayerNorm(normalized_shape = [self.emb_dimension],device = self.device)
        self._is_built = True
        if self.use_dropout:
            self.dropout = torch.nn.Dropout(DROPOUT_RATE)

    def forward(self,x_in, positional_embedding_KQ = None, att_outputs = False):
        """
        Args:
          x_in : a pytorch tensor (input) : shape (n_seq x n_emb)
          positional_embedding_KQ (optional) : A positional embedding to be applied (additively) to the KQ attention output.
        """
        if not self._is_built:
            self._build(x_in)
        h = self.layer_norm_a(x_in)
        if att_outputs:
            mhsa_out, smKQ = self.mhsa_module(
                h, positional_embedding_KQ, return_attention_outputs = True
            )
        else:
            mhsa_out,_ = self.mhsa_module.forward(
                h,
                positional_embedding_KQ, 
                return_attention_outputs = False
            )
        h = self.layer_norm_b(mhsa_out + x_in)
        h = self.out_dense(h)

        if att_outputs:
            return h, smKQ
        
        return h
    
