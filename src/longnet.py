import torch
import numpy as np
from src.dilated_attention import DilatedTransformerBlock

def _make_alibi(l, m = 1., is_causal = True, device = None):
    """
    make alibi embeddings of size "l"
    Args:
      l : the size of the self-attention matrix
      m : the scaling for the embeddings 
    """
    V = []
    for i in range(l):
        v = [np.float32(-i+l) for i in range(i+1+l-1,i, -1)]
        V.append(v)
        
    V = torch.Tensor(np.array(V).astype('float32')).to(device)
    assert( all ( [s==l for s in V.shape]))
           
    if is_causal:
        V = torch.tril(V)
    return V * m

class LongNetDecoder(torch.nn.Module):
    def __init__(
            self,
            n_layers = 2,
            num_heads = 6,
            dilation_schedule = [1], 
            segment_schedule = [512],
            pos_emb_scaling = [0.1],
            d_k = None,
            emb_dic_len = None,
            emb_dim = 512,
            device = None,
            tokenizer = None,
            segment_length = None,
            build_lazy = False
        ):
        super(LongNetDecoder, self).__init__()
        """
        A Longnet Decoder network. Dilation/segment lists are per-head in the MHSA layers. 

        Args:
          n_layers : how many transformer block (residual) layers to create
          num_heads : number of heads per multi-head layer
          dilation_schedule : list that contains the dilation schedule (1 = no "dilation") see also 
                         `DilatedTransformerBlock` documentation.
          segment_schedule: see also `DilatedTransformerBlock` documentation
          pos_emb_scaling : scaling for the positional embedding (per head) see also `DilatedTransformerBlock`
          d_k : the key/querry size of the transformer created. It is not (necessarily) the same as Nemb/Nheads 
                and if it set to None (default) it is automatically determined by the embedding size and number of heads (as d_v).
          emb_dic_len : the output logits size
          emb_dim : the embedding dimension. It must be divisible by the number of heads.
        """
        if emb_dic_len is None:
            raise Exception("You need to specify the size of the vocabulary!")
        
        self.d_k = d_k # if None, it is determined  in the `DilatedTransformerBlock` 
        self.dilation_schedule = dilation_schedule
        self.segment_schedule = segment_schedule
        self.num_heads = num_heads
        
        # if the number of heads is larger than the "dilation_schedule" and the "segment_schedule"
        # during build the dilation and segment schedule are repeated the necessary number of times.

        self.n_layers = n_layers

        # Emb dim should be evenly dividable with the number of heads (each head 
        # returns for a different part of the embedding and then "mixed" through
        # a emb x emb matrix mult.
        self.emb_dim = emb_dim
        self.emb_dic_len = emb_dic_len
        self._is_built = False
        self.device = device 
        self.model_min_length = int(np.max([d * s for d, s in zip(self.dilation_schedule, self.segment_schedule)]))
        self.pos_emb_scaling = pos_emb_scaling
        
        self.tokenizer = tokenizer
        self.segment_length = segment_length
        if segment_length is not None:
            P = _make_alibi(
                self.segment_length,
                m = 1/self.segment_length,
                is_causal = True,
                device = self.device
            )
            self.register_buffer('P', P, persistent = True)
        else:
            raise Exception('Currently only staticaly sized LongNets can be constructed safely!')
        
        if not build_lazy:
            if self.segment_length is None:
                raise Exception('You need to define segment_length in the init of the class to build it on init!')
            t = torch.zeros(2,self.segment_length, dtype = torch.long).to(self.device)
            self._build(t)


    def _build(self, x_in):

        self.blocks = torch.nn.ModuleList(
            [
                DilatedTransformerBlock(
                    d_k = self.d_k,
                    num_heads=self.num_heads,
                    dilation_schedule = self.dilation_schedule,
                    segment_schedule = self.segment_schedule,
                    pos_emb_scaling = self.pos_emb_scaling,
                    device = self.device,
                    build_lazy = False,
                    emb_dimension= self.emb_dim,
                    segment_length = self.segment_length
                ) for i in range(self.n_layers)
            ]
        )
        
        self.model_min_length = x_in.shape[1]

        if self.segment_length is None:
            self.segment_length = self.model_min_length

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.emb_dic_len,
            embedding_dim=self.emb_dim,
            device= self.device
        )

        self.out_layer = torch.nn.Linear(
            self.emb_dim,
            self.emb_dic_len,
            bias = False,
            device = self.device
        )
        self._is_built = True
        self.model_length = x_in.shape[1]
        
    def generate(
            self,
            init_text,
            num_tokens = 100
        ):
        
        tok = self.tokenizer
        tokenized_text = tok.tokenize(init_text)
        if tokenized_text[0] == '<s>':
            tokenized_text = tokenized_text[1:]
            
        codes = tok.encode(init_text, return_tensors='pt',).to(self.device)
        for i in range(num_tokens):
            size_zeros = self.model_length - codes.shape[1]
            zeropad = torch.zeros(size_zeros, device = device, dtype = torch.long).reshape(1,-1)
            codes_for_model = torch.cat([codes, zeropad], dim = 1)
            new_codes = self.forward(codes_for_model)[0,len(codes)+1,:]
            p = torch.nn.functional.softmax(new_codes,dim = -1)
            s = torch.multinomial(p, num_samples = 1)
            codes = torch.cat([codes[:,:], s.reshape(1,-1)], axis = 1)
            text_to_print = tok.decode(codes[0])
            
        return text_to_print
    
        
    def forward(self, x_in, get_layer_outputs = False):
        """
        x_in is an iterable with integers.
        """
        if not self._is_built:
            self._build(x_in)
            
        x_curr = self.embedding(x_in)
        att_layer_outputs = []
        for m in self.blocks:
            if not get_layer_outputs:
                x_curr = m(x_curr, positional_embedding_KQ = self.P.clone())
            else:
                x_curr, outputs = m.forward(x_curr, positional_embedding_KQ = self.P.clone(), att_outputs = True)
                att_layer_outputs.append(outputs)
        
        logits = self.out_layer(x_curr)
        if get_layer_outputs:
            return logits , att_layer_outputs
        return logits