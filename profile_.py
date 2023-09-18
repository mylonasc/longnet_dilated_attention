import torch
from src.dilated_attention import DilatedTransformerBlock, MultiHeadDilatedAttention, SingleHeadDilatedSelfAttentionV2
#from src.longnet import LongNetDecoder
from torch.profiler import profile, record_function, ProfilerActivity

def _run_profiling_mhsa(init_params):
    init_params['d_v'] = init_params['d_model']
    # del init_params['n_heads']
    x = torch.randn(1,init_params['max_seq_length'], init_params['d_k']).to(device)
    del init_params['max_seq_length']
    init_params['pos_emb_scaling'] = None

    device = 'cpu'
    dilt = MultiHeadDilatedAttention(
        **init_params,
        is_causal=True,
        device = device
    )

    def eeval(x):
        return dilt.forward(x)
    
    # just to init.
    eeval(x)
    with profile(activities=[ProfilerActivity.CPU],record_shapes = True, profile_memory=True) as prof:
        eeval(x)
    
    res = sorted([(k.key, k.cpu_memory_usage/1024/1024) for k in prof.key_averages()], key = lambda x : -x[1])
    return init_params, res

def _run_profiling_dtb(init_params):
    device = 'cpu'
    dilt = DilatedTransformerBlock(
        **init_params,
        build_lazy=True,
        is_causal=True,
        device = device
    )

    def eeval(x):
        return dilt.forward(x)
    
    x = torch.randn(1,init_params['segm_len'], init_params['d_k']).to(device)
    # just to init.
    eeval(x)

    with profile(activities=[ProfilerActivity.CPU],record_shapes = True, profile_memory=True) as prof:
        eeval(x)
    
    res = sorted([(k.key, k.cpu_memory_usage/1024/1024) for k in prof.key_averages()], key = lambda x : -x[1])
    return init_params, res    

def _run_profiling_sha(init_params):
    device = 'cpu'

    
    x = torch.randn(1,init_params['max_seq_length'], init_params['d_k']).to(device)
    del init_params['max_seq_length']
    model = SingleHeadDilatedSelfAttentionV2(**init_params, is_causal = True)
    
    
    # just to init.
    def eeval(x):
        return model.forward(x)

    eeval(x)

    with profile(activities=[ProfilerActivity.CPU],record_shapes = True, profile_memory=True) as prof:
        eeval(x)
    
    res = sorted([(k.key, k.cpu_memory_usage/1024/1024) for k in prof.key_averages()], key = lambda x : -x[1])
    return init_params, res

if __name__ == "__main__":
    
    LAYER = 'sha' #'dtb' # 'mhsa'

    for num_heads in [2,4,8,16]:
        d_k = 64
        # num_heads = 16
        emb_per_head = d_k // num_heads
        d_model = num_heads*emb_per_head
        segm_len = 8*64
        dilation_schedule = [1,2,4,8]
        segm_schedule = [32,64,64,64]
        
        init_params_block = {
            'd_k' : d_k, 
            'n_heads' : num_heads,
            'd_model' : d_model,
            'max_seq_length' : segm_len,
            'dilation_schedule' : dilation_schedule,
            'segment_schedule' : segm_schedule
        }

        init_params_block['pos_emb_scaling'] = [1.] * num_heads
        
        if LAYER=='sha':
            del init_params_block['n_heads']
            del init_params_block['d_model']
            params, res = _run_profiling_sha(init_params_block)

        if LAYER == 'mhsa':
            params, res = _run_profiling_mhsa(init_params_block)
        if LAYER == 'dtb':
            params, res = _run_profiling_dtb(init_params_block)

        import json
        from hashlib import md5
        import os

        s = json.dumps({'params' : params, 'res' : res,'layer' : LAYER})
        s1 = str([init_params_block[k] for k in sorted(init_params_block.keys())])
        s2 = str([res[0] for k in res])
        fname = md5( ( s1 + s2).encode()).hexdigest() + '.json'
        with open(os.path.join('profiling_results', fname),'w') as f:
            f.write(s)