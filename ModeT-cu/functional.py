import torch
from torch.autograd import Function
from modet import modet_fw, modet_bw

class ModeTFunction(Function):
    @staticmethod
    def forward(ctx, query, key, rpb):
        query = query.contiguous()
        key = key.contiguous()
        attn = modet_fw(query, key, rpb)
        ctx.save_for_backward(query, key)
        ctx.bias = rpb is not None
        return attn

    @staticmethod
    # @custom_bwd
    def backward(ctx, grad_out):
        outputs = modet_bw(
            grad_out.contiguous(),
            ctx.saved_variables[0],
            ctx.saved_variables[1],
            ctx.bias,
        )
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb

def modetqkrpb_cu(query, key, rpb):
    return ModeTFunction.apply(query, key, rpb)