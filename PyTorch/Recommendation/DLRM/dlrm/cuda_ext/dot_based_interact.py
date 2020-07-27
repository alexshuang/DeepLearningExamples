import torch
from torch.autograd import Function
from apex import amp

is_rocm_pytorch = False
if torch.__version__ >= '1.5':
    from torch.utils.cpp_extension import ROCM_HOME
    is_rocm_pytorch = True \
            if ((torch.version.hip is not None) and (ROCM_HOME is not None)) \
            else False

if not is_rocm_pytorch and torch.cuda.get_device_capability()[0] >= 8:
    print('Using the Ampere-optimized dot interaction kernels')
    from dlrm.cuda_ext import interaction_ampere as interaction
else:
    print('Using the Volta-optimized dot interaction kernels')
    from dlrm.cuda_ext import interaction_volta as interaction


class DotBasedInteract(Function):
    """ Forward and Backward paths of cuda extension for dot-based feature interact."""

    @staticmethod
    @amp.half_function
    def forward(ctx, input, bottom_mlp_output):
        output = interaction.dotBasedInteractFwd(input, bottom_mlp_output)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    @amp.half_function
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad, mlp_grad = interaction.dotBasedInteractBwd(input, grad_output)
        return grad, mlp_grad


dotBasedInteract = DotBasedInteract.apply
