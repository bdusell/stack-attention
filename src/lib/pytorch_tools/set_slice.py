import torch

class SetSlice(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, index, y):
        ctx.index = index
        x.data[index] = y
        return x

    @staticmethod
    def backward(ctx, grad):
        if ctx.needs_input_grad[0]:
            x_grad = grad.clone()
            x_grad[ctx.index] = 0
        else:
            x_grad = None
        if ctx.needs_input_grad[2]:
            y_grad = grad[ctx.index].clone()
        else:
            y_grad = None
        return (x_grad, None, y_grad)

set_slice = SetSlice.apply
