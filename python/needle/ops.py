"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


def reduce_as(input_shape: list, output_shape: list, tensor: Tensor):
    input_shape = [1] * (len(output_shape) - len(input_shape)) + input_shape
    reduce_axes = []
    for i in range(len(output_shape)):
        if input_shape[i] == 1:
            reduce_axes.append(i)
            continue
    if len(reduce_axes) == 0:
        return tensor
    return reshape(summation(tensor, axes=tuple(reduce_axes)), input_shape)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        assert(a.shape == b.shape)
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        assert (a.shape == b.shape)
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return self.scalar * (a ** (self.scalar-1)) * out_grad


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        assert (a.shape == b.shape)
        return a / b

    def gradient(self, out_grad, node: Tensor):
        a, b = node.inputs
        return out_grad / b, -1 * out_grad * a / (b ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a / self.scalar

    def gradient(self, out_grad, node):
        a: Tensor = node.inputs[0]
        return divide_scalar(out_grad , self.scalar)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        new_axes = [i for i in range(len(a.shape))]
        if self.axes is None:  # defaults to the last two axes
            new_axes[-1], new_axes[-2] = new_axes[-2], new_axes[-1]
        else:
            new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
        return a.permute(tuple(new_axes))

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return a.reshape(self.shape).compact()

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape).compact()


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray:
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = list(node.inputs[0].shape)
        output_shape = list(out_grad.shape)
        return reduce_as(input_shape, output_shape, out_grad)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return a.sum(axis=self.axes)

    def gradient(self, out_grad, node):
        a: Tensor = node.inputs[0]
        shape = tuple([1] * len(a.shape))
        if self.axes is not None:
            shape = list(a.shape)
            if isinstance(self.axes, tuple):
                for axis in self.axes:
                    shape[axis] = 1
            else:
                shape[self.axes] = 1
        out_grad = reshape(out_grad, shape)
        return broadcast_to(out_grad, a.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        a_grad = out_grad @ transpose(b)
        a_grad = reduce_as(list(a.shape), list(a_grad.shape), a_grad)
        b_grad = transpose(a) @ out_grad
        b_grad = reduce_as(list(b.shape), list(b_grad.shape), b_grad)
        return a_grad, b_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return a * -1

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return -1 * out_grad * Tensor(array_api.ones(a.shape))


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad / a


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray) -> NDArray:
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * exp(a)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a) -> NDArray:
        return a.maximum(0)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        if isinstance(a, Tensor):
            a = a.realize_cached_data()
        return out_grad * Tensor(array_api.where(a <= 0, 0, 1))


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes, )
        else:
            self.axes = axes

    def compute(self, Z):
        max_z = Z.max(axis=self.axes)
        expand_max_z = max_z
        if self.axes is None:
            expand_max_z = expand_max_z.reshape(tuple([1] * len(Z.shape)))
            expand_max_z = expand_max_z.broadcast_to(Z.shape)
        else:
            new_shape = list(Z.shape)
            for i in self.axes:
                new_shape[i] = 1
            expand_max_z = max_z.reshape(tuple(new_shape)).broadcast_to(Z.shape)
        Z = Z - expand_max_z
        Z = array_api.exp(Z)
        Z = Z.sum(axis=self.axes)
        Z = array_api.log(Z)
        return Z + max_z

    def gradient(self, out_grad, node):
        # TODO: WHAT THE FUCK IS THIS?
        # All I know is this is a trick:
        # https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        Z = node.inputs[0]
        if self.axes:
            shape = [1] * len(Z.shape)
            j = 0
            for i in range(len(shape)):
                if isinstance(self.axes, tuple) and i not in self.axes:
                    shape[i] = node.shape[j]
                    j += 1
                if isinstance(self.axes, int) and i != self.axes:
                    shape[i] = node.shape[j]
                    j += 1
            node_new = node.reshape(shape)
            grad_new = out_grad.reshape(shape)
        else:
            node_new = node
            grad_new = out_grad
        return grad_new.broadcast_to(Z.shape) * exp(Z - node_new.broadcast_to(Z.shape))


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        return (array_api.exp(a) - array_api.exp(-1*a)) / (array_api.exp(a) + array_api.exp(-1*a))

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (1 - tanh(a) ** 2) * out_grad


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: List[Tensor]):
        stack_shape = list(args[0].shape)
        stack_shape = stack_shape[:self.axis] + [len(args)] + stack_shape[self.axis:]
        stack_shape = tuple(stack_shape)
        out = init.zeros(*stack_shape, device=args[0].device).cached_data
        slices = [slice(None)] * len(out.shape)
        for i, arg in enumerate(args):
            slices[self.axis] = slice(i, i+1, 1)
            out[tuple(slices)] = arg
        return out

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        outs = []
        cnt = A.shape[self.axis]
        slices = [slice(None)] * len(A.shape)
        for i in range(cnt):
            slices[self.axis] = slice(i, i + 1, 1)
            input_tensor = A[tuple(slices)]
            outs.append(input_tensor.sum(axis=(self.axis,)))
        return tuple(outs)

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



