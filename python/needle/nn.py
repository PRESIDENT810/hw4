"""The module.
"""
from typing import List, Callable, Any, Optional
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    in_features: int
    out_features: int
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights using Kaiming initialization
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, shape=(in_features, out_features),
                                 device=device, requires_grad=True, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, shape=(out_features, 1), device=device,
                                                       requires_grad=True, dtype=dtype).reshape((1, out_features)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        y = X @ self.weight
        if self.bias is None:
            return y
        return y + self.bias.broadcast_to(y.shape)



class Flatten(Module):
    def forward(self, X):
        shape = X.shape
        if len(shape) == 1:
            return X
        total = 1
        for dim in X.shape:
            total *= dim
        return ops.reshape(X, (shape[0], int(total / shape[0])))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (1+ops.exp(-1*x)) ** -1


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for mod in self.modules:
            x = mod(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        loss = ops.logsumexp(logits, axes=1)
        y_onehot = logits * init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        return (loss - y_onehot.sum(axes=1)).sum() / logits.shape[0]


class BatchNorm1d(Module):
    weight: Parameter
    bias: Parameter
    running_mean: ops.NDArray
    running_var: ops.NDArray

    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(1, dim, requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, requires_grad=True, device=device, dtype=dtype))
        self.running_mean: Tensor = init.zeros(1, dim, requires_grad=True, device=device, dtype=dtype).reshape((dim, ))
        self.running_var: Tensor = init.ones(1, dim, requires_grad=True, device=device, dtype=dtype).reshape((dim, ))

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.summation(x, axes=0)
        mean = mean / x.shape[0]
        # BatchNorm uses the running estimates of mean and variance instead of batch statistics at test time
        mean = mean.reshape((1, mean.shape[0]))
        var = x - ops.broadcast_to(mean, x.shape)
        var = ops.power_scalar(var, 2)
        var = ops.summation(var, 0)
        var = var.reshape((1, var.shape[0]))
        var = var / x.shape[0]
        running_mean: Tensor = self.running_mean.reshape((1, self.dim))
        running_var: Tensor = self.running_var.reshape((1, self.dim))

        if self.training:  # Training phase
            running_mean = (running_mean * (1 - self.momentum) + mean * self.momentum)
            running_var = (running_var * (1 - self.momentum) + var * self.momentum)
        else:  # Testing phase
            mean = self.running_mean
            var = self.running_var
        self.running_mean = running_mean.reshape((self.dim, )).detach()
        self.running_var = running_var.reshape((self.dim, )).detach()
        var = var + self.eps
        var = ops.power_scalar(var, 0.5)
        y = x - ops.broadcast_to(mean, x.shape)
        y = y / ops.broadcast_to(var, x.shape)
        y = y * self.weight.broadcast_to(x.shape)
        y = y + self.bias.broadcast_to(y.shape)
        return y

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    weight: Parameter
    bias: Parameter
    dim: int
    eps: float

    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(1, dim, requires_grad=True, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, dim, requires_grad=True, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.summation(x, axes=1)
        mean = mean / x.shape[1]
        mean = mean.reshape((mean.shape[0], 1))
        mean = ops.broadcast_to(mean, x.shape)
        var = x - mean
        var = ops.power_scalar(var, 2)
        var = ops.summation(var, 1)
        var = var.reshape((var.shape[0], 1))
        var = var / x.shape[1]
        var = var + self.eps
        var = ops.power_scalar(var, 0.5)
        var = ops.broadcast_to(var, x.shape)
        y = x - mean
        y = y / var
        y = y * self.weight.broadcast_to(x.shape)
        y = y + self.bias.broadcast_to(y.shape)
        return y


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:  # Testing phase
            return x
        else:  # Training phase
            mask = init.randb(*x.shape, p=(1 - self.p)) / (1 - self.p)
            return x * mask



class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        kernel_shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = Parameter(
            init.kaiming_uniform(
                in_channels*kernel_size*kernel_size,
                out_channels*kernel_size*kernel_size,
                shape=kernel_shape, requires_grad=True, device=device, dtype=dtype
            )
        )
        if bias:
            bound = 1 / ((in_channels * kernel_size**2)**0.5)
            self.bias = Parameter(init.rand(out_channels, low=-1*bound, high=bound, requires_grad=True, device=device, dtype=dtype))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: NCHW -> NHWC
        x = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(x, self.weight, self.stride, padding=self.kernel_size//2)
        if self.bias is not None:
            out = out + self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(out.shape)
        return out.transpose((2, 3)).transpose((1, 2))

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = ops.tanh if nonlinearity == "tanh" else ops.relu

        k = 1 / hidden_size
        bound = k**0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-1*bound, high=bound, device=device,
                                        dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-1 * bound, high=bound, device=device,
                                        dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-1 * bound, high=bound, device=device,
                                               dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-1 * bound, high=bound, device=device,
                                               dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None


    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor containing the next hidden state
            for each element in the batch.
        """
        bs = X.shape[0]
        out = X @ self.W_ih
        if self.bias_ih is not None:
            out += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(out.shape)
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=True)
        out += h @ self.W_hh
        if self.bias_hh is not None:
            out += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(out.shape)
        out = self.nonlinearity(out)
        return out


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise, the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        self.rnn_cells: list[RNNCell] = []
        self.num_layers = num_layers
        for i in range(num_layers):
            if i == 0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))


    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len = X.shape[0]
        last_layer = list(ops.split(X, axis=0))
        h0 = list(ops.split(h0, axis=0)) if h0 is not None else [None] * self.num_layers
        hn = []
        for l in range(self.num_layers):
            last_cell = h0[l]
            for t in range(seq_len):
                last_cell = self.rnn_cells[l](last_layer[t], last_cell)
                last_layer[t] = last_cell
            hn.append(last_cell)
        return ops.stack(last_layer, axis=0), ops.stack(hn, axis=0)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid = Sigmoid()

        k = 1 / hidden_size
        bound = k ** 0.5
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-1 * bound, high=bound, device=device,
                                        dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-1 * bound, high=bound, device=device,
                                        dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-1 * bound, high=bound, device=device,
                                               dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-1 * bound, high=bound, device=device,
                                               dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        bs = X.shape[0]
        h0, c0 = h if h is not None else (None, None)

        ifgo = X @ self.W_ih
        if self.bias_ih is not None:
            ifgo += self.bias_ih.reshape((1, 4*self.hidden_size)).broadcast_to(ifgo.shape)
        if h0 is None:
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype, requires_grad=True)
        ifgo += h0 @ self.W_hh
        if self.bias_hh is not None:
            ifgo += self.bias_hh.reshape((1, 4*self.hidden_size)).broadcast_to(ifgo.shape)

        ifgo_split = ops.split(ifgo, 1)
        i = ops.stack([ifgo_split[i] for i in range(0, self.hidden_size)], 1)
        f = ops.stack([ifgo_split[i] for i in range(self.hidden_size, 2 * self.hidden_size)], 1)
        g = ops.stack([ifgo_split[i] for i in range(2 * self.hidden_size, 3 * self.hidden_size)], 1)
        o = ops.stack([ifgo_split[i] for i in range(3 * self.hidden_size, 4 * self.hidden_size)], 1)

        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = ops.tanh(g)
        o = self.sigmoid(o)

        c = i * g
        if c0 is not None:
            c += f * c0
        h = o * ops.tanh(c)

        return h, c



class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        super().__init__()
        self.lstm_cells: list[LSTMCell] = []
        self.num_layers = num_layers
        for i in range(num_layers):
            if i == 0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        h0, c0 = h if h is not None else (None, None)

        seq_len = X.shape[0]
        last_layer = list(ops.split(X, axis=0))
        h0 = list(ops.split(h0, axis=0)) if h0 is not None else [None] * self.num_layers
        c0 = list(ops.split(c0, axis=0)) if c0 is not None else [None] * self.num_layers
        hn = []
        cn = []
        for l in range(self.num_layers):
            last_h0 = h0[l]
            last_c0 = c0[l]
            for t in range(seq_len):
                last_h0, last_c0 = self.lstm_cells[l](last_layer[t], (last_h0, last_c0))
                last_layer[t] = last_h0
            hn.append(last_h0)
            cn.append(last_c0)
        return ops.stack(last_layer, axis=0), (ops.stack(hn, axis=0), ops.stack(cn, axis=0))


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, requires_grad=True, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        seq_len, bs = x.shape
        x = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype, requires_grad=True)
        x = x.reshape((seq_len*bs, self.num_embeddings))
        out = x @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))
