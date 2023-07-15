import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    losses = []
    wrong = 0
    total = 0
    if opt is None:
        model.eval()
    else:
        model.train()

    for batch in dataloader:
        X, y = batch
        output = model(X)
        loss = loss_fn(output, y)
        losses.append(loss.cached_data)
        # Update weights
        if opt is not None:
            loss.backward()
            opt.step()
        wrong += (y.numpy() != output.numpy().argmax(axis=1)).sum()
        total += y.shape[0]
    return wrong / total, np.average(np.array(losses))


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    train_acc, train_avg_loss = None, None
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    for e in range(n_epochs):
        train_acc, train_avg_loss = epoch_general_cifar10(dataloader, model, loss_fn)
        # print("train_acc={}, train_avg_loss={}".format(train_acc, train_avg_loss))

    return train_acc, train_avg_loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    test_acc, test_avg_loss = None, None

    test_acc, test_avg_loss = epoch_general_cifar10(dataloader, model, loss_fn())

    return test_acc, test_avg_loss



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    correct, loss_sum, n_step, n_samplers = 0., 0., 0, 0
    if opt is None:
        model.eval()
    else:
        model.train()

    nbatch, batch_size = data.shape
    h = None
    for i in range(0, data.shape[0]-1, seq_len):
        # batch_data.shape = (seq_len, batch_size), batch_target.shape = (seq_len * batch_size, )
        batch_data, batch_target = ndl.data.get_batch(data, i, seq_len, device, dtype)
        output, h = model(batch_data, h)
        if isinstance(h, tuple):
            h = (h[0].detach(), h[1].detach())
        else:
            h = h.detach()

        loss = loss_fn(output, batch_target)
        loss_sum += loss.numpy() * batch_target.shape[0]
        # Update weights
        if opt is not None:
            loss.backward()
            opt.step()
        correct += (batch_target.numpy() == output.numpy().argmax(axis=1)).sum()
        n_samplers += batch_target.shape[0]
    return correct / n_samplers, loss_sum / n_samplers



def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    train_acc, train_avg_loss = None, None
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    for e in range(n_epochs):
        train_acc, train_avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), opt, clip, device, dtype)
        # print("train_acc={}, train_avg_loss={}".format(train_acc, train_avg_loss))

    return train_acc, train_avg_loss


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    test_acc, test_avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), None, None, device, dtype)

    return test_acc, test_avg_loss


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
