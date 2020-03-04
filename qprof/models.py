"""
models
======

This module provides an implementation of quantile regression neural networks (QRNNs)
using the pytorch deep learning framework.
"""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from tqdm.auto import tqdm

################################################################################
# The quantile loss
################################################################################

class QuantileLoss:
    r"""
    The quantile loss function

    This function object implements the quantile loss defined as


    .. math::

        \mathcal{L}(y_\text{pred}, y_\text{true}) =
        \begin{cases}
        \tau \cdot |y_\text{pred} - y_\text{true}| & , y_\text{pred} < y_\text{true} \\
        (1 - \tau) \cdot |y_\text{pred} - y_\text{true}| & , \text{otherwise}
        \end{cases}


    as a training criterion for the training of neural networks. The loss criterion
    expects a vector :math:`\mathbf{y}_\tau` of predicted quantiles and the observed
    value :math:`y`. The loss for a single training sample is computed by summing the losses
    corresponding to each quantiles. The loss for a batch of training samples is
    computed by taking the mean over all samples in the batch.
    """
    def __init__(self, quantiles):
        """
        Create an instance of the quantile loss function with the given quantiles.

        Arguments:
            quantiles: Array or iterable containing the quantiles to be estimated.
        """
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

    def __call__(self, y_pred, y_true):
        """
        Compute the mean quantile loss for given inputs.

        Arguments:
            y_pred: N-tensor containing the predicted quantiles along the last
                dimension
            y_true: (N-1)-tensor containing the true y values corresponding to
                the predictions in y_pred

        Returns:
            The mean quantile loss.
        """
        l = torch.full_like(y_true, 0.0)
        y_pred = y_pred.view(-1, self.n_quantiles)
        for i, q in enumerate(self.quantiles):
            y = y_pred[:, i]
            dy = y - y_true
            dya = torch.abs(y - y_true)
            l += torch.where(dy <= 0.0,
                             q * dya,
                             (1.0 - q) * dya)
        return torch.mean(l, axis = 0)

################################################################################
# QRNN
################################################################################

class QRNN:
    """
    Quantile regression neural network (QRNN)

    This class implements QRNNs as a fully-connected network with
    a given number of layers.
    """
    def __init__(self,
                 input_dimension,
                 quantiles,
                 depth = 3,
                 width = 128,
                 activation = nn.ReLU):
        """
        Arguments:
            input_dimension(int): The number of input features.
            quantiles(array): Array of the quantiles to predict.
            depth(int): Number of layers of the network.
            width(int): The number of neurons in the inner layers of the network.
            activation: The activation function to use for the inner layers of the network.
        """
        self.input_dimension = input_dimension
        self.quantiles = np.array(quantiles)
        self.depth = depth
        self.width = width
        self.activation = nn.ReLU

        n_quantiles = len(quantiles)

        self.main = nn.Sequential()

        self.main.add_module("fc_0", nn.Linear(input_dimension, width))
        self.main.add_module("act_0", activation())
        for i in range(1, depth - 1):
            self.main.add_module("fc_{}".format(i), nn.Linear(width, width))
            self.main.add_module("act_{}".format(i), activation())
        self.main.add_module("fc_{}".format(depth - 1), nn.Linear(width, n_quantiles))

        self.optimizer = optim.Adam(self.main.parameters())
        self.criterion = QuantileLoss(self.quantiles)

        self.training_errors = []
        self.validation_errors = []

    def _make_adversarial_samples(self, x, y, eps):
        self.main.zero_grad()
        x.requires_grad = True
        y_pred = self.main(x)
        c = self.criterion(y_pred, y)
        c.backward()
        x_adv = x.detach() + eps * torch.sign(x.grad.detach())
        return x_adv

    def train(self,
              training_data,
              validation_data,
              n_epochs = 1,
              adversarial_training = False,
              eps_adv = 1e-6,
              lr = 1e-3,
              momentum = 0.9):

        """
        Train the network.

        This trains the network for the given number of epochs using the provided
        training and validation data.

        If desired, the training can be augmented using adversarial training. In this
        case the network is additionally trained with an adversarial batch of examples
        in each step of the training.

        Arguments:
            training_data: pytorch dataloader providing the training data
            validation_data: pytorch dataloader providing the validation data
            n_epochs: the number of epochs to train the network for
            adversarial_training: whether or not to use adversarial training
            eps_adv: The scaling factor to use for adversarial training.
        """

        self.main.train()
        for i in range(n_epochs):

            err = 0.0
            n = 0
            iterator = tqdm(enumerate(training_data), total = len(training_data))
            for j, (x, y) in iterator:

                self.optimizer.zero_grad()
                y_pred = self.main(x)
                c = self.criterion(y_pred, y)
                c.backward()
                self.optimizer.step()

                err += c.item() * x.size()[0]
                n += x.size()[0]

                if adversarial_training:
                    self.optimizer.zero_grad()
                    x_adv = self._make_adversarial_samples(x, y, eps_adv)
                    y_pred = self.main(x)
                    c = self.criterion(y_pred, y)
                    c.backward()
                    self.optimizer.step()

                if j % 100:
                    iterator.set_postfix({"Training errror" : err / n})

            # Save training error
            self.training_errors.append(err / n)

            val_err = 0.0
            n = 0

            for x, y in validation_data:
                y_pred = self.main(x)
                c = self.criterion(y_pred, y)

                val_err += c.item() * x.size()[0]
                n += x.size()[0]

            self.validation_errors.append(val_err / n)
        self.main.eval()

    def predict(self, x):
        """
        Predict quantiles for given input.

        Args:
            x: 2D tensor containing the inputs for which for which to
                predict the quantiles.

        Returns:
            tensor: 2D tensor containing the predicted quantiles along
                the last dimension.
        """
        return self.main(x)

    def calibration(self, data):
        """
        Computes the calibration of the predictions from the neural network.

        Arguments:
            data: torch dataloader object providing the data for which to compute
                the calibration.

        Returns:
            (intervals, frequencies): Tuple containing the confidence intervals and
                corresponding observed frequencies.
        """
        n_intervals = self.quantiles.size // 2
        qs = self.quantiles
        intervals = np.array([q_r - q_l for (q_l, q_r) in zip(qs, reversed(qs))])[:n_intervals]
        counts = np.zeros(n_intervals)

        total = 0.0

        iterator = tqdm(data)
        for x, y in iterator:
            y_pred = self.predict(x)

            for i in range(n_intervals):
                l = y_pred[:, i]
                r = y_pred[:, -(i + 1)]
                counts[i] += np.logical_and(y >= l, y < r).sum()

            total += x.size()[0]

        return intervals[::-1], (counts / total)[::-1]


    def save(self, path):
        """
        Save QRNN to file.

        Arguments:
            The path in which to store the QRNN.
        """
        torch.save({"input_dimension" : self.input_dimension,
                    "quantiles" : self.quantiles,
                    "width" : self.width,
                    "depth" : self.depth,
                    "activation" : self.activation,
                    "network_state" : self.main.state_dict(),
                    "optimizer_state" : self.optimizer.state_dict()},
                    path)

    @staticmethod
    def load(self, path):
        """
        Load QRNN from file.

        Arguments:
            path: Path of the file where the QRNN was stored.
        """
        state = torch.load(path)
        keys = ["input_dimension", "quantiles", "depth", "width", "activation"]
        qrnn = QRNN(*[state[k] for k in keys])
        qrnn.main.load_state_dict["network_state"]
        qrnn.optimizer.load_state_dict["optimizer_state"]


class CNet:
    """
    Quantile regression neural network (QRNN)

    This class implements QRNNs as a fully-connected network with
    a given number of layers.
    """
    def __init__(self,
                 output_dim,
                 latent_dim = 5,
                 depth = 3,
                 width = 128):
        """
        Arguments:
            input_dimension(int): The number of input features.
            quantiles(array): Array of the quantiles to predict.
            depth(int): Number of layers of the network.
            width(int): The number of neurons in the inner layers of the network.
            activation: The activation function to use for the inner layers of the network.
        """
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.depth = depth
        self.width = width

        # Generator
        self.generator = nn.Sequential()
        self.generator.add_module("fc_0", nn.Linear(1 + self.latent_dim, width))
        self.generator.add_module("bn_0", nn.BatchNorm1d(width))
        self.generator.add_module("act_0", nn.LeakyReLU())
        for i in range(1, depth - 1):
            self.generator.add_module("fc_{}".format(i), nn.Linear(width, width))
            self.generator.add_module("bn_{}".format(i), nn.BatchNorm1d(width))
            self.generator.add_module("act_{}".format(i), nn.LeakyReLU())
        self.generator.add_module("fc_{}".format(depth - 1), nn.Linear(width,
                                                                       self.output_dim))
        self.generator.add_module("act_{}".format(i), nn.Tanh())

        beta1 = 0.5
        lr_dis = 0.00002
        self.optimizer_gen = optim.Adam(self.generator.parameters(),
                                        lr=lr_dis,
                                        betas=(beta1, 0.999))

        # Discriminator
        self.discriminator = nn.Sequential()
        self.discriminator.add_module("fc_0", nn.Linear(1 + self.output_dim, width))
        self.discriminator.add_module("act_0", nn.LeakyReLU())
        for i in range(1, depth - 1):
            self.discriminator.add_module("fc_{}".format(i), nn.Linear(width, width))
            self.discriminator.add_module("bn_{}".format(i), nn.BatchNorm1d(width))
            self.discriminator.add_module("act_{}".format(i), nn.LeakyReLU())
        self.discriminator.add_module("fc_{}".format(depth - 1), nn.Linear(width, 1))

        lr_gen = 0.0002
        self.optimizer_dis = optim.Adam(self.discriminator.parameters(),
                                        lr=lr_dis,
                                        betas=(beta1, 0.999))

        self.generator_losses = []
        self.discriminator_losses = []

        if torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
        dev = torch.device("cpu")
        self.device = dev
        self.criterion = nn.BCEWithLogitsLoss()

    def generate(self, y, n = 1):
        z = torch.randn(n, self.latent_dim, device = self.device)
        z = torch.cat((z, y.reshape(-1, 1)), 1)
        return self.generator.forward(z)

    def train(self,
              dataloader):

        self.discriminator.to(self.device)
        self.discriminator.train()
        self.generator.to(self.device)
        self.generator.train()

        self.discriminator.to(self.device)
        self.generator.to(self.device)

        real_label = 0.9
        fake_label = 0
        iters = 0

        for i, (x, y) in enumerate(dataloader, 0):

            self.discriminator.zero_grad()

            x = x.to(self.device)
            y = y.to(self.device)
            x = torch.cat((x, y.reshape((-1, 1))), 1)

            real = x
            #real = torch.clamp(real, -1.0, 1.0)

            bs = real.size(0)
            label = torch.full((bs,), real_label, device = self.device)

            # forward pass real batch through d
            output = self.discriminator(real).view(-1)
            err_dis_real = self.criterion(output, label)
            err_dis_real.backward()
            d_x = nn.Sigmoid()(output.detach()).mean().item()

            ## train with all-fake batch
            fake = self.generate(y, x.shape[0])
            fake = torch.cat((fake, y.reshape(-1, 1)), 1)
            fake = torch.clamp(fake, -1.0, 1.0)

            output = self.discriminator(fake.detach()).view(-1)
            label.fill_(fake_label)
            err_dis_fake = self.criterion(output, label)
            err_dis_fake.backward()
            d_g_z1 = nn.Sigmoid()(output.detach()).mean().item()

            # add the gradients from the all-real and all-fake batches
            err_dis = err_dis_real + err_dis_fake
            self.optimizer_dis.step()

            #
            # train generator
            #

            self.generator.zero_grad()
            label.fill_(real_label)
            # since we just updated d, perform another forward pass
            # of all-fake batch through d
            output = self.discriminator(fake).view(-1)
            errg = self.criterion(output, label)
            errg.backward()

            d_g_z2 = nn.Sigmoid()(output.detach()).mean().item()
            self.optimizer_gen.step()

            # output training stats
            if i % 50 == 0:
                print('[%d/%d]\tloss_d: %.4f\tloss_g: %.4f\td(x): %.4f\td(g(z)): %.4f / %.4f'
                    % (i, len(dataloader),
                        err_dis.item(), errg.item(), d_x, d_g_z1, d_g_z2))

            # save losses for plotting later
            self.generator_losses.append(errg.item())
            self.discriminator_losses.append(err_dis.item())
            iters += 1


    def save(self, path):
        torch.save({"latent_dim" : self.latent_dim,
                    "output_dim" : self.output_dim,
                    "discriminator_state" : self.discriminator.state_dict(),
                    "generator_state" : self.generator.state_dict(),
                    "discriminator_losses" : self.discriminator_losses,
                    "generator_losses" : self.generator_losses,
                    "depth" : self.depth,
                    "width" : self.width}, path)

        self.output_dimension = output_dimension
        self.latent_dim = latent_dim
        self.depth = depth
        self.width = width

    @staticmethod
    def load(self, path):
        """
        Load QRNN from file.

        Arguments:
            path: Path of the file where the QRNN was stored.
        """
        state = torch.load(path)

        keys = ["output_dim", "latent_dim",  "depth", "width"]
        args = [state[k] for k in keys]
        gan = CNet(*args)
        gan.generator.load_state_dict(state["generator_state"])
        gan.discriminator.load_state_dict(state["discriminator_state"])
        gan.discriminator_losses = state["discriminator_losses"]
        gan.generator_losses = state["generator_losses"]
        return gan

class VAE(nn.Module):
    def __init__(self,
                 output_dim = 13,
                 latent_dim = 5,
                 width = 128):

        super(VAE, self).__init__()

        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.output_dim, width)
        self.fc21 = nn.Linear(width, width)
        self.fc22 = nn.Linear(width, width)
        self.fc23 = nn.Linear(width, width)
        self.fc31 = nn.Linear(width, self.latent_dim)
        self.fc32 = nn.Linear(width, self.latent_dim)

        self.fc4 = nn.Linear(self.latent_dim + 1, width)
        self.fc51 = nn.Linear(width, width)
        self.fc52 = nn.Linear(width, width)
        self.fc53 = nn.Linear(width, width)
        self.fc6 = nn.Linear(width, output_dim)

        if torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
        dev = torch.device("cpu")
        self.device = dev

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc21(h1))
        h2 = F.relu(self.fc22(h2))
        h2 = F.relu(self.fc23(h2))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        z = torch.cat((z, y.reshape(-1, 1)), 1)
        h3 = F.relu(self.fc4(z))
        h3 = F.relu(self.fc51(h3))
        h3 = F.relu(self.fc52(h3))
        h3 = F.relu(self.fc53(h3))
        return torch.sigmoid(self.fc6(h3))

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, self.output_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.output_dim), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE, KLD

    def train(self, train_loader, beta = 1.0):
        nn.Module.train(self)
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.forward(x, y)
            bce, kld  = self.loss(recon_batch, x, mu, logvar)
            loss = bce + kld
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % 1000 == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f} + {:.6f}'.format(
                    batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    bce.item() / len(x), kld.item() / len(x)))

        print('====> Epoch: Average loss: {:.4f}'.format(
            train_loss / len(train_loader.dataset)))
