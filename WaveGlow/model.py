import chainer
import numpy

from .modules import Flow


def _squeeze(x, squeeze_factor):
    batchsize, channel, length = x.shape
    x = x.reshape(
        (batchsize, channel, length // squeeze_factor, squeeze_factor))
    x = x.transpose((0, 1, 3, 2))
    x = x.reshape(
        (batchsize, channel * squeeze_factor, length // squeeze_factor))
    return x


def _unsqueeze(x, squeeze_factor):
    batchsize, channel, length = x.shape
    x = x.reshape(
        (batchsize, channel // squeeze_factor, squeeze_factor, length))
    x = x.transpose((0, 1, 3, 2))
    x = x.reshape(
        (batchsize, channel // squeeze_factor, length * squeeze_factor))
    return x


class Glow(chainer.Chain):
    def __init__(
            self, hop_length=256, n_mels=80, input_channel=1,
            squeeze_factor=8, n_flows=12, n_layers=8,
            wn_channel=512, early_every=4, early_size=2, var=0.5):
        super(Glow, self).__init__()
        self.input_channel = input_channel
        self.squeeze_factor = squeeze_factor
        self.n_flows = n_flows
        self.early_every = early_every
        self.early_size = early_size
        self.var = float(var)
        self.ln_var = float(numpy.log(var))
        flows = chainer.ChainList()
        for i in range(n_flows):
            flows.add_link(Flow(
                input_channel * squeeze_factor -
                early_size * (i // early_every),
                n_mels * squeeze_factor, n_layers, wn_channel))
        with self.init_scope():
            self.encoder = chainer.links.Deconvolution1D(
                n_mels, n_mels, hop_length * 4, hop_length,
                pad=hop_length * 3 // 2)
            self.flows = flows

    def __call__(self, x, condition):
        z, gaussian_nll, sum_log_s, sum_log_det_W = self._forward(x, condition)
        nll = gaussian_nll - sum_log_s - sum_log_det_W + float(numpy.log(2 ** 16))
        loss = chainer.functions.mean(z * z / (2 * self.var)) - \
            sum_log_s - sum_log_det_W
        chainer.reporter.report(
            {
                'nll': nll, 'log_s': sum_log_s,
                'log_det_W': sum_log_det_W, 'loss': loss}, self)
        return loss

    def _forward(self, x, condition):
        condition = self.encoder(condition)
        x = _squeeze(x, self.squeeze_factor)
        condition = _squeeze(condition, self.squeeze_factor)
        sum_log_s = 0
        sum_log_det_W = 0
        outputs = []
        for i, flow in enumerate(self.flows.children()):
            x, log_s, log_det_W = flow(x, condition)
            if (i + 1) % self.early_every == 0:
                output, x = x[:, :self.early_size], x[:, self.early_size:]
                outputs.append(output)
            sum_log_s += log_s
            sum_log_det_W += log_det_W
        outputs.append(x)
        z = chainer.functions.concat(outputs, axis=1)
        gaussian_nll = chainer.functions.gaussian_nll(
            z,
            mean=self.xp.zeros_like(z, dtype=self.xp.float32),
            ln_var=self.ln_var * self.xp.ones_like(z, dtype=self.xp.float32)
        )
        gaussian_nll /= numpy.prod(z.shape)
        sum_log_s /= numpy.prod(z.shape)
        sum_log_det_W /= numpy.prod(z.shape)
        return z, gaussian_nll, sum_log_s, sum_log_det_W

    def _reverse(self, z, condition, var=0):
        condition = self.encoder(condition)
        condition = _squeeze(condition, self.squeeze_factor)
        batchsize, _, length = condition.shape
        if z is None:
            z = self.xp.random.normal(
                0, var,
                (batchsize, self.input_channel * self.squeeze_factor, length))
            z = z.astype(self.xp.float32)
        _, channel, _ = z.shape
        start_channel = channel - \
            self.early_size * (self.n_flows // self.early_every)
        x, z = z[:, -start_channel:], z[:, :-start_channel]
        for i, flow in enumerate(reversed(list(self.flows.children()))):
            if (self.n_flows - i) % self.early_every == 0:
                x, z = chainer.functions.concat((
                    z[:, -self.early_size:], x)), z[:, :-self.early_size]
            x = flow.reverse(x, condition)
        x = _unsqueeze(x, self.squeeze_factor)
        return x

    def generate(self, condition, var=0.6 ** 2):
        return self._reverse(None, condition, var)
