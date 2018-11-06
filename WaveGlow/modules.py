import chainer
import chainer.functions as F
import chainer.links as L


def _normalize(W):
    xp = chainer.cuda.get_array_module(W)
    g = xp.sqrt(xp.sum(W ** 2)).reshape((1,))
    v = W / g
    return g, v


def weight_norm(link):
    assert hasattr(link, 'W')

    def _W(self):
        return self.v * self.g

    def _remove(self):
        W = _W(self)
        del self.g
        del self.v
        del self.W
        with self.init_scope():
            self.W = chainer.Parameter(W)

    def _replace(args):
        W = _W(args.link)
        g, v = _normalize(_W(args.link).array)
        args.link.g.array[...] = g
        args.link.v.array[...] = v
        args.link.W = W

    g, v = _normalize(link.W.array)
    del link.W
    with link.init_scope():
        link.g = chainer.Parameter(g)
        link.v = chainer.Parameter(v)

    link.remove = _remove

    hook = chainer.LinkHook()
    hook.forward_preprocess = _replace
    link.add_hook(hook)
    return link


class Invertible1x1Convolution(chainer.link.Link):
    def __init__(self, channel):
        super(Invertible1x1Convolution, self).__init__()
        xp = self.xp

        W = xp.linalg.qr(xp.random.normal(
            0, 1, (channel, channel)))[0].astype(xp.float32)
        W = W.reshape(W.shape + (1,))

        with self.init_scope():
            self.W = chainer.Parameter(W)

    @property
    def invW(self):
        return F.expand_dims(F.inv(self.W[..., 0]), axis=2)

    def __call__(self, x):
        return F.convolution_1d(x, self.W), \
            x.shape[0] * x.shape[-1] * F.log(F.absolute(F.det(self.W[..., 0])))

    def reverse(self, x):
        return F.convolution_1d(x, self.invW)


class WaveNet(chainer.Chain):
    def __init__(self, out_channel, n_condition, n_layers, n_channel):
        super(WaveNet, self).__init__()
        dilated_convs = chainer.ChainList()
        residual_convs = chainer.ChainList()
        skip_convs = chainer.ChainList()
        condition_convs = chainer.ChainList()
        for i in range(n_layers):
            dilated_convs.add_link(weight_norm(
                L.Convolution1D(
                    n_channel, 2 * n_channel, 3, pad=2 ** i, dilate=2 ** i)))
            residual_convs.add_link(weight_norm(
                L.Convolution1D(n_channel, n_channel, 1)))
            skip_convs.add_link(weight_norm(
                L.Convolution1D(n_channel, n_channel, 1)))
            condition_convs.add_link(weight_norm(
                L.Convolution1D(n_condition, 2 * n_channel, 1)))
        with self.init_scope():
            self.input_conv = weight_norm(
                L.Convolution1D(out_channel // 2, n_channel, 1))
            self.dilated_convs = dilated_convs
            self.residual_convs = residual_convs
            self.skip_convs = skip_convs
            self.condition_convs = condition_convs
            self.output_conv = L.Convolution1D(
                n_channel, out_channel, 1,
                initialW=chainer.initializers.Zero())

    def __call__(self, x, condition):
        x = self.input_conv(x)
        skip_connection = 0
        for dilated, residual, skip, condition_conv in zip(
                self.dilated_convs, self.residual_convs, self.skip_convs,
                self.condition_convs):
            z = dilated(x) + condition_conv(condition)
            z_tanh, z_sigmoid = F.split_axis(z, 2, axis=1)
            z = F.tanh(z_tanh) * F.sigmoid(z_sigmoid)
            x = residual(z)
            skip_connection += skip(z)
        y = self.output_conv(skip_connection)
        log_s, t = F.split_axis(y, 2, axis=1)
        return log_s, t


class AffineCouplingLayer(chainer.Chain):
    def __init__(self, *args, **kwargs):
        super(AffineCouplingLayer, self).__init__()
        with self.init_scope():
            self.encoder = WaveNet(*args, **kwargs)

    def __call__(self, x, condition):
        x_a, x_b = F.split_axis(x, 2, axis=1)
        log_s, t = self.encoder(x_a, condition)
        x_b = F.exp(log_s) * (x_b + t)
        return F.concat((x_a, x_b), axis=1), F.sum(log_s)

    def reverse(self, z, condition):
        x_a, x_b = F.split_axis(z, 2, axis=1)
        log_s, t = self.encoder(x_a, condition)
        x_b = x_b * F.exp(-log_s) - t
        return F.concat((x_a, x_b), axis=1)


class Flow(chainer.Chain):
    def __init__(self, channel, n_condition, n_layers, wn_channel):
        super(Flow, self).__init__()
        with self.init_scope():
            self.invertible1x1convolution = Invertible1x1Convolution(
                channel)
            self.affinecouplinglayer = AffineCouplingLayer(
                channel, n_condition, n_layers, wn_channel)

    def __call__(self, x, condition):
        x, log_det_W = self.invertible1x1convolution(x)
        z, log_s = self.affinecouplinglayer(x, condition)
        return z, log_s, log_det_W

    def reverse(self, z, condition):
        z = self.affinecouplinglayer.reverse(z, condition)
        x = self.invertible1x1convolution.reverse(z)
        return x
