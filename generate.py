import argparse

import numpy
import librosa
import chainer

from WaveGlow import Glow
from utils import Preprocess
import params

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='Input file')
parser.add_argument('--output', '-o', default='Result.wav', help='output file')
parser.add_argument('--model', '-m', help='Snapshot of trained model')
parser.add_argument('--var', '-v', type=float, default=0.6 ** 2,
                    help='Variance of Gaussian distribution')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
if args.gpu != [-1]:
    chainer.cuda.set_max_workspace_size(2 * 512 * 1024 * 1024)
    chainer.global_config.autotune = True

# set data
path = args.input

# preprocess
n = 1  # batchsize; now suporrts only 1
inputs = Preprocess(
    params.sr, params.n_fft, params.hop_length, params.n_mels, params.fmin,
    params.fmax, params.top_db, None)(path)

_, condition = inputs
condition = numpy.expand_dims(condition, axis=0)

# make model
glow = Glow(
    params.hop_length, params.n_mels, 1,
    params.squeeze_factor, params.n_flows, params.n_layers,
    params.wn_channel, params.early_every, params.early_size,
    params.var)

# load trained parameter
chainer.serializers.load_npz(args.model, glow, 'updater/model:main/')

if args.gpu >= 0:
    use_gpu = True
    chainer.cuda.get_device_from_id(args.gpu).use()
else:
    use_gpu = False

# forward
if use_gpu:
    condition = chainer.cuda.to_gpu(condition, device=args.gpu)
    glow.to_gpu(device=args.gpu)
condition = chainer.Variable(condition)

with chainer.using_config('enable_backprop', False):
    output = glow.generate(condition)

output = chainer.cuda.to_cpu(output.array)
output = numpy.squeeze(output)
librosa.output.write_wav(args.output, output, params.sr)
