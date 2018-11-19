# parameters of training
batchsize = 2
lr = 1e-4
trigger = (500000, 'iteration')
annealing_interval = (200000, 'iteration')
evaluate_interval = (10000, 'iteration')
snapshot_interval = (10000, 'iteration')
report_interval = (100, 'iteration')

# parameters of dataset
root = '/media/hdd1/datasets/LJSpeech-1.1'
dataset_type = 'LJSpeech'
split_seed = None

# parameters of preprocessing
sr = 22050
n_fft = 1024
hop_length = 256
n_mels = 80
fmin = 0
fmax = None
top_db = 20
length = 8192 * 2

# parameters of WaveNet
n_layers = 8
wn_channel = 512

# parameters of WaveGlow
squeeze_factor = 8
n_flows = 12
early_every = 4
early_size = 2
var = 1
