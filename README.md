# chainer-WaveGlow

A Chainer implementation of WaveGlow( https://nv-adlr.github.io/WaveGlow ).

# Generated samples

I uploaded generated samples to SoundCloud.
https://soundcloud.com/dhgrs/sets/waveglow

And uploaded pretrained model for LJSpeech to Google Drive.
https://drive.google.com/drive/folders/1Rsr9tymkrMrPYvo_PMC3o4_Q6C0cqxQR?usp=sharing

# Requirements
I trained and generated with

- python(3.5.2)
- chainer (5.0.0)
- librosa (0.6.2)
- matplotlib (3.0.1)

# Usage
## download dataset
You can download VCTK Corpus(en multi speaker)/LJ-Speech(en single speaker) very easily via [my repository](https://github.com/dhgrs/download_dataset).

## set parameters
I'll write details later.

## training
You can use same command in each directory.
```
(without GPU)
python train.py

(with GPU #n)
python train.py -g n
```

You can resume snapshot and restart training like below.
```
python train.py -r snapshot_iter_100000
```
Other arguments `-f` and `-p` are parameters for multiprocess in preprocessing. `-f` means the number of prefetch and `-p` means the number of processes.

## generating
```
python generate.py -i <input file> -o <output file> -m <trained model>
```

If you don't set `-o`, default file name `result.wav` is used. If you don't set `-s`, the speaker is same as input file that got from filepath.
