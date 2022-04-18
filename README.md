# dreamer-pytorch
IFT6163 course project, PyTorch reimplementation of the Dreamer model

## OOM

If you are getting OOM errors, pass `deterministic=True`. This sets `torch.backends.cudnn.deterministic=True`, which
uses more memory-efficient algorithms for convolution.

## Dependency

```sh
conda create -n dreamer-pytorch python=3.8
conda activate dreamer-pytorch
pip install -r requirements.txt
```

## Training

```sh
python dreamer.py
```

## Quick debug

```sh
python dreamer.py prefill=100 train_steps=2 batch_size=10 batch_length=10
```

## Logging

Tensorboard:

```sh
tensorboard --logdir ./output/ --port 8888 --host 0.0.0.0
```

videos: see `./output/video/`. `video/model/` contains reconstruction and generation, `video/interaction` contains videos of environment interaction

## TODO
* Get action: `Dreamer.get_action`, `Dreamer.policy`, `Dreamer.exploration`. Currently it is just random
* Saving/loading agent: `Dreamer.load`, `Dreamer.save`
* Debug world models: let's just use random policy and see it can learn the video prediction model
* Debug actor value heads

## Notes

* `length`: except when logging, it always mean the number of *observations* instead of the number of *transitions*.
* `global_frames` vs `global_steps`: `global_frames == global_steps ** action_repeat`. When we talk about "step", we always mean `global_frames`
