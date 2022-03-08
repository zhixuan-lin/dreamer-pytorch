# dreamer-pytorch
IFT6163 course project, PyTorch reimplementation of the Dreamer model


## TODO
* Save agent

## Notes

* `length`: except when logging, it always mean the number of *observations* instead of the number of *transitions*.
* `global_frames` vs `global_steps`: `global_frames == global_steps ** action_repeat`. When we talk about "step", we always mean `global_frames`
