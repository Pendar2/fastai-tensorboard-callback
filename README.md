# Fast.ai Tensorboard Callback

Updated to support Fastai v1

[Fastai forum post](https://forums.fast.ai/t/tensorboard-callback-for-fastai/19048)

This callback plots training loss, validation loss, metrics, learning rate, and momentum. Every X iterations a snapshot of the model’s weights are logged and can be viewed in Tensorboard histogram and distribution tab. Every epoch, the embedding layers are saved and can be viewed in 3D with dimensionality reduction in the projector tab. Lastly, the model’s dataflow graph can be viewed in the graph tab.

### Installation
```bash
pip install git+https://github.com/Pendar2/fastai-tensorboard-callback.git
```


Launch the Tensorboard server with `tensorboard --logdir="learn.path/logs"`, then navigate to localhost:6006
