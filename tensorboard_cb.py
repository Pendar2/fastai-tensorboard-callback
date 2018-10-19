from tensorboardX import SummaryWriter
from fastai import *

@dataclass
class TensorboardLogger(Callback):
    learn:Learner
    run_name:str
    histogram_freq:int=100
    path:str=None
    def __post_init__(self):
        self.path = self.path or os.path.join(self.learn.path, "logs")
        self.log_dir = os.path.join(self.path, self.run_name)
    def on_train_begin(self, **kwargs):
        self.writer = SummaryWriter(log_dir=self.log_dir)
    def on_epoch_end(self, **kwargs):
        iteration = kwargs["iteration"]
        metrics = kwargs["last_metrics"]
        metrics_names = ["valid_loss"] + [o.__name__ for o in self.learn.metrics]
        
        for val, name in zip(metrics, metrics_names):
            self.writer.add_scalar(name, val, iteration)
            
        for name, emb in self.learn.model.named_children():
            if isinstance(emb, nn.Embedding):
                self.writer.add_embedding(list(emb.parameters())[0], global_step=iteration, tag=name)
                
    def on_batch_end(self, **kwargs):
        iteration = kwargs["iteration"]
        loss = kwargs["last_loss"]
        
        self.writer.add_scalar("learning_rate", self.learn.opt.lr, iteration)
        self.writer.add_scalar("momentum", self.learn.opt.mom, iteration)
        
        self.writer.add_scalar("loss", loss, iteration)
        if iteration%self.histogram_freq==0:
            for name, param in self.learn.model.named_parameters():
                self.writer.add_histogram(name, param, iteration)
    def on_train_end(self, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dummy_input = next(iter(self.learn.data.train_dl))[0]
                self.writer.add_graph(self.learn.model, tuple(dummy_input))
        except Exception as e:
            print("Unable to create graph.")
            print(e)
        self.writer.close()