from omegaconf import OmegaConf
from rnn_trainer_attention import BrainToTextDecoder_Trainer_Attention

args = OmegaConf.load('rnn_args_attention.yaml')
trainer = BrainToTextDecoder_Trainer_Attention(args)
metrics = trainer.train()
