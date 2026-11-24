# character-level GPT trained ONLY on the complete works of Shakespeare (embedding size 512)
# first run:  python data/shakespeare_complete_char/prepare.py
# then:       python train.py config/train_shakespeare_complete_char.py

out_dir = 'out-shakespeare-complete-512'
eval_interval = 250  # how often we measure the loss
eval_iters = 100
log_interval = 50

ckpt_interval = 1000 # numbered checkpoints: ckpt_001.pt ... ckpt_010.pt
always_save_checkpoint = True
init_from = 'scratch'

wandb_log = False
dataset = 'shakespeare_complete_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 512 # the model sees up to 512 previous characters

# baby GPT model, embedding size 512 (about 19 million parameters)
n_layer = 6
n_head = 8       # 512 / 8 = 64 numbers per attention head
n_embd = 512
dropout = 0.2

learning_rate = 1e-3
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# on a laptop without a GPU also add:  --device=cpu --compile=False
