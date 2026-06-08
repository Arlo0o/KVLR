_base_ = ["stage2.py"]

plugin = "hybrid"
plugin_config = dict(
    tp_size=1,
    pp_size=1,
    sp_size=4,
    sequence_parallelism_mode="ring_attn",
    enable_sequence_parallelism=True,
    static_graph=True,
    zero_stage=2,
    reduce_bucket_size_in_m=32,
    overlap_allgather=False,
)
accumulation_steps = 4

model = dict(cond_embed=True)
grad_ckpt_buffer_size = 0

condition_config = dict(
    t2v=1,
    i2v_head=5,
    i2v_loop=1,
    i2v_tail=1,
)

is_causal_vae = True

bucket_config = {
    "_delete_": True,
    "768px": {
        33: (1.0, 1),
    },
}

epochs = 100
lr = 1e-5
optim = dict(lr=lr)
ckpt_every = 400
keep_n_latest = 200
