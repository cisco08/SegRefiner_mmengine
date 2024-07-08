_base_ = [
    './segrefiner_lr.py'
]

object_size = 256
from mmcv.utils import to_2tuple
object_size = to_2tuple([128, 128])

task='semantic'

model = dict(
    type='SegRefinerSemantic',
    task=task,
    step=1,
    denoise_model=dict(num_res_blocks=1,
                       channel_mult = (1, 2),),
    diffusion_cfg=dict(betas=dict(num_timesteps=2)),
    test_cfg=dict(
        model_size=object_size,
        fine_prob_thr=0.9,
        iou_thr=0.3,
        batch_max=32))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadCoarseMasks', test_mode=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'coarse_masks'])
]


img_root = 'data/'

data = dict(
    train=dict(),
    val=dict(),
    test=dict(
        pipeline=test_pipeline,
        type='BigDataset',
        data_root='data/big/coarse/deeplab'),
    test_dataloader = dict(
        samples_per_gpu=1,
        workers_per_gpu=1))
