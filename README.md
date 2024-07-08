# SegRefiner_mmengine
clone自https://github.com/MengyuWang826/SegRefiner 由于最新的mmsegmentation使用了mmengine等新库，
所以修改了一下；并且在SegRefiner模块里增加forward_dummy来支持get_flops.py的运行.

```bash
python scripts/gen_coarse_masks_hr.py
python tools/analysis_tools/get_flops.py configs/segrefiner/segrefiner_big.py --shape 128 128
```
