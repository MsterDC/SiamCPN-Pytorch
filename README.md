# SiamCPN | visual tracking with siamese center-prediction network

SiamCPN is developed based on [PySot](https://github.com/STVIR/pysot). Our work has achieved a single target tracking method using a central point. The code base is for reference by relevant researchers only, and may not be used in any commercial activities without permission.

**PySOT** is a software system designed by SenseTime Video Intelligence Research team. It implements state-of-the-art single object tracking algorithms, including [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html) and [SiamMask](https://arxiv.org/abs/1812.05050). It is written in Python and powered by the [PyTorch](https://pytorch.org) deep learning framework. This project also contains a Python port of toolkit for evaluating trackers.

PySOT has enabled research projects, including: [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_High_Performance_Visual_CVPR_2018_paper.html), [DaSiamRPN](https://arxiv.org/abs/1808.06048), [SiamRPN++](https://arxiv.org/abs/1812.11703), and [SiamMask](https://arxiv.org/abs/1812.05050).

## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using SCPN

### Add SCPN to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/scpn:$PYTHONPATH
```

###  Training :wrench:
See [TRAIN.md](TRAIN.md) for detailed instruction.


## Contributors
- [Dong Chen](https://github.com/KevinDongDong)

## Acknowledgement
Thanks a lot to the contributors of PySot. Based on their extraordinary efforts, we can quickly develop this work, which saves us a lot of time in our research process.
