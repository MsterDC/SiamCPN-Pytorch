# SCPN Training Tutorial

This implements training of SCPN with backbone architectures, such as ResNet.
### Add SCPN to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/SCPN:$PYTHONPATH
```

## Prepare training dataset
Prepare training dataset, detailed preparations are listed in [training_dataset](training_dataset) directory.
* [VID](http://image-net.org/challenges/LSVRC/2017/)
* [YOUTUBEBB](https://research.google.com/youtube-bb/) (New link for cropped data, [BaiduYun](https://pan.baidu.com/s/1xvgzU0pjQXXgVeJnK7vHSg), extract code: 6dd5. **NOTE: Data in old link is not correct. Please use cropped data in this new link.**)
* [DET](http://image-net.org/challenges/LSVRC/2017/)
* [COCO](http://cocodataset.org)

## Download pretrained backbones
Download pretrained backbones from [Google Drive](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) and put them in `pretrained_models` directory

## Training

To train a model (SCPN), run `train.py` with the desired configs:

```bash
cd experiments/siamcpn_r50_l234_dwxcorr_8gpu
```

### Multi-processing Distributed Data Parallel Training

#### Single node, multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=2333 \
    ../../tools/train.py --cfg config.yaml
```

## Testing
For `ResNet`, you need to test snapshots from 10 to 20 epoch.

```bash 
START=10
END=20
seq $START 1 $END | \
    xargs -I {} echo "snapshot/checkpoint_e{}.pth" | \
    xargs -I {} \ 
    python -u ../../tools/test.py \
        --snapshot {} \
	--config config.yaml \
	--dataset VOT2018 2>&1 | tee logs/test_dataset.log
```

## Evaluation
```
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 4 		 \ # number thread to eval
	--tracker_prefix 'ch*'   # tracker_name
```
