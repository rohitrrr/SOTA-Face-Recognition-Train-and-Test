"""
AdaFace R100 training on BUPT-CBFace-12 dataset.

Before training, generate the LMDB with:
    python utils/bupt_cbface_to_lmdb.py \
        --dataset_dir /path/to/BUPT-CBFace-12 \
        --destination ./dataset \
        --file_name bupt_cbface

Run:
    torchrun --nproc_per_node=NUM_GPUS train.py \
        --config_file ./configs/adaface_r100_bupt.py
"""
from easydict import EasyDict

config = EasyDict()

config.prefix = "adaface-r100-bupt-cbface12"
config.head = "adaface"
config.depth = "100"
config.batch_size = 64
config.lr = 0.025
config.epochs = 20
config.reduce_lr = [8, 12, 15, 18]
config.scheduler = True
config.warmup_epoch = 1
config.margin = 0.4
config.num_ims = 500000
config.train_source = "./dataset/bupt_cbface.lmdb"
config.val_list = []
config.val_source = "./test_set_package_5"
config.add_flip = False
config.add_norm = True
