export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
python -u train.py --network vgg16_reduced --train-path /data1/datasets/VOCdevkit/vocrec/train.rec --val-path /data1/datasets/VOCdevkit/vocrec/val.rec \
 --resume 1 --batch-size 75 --label-width 380 --num-class 20 --lr-steps '60,100' --end-epoch 100 \
 --prefix /data1/yry/yuanry/ssd/checkpoint/stage1_dssd-vgg --freeze "(broadcast_mul0*|relu*|fc*|conv\d_\d_weight|conv\d_\d_bias|multi_feat_*)" \
 --gpus 1,2,3  --data-shape 512 --log logs/stage1_dssd_train_res_voc.log --lr 0.001 \
 --class-names 'aeroplane, bicycle, bird, boat,bottle,bus,car,cat,chair,cow,diningtable,dog, horse, motorbike,person, pottedplant,sheep,sofa,train,tvmonitor'
