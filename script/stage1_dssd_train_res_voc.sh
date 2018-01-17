export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
python -u train.py --network resnet101 --train-path /data1/yry/vocrec/train.rec --val-path /data1/yry/vocrec/val.rec \
 --resume 70 --batch-size 120 --label-width 380 --num-class 20 --lr-steps '60,100' --end-epoch 100 \
 --prefix /data1/yry/ssd/checkpoint/stage1_dssd-res --freeze "(_plus*|bn*|conv0_weight|multi_feat_*|stage*)" \
 --gpus 0,1,2,3  --data-shape 512 --log logs/dssd_train_res_voc.log --lr 0.001 \
 --class-names 'aeroplane, bicycle, bird, boat,bottle,bus,car,cat,chair,cow,diningtable,dog, horse, motorbike,person, pottedplant,sheep,sofa,train,tvmonitor'
