export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
python -u train.py --network resnet101 --train-path /data1/datasets/VOCdevkit/vocrec/train.rec --val-path /data1/datasets/VOCdevkit/vocrec/val.rec \
 --resume -1 --batch-size 32 --label-width 380 --num-class 20  --pretrained /data1/yry/ssd/model/resnet101 --epoch 0 \
 --prefix /data1/yry/ssd/checkpoint/ssd-res \
 --gpus 0,1,2,3  --data-shape 512 --log logs/ssd_train_res_voc.log  \
 --class-names 'aeroplane, bicycle, bird, boat,bottle,bus,car,cat,chair,cow,diningtable,dog, horse, motorbike,person, pottedplant,sheep,sofa,train,tvmonitor'
