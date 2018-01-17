export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
python -u train.py --network vgg16_reduced --train-path /data1/datasets/VOCdevkit/vocrec/train.rec --val-path /data1/datasets/VOCdevkit/vocrec/val.rec \
 --resume -1 --batch-size 32 --label-width 380 --num-class 20  --pretrained /data1/yry/yuanry/ssd/model/vgg16_reduced --epoch 0 \
 --prefix /data1/yry/yuanry/ssd/checkpoint/ssd-vgg \
 --gpus 0,1,2,3  --data-shape 512 --log logs/ssd_train_vgg_voc.log  \
 --class-names 'aeroplane, bicycle, bird, boat,bottle,bus,car,cat,chair,cow,diningtable,dog, horse, motorbike,person, pottedplant,sheep,sofa,train,tvmonitor'
