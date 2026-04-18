#!/bin/bash

if [ "$1" == "dip" ]; then
    echo "Finetuning on DIP..." 
    [ -d "checkpoints/$2/finetuned_dip" ] && rm -r "checkpoints/$2/finetuned_dip"
    python train.py --module joints --init-from checkpoints/$2/joints --finetune dip
    python train.py --module poser --init-from checkpoints/$2/poser --finetune dip
elif [ "$1" == "imuposer" ]; then 
    echo "Finetuning on IMUPoser..." 
    [ -d "checkpoints/$2/finetuned_imuposer" ] && rm -r "checkpoints/$2/finetuned_imuposer"
    python train.py --module joints --init-from checkpoints/$2/finetuned_dip/joints --finetune imuposer
    python train.py --module poser --init-from checkpoints/$2/finetuned_dip/poser --finetune imuposer
elif [ "$1" == "huawei" ]; then 
    echo "Finetuning on Huawei Dataset..." 
    [ -d "checkpoints/$2/finetuned_huawei" ] && rm -r "checkpoints/$2/finetuned_huawei"
    python train.py --module joints --init-from checkpoints/$2/joints --finetune huawei
    python train.py --module poser --init-from checkpoints/$2/poser --finetune huawei
else
    echo "Invalid argument. Please specify 'dip', 'imuposer', or 'huawei'"
fi
