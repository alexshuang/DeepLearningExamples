#! /bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#nvidia-smi

RESULTS_DIR='/results'
CHECKPOINTS_DIR='/results/checkpoints'
STAT_FILE=${RESULTS_DIR}/run_log.json
mkdir -p $CHECKPOINTS_DIR

PREC=${1:-'amp'}
SEED=${2:-1}
LR=${3:-0.000846}
WARMUP=${4:-4000}
NUM_EPOCHS=${5:-40}
BATCH_SIZE=${6:-10240}
NUM_GPU=${7:-1}
: ${USE_SLURM:=0}

export CUDA_VISIBLE_DEVICES=6

DISTRIBUTED="-m torch.distributed.launch --nproc_per_node=${NUM_GPU}"
[ ${USE_SLURM} = 1 ] && DISTRIBUTED+=" --nnodes ${WORLD_SIZE} --node_rank ${SLURM_NODEID}  \
        --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} "

#job_name="Transformer ${PREC} $(( ${NUM_GPU} * ${WORLD_SIZE} ))GPU BS${BATCH_SIZE} LR${LR} WARMUP${WARMUP} SEED ${SEED}"

if [ "$PREC" = "amp" ];
then
    PREC='--amp --amp-level O2 '
else
    PREC=''
fi

STEPS=${8:-1000}
WARMUP_STEPS=${9:-30}
OUT_DIR=${10:-out}
TMP_DIR=$OUT_DIR/tmp
DEVICE=${11:-1}

mkdir -p $OUT_DIR $TMP_DIR

export CUDA_VISIBLE_DEVICES=6

#python ${DISTRIBUTED} train.py \
CMD="python train.py \
  /data/newdata/wmt14_en_de_joined_dict \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9,0.997)' \
  --adam-eps '1e-9' \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates ${WARMUP} \
  --lr $LR \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-tokens ${BATCH_SIZE} \
  --seed ${SEED} \
  --max-epoch ${NUM_EPOCHS} \
  --no-epoch-checkpoints \
  --fuse-layer-norm \
  --online-eval \
  --log-interval 10000 \
  --save-dir ${RESULTS_DIR} \
  --stat-file ${STAT_FILE} \
  --warmup-updates $WARMUP_STEPS \
  ${PREC}"

$CMD --result-dir $OUT_DIR --max-update $STEPS | tee $TMP_DIR/run.log

nvprof --print-gpu-trace --concurrent-kernels off --csv \
$CMD --max_steps 100

#--distributed-init-method env:// \

#python scripts/draw_summary.py --log-file ${STAT_FILE} --output ${RESULTS_DIR}/summary.png --title "${job_name}" -j ${RESULTS_DIR}/summary.json
