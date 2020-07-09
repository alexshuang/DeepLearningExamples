#!/bin/sh

TOP_DIR=../
STEPS=${1:-120}
WARMUP_STEPS=${2:-20}
if [ $STEPS -le $WARMUP_STEPS ]; then
    WARMUP_STEPS=$(expr $STEPS / 5)
fi

OUT_DIR=/data/rocm${ROCM_VERSION}_rocblas${ROCBLAS_VERSION}/resnet50v1.5
rm -rf $OUT_DIR
mkdir -p $OUT_DIR

# record kernels
export ROCBLAS_LAYER=2
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
rm -f ${ROCBLAS_LOG_BENCH_PATH}

echo "pmc: FetchSize L2CacheHit" > input.txt
/opt/rocm/bin/rocprof -i input.txt --obj-tracking on --timestamp on --stats -o ${OUT_DIR}/kernel_prof.csv \
python3.6 ./main.py --arch resnet50 -c fanin --label-smoothing 0.1 --amp --static-loss-scale 256 --data-backend "syntetic" --warmup-steps $WARMUP_STEPS --max-steps $STEPS --result-dir $OUT_DIR $OUT_DIR 
#python3.6 ./main.py $OUT_DIR --data-backend "syntetic" --raport-file raport.json -j8 -p 100 --lr 2.048 --optimizer-batch-size 2048 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 -b 256 --static-loss-scale 128 --epochs 1 --warmup-steps $WARMUP_STEPS --max-steps $STEPS --result-dir $OUT_DIR --amp #--workspace $OUT_DIR --data $OUT_DIR
#python3.6 -m torch.distributed.launch --nproc_per_node 1 ./main.py $OUT_DIR --data-backend syntetic --arch resnet50 -b 32 --epochs 1 --warmup-steps $WARMUP_STEPS --max-steps $STEPS --result-dir $OUT_DIR --amp #--workspace $OUT_DIR --data $OUT_DIR

