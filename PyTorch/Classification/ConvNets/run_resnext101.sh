#!/bin/sh

TOP_DIR=../
STEPS=${1:-120}
WARMUP_STEPS=${2:-20}
if [ $STEPS -le $WARMUP_STEPS ]; then
    WARMUP_STEPS=$(expr $STEPS / 5)
fi

OUT_DIR=/data/rocm${ROCM_VERSION}_rocblas${ROCBLAS_VERSION}/resnext101-32x4d
rm -rf $OUT_DIR
mkdir -p $OUT_DIR

BS=16
CMD="python3.6 ./multiproc.py --nproc_per_node 1 ./main.py $OUT_DIR --raport-file raport.json -j8 -p 100 --data-backend syntetic --arch resnext101-32x4d -c fanin --label-smoothing 0.1 --workspace $OUT_DIR -b $BS --amp --static-loss-scale 128 --optimizer-batch-size $BS --lr 1.024 --mom 0.875 --lr-schedule cosine --epochs  90 --warmup 8 --wd 6.103515625e-05 --memory-format nhwc"

# training
#$CMD --warmup-steps $WARMUP_STEPS --max-steps $STEPS --result-dir $OUT_DIR

# record kernels
export ROCBLAS_LAYER=2
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
rm -f ${ROCBLAS_LOG_BENCH_PATH}
$CMD --warmup-steps $WARMUP_STEPS --max-steps $STEPS && wc -l $ROCBLAS_LOG_BENCH_PATH
rm -f ${OUT_DIR}/*.db ${OUT_DIR}/*.json ${OUT_DIR}/*.txt

