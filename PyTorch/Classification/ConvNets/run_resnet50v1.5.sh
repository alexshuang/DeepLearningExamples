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

#CMD="python3.6 ./main.py $OUT_DIR --arch resnet50 -c fanin --label-smoothing 0.1 --amp --static-loss-scale 128 --data-backend syntetic"
CMD="python3.6 ./main.py $OUT_DIR --data-backend syntetic --raport-file raport.json -j8 -p 100 --lr 2.048 --warmup 8 --arch resnet50 -c fanin --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace $OUT_DIR --amp --static-loss-scale 128 --epochs 50"

# training
$CMD --warmup-steps $WARMUP_STEPS --max-steps $STEPS --result-dir $OUT_DIR

# record kernels
export ROCBLAS_LAYER=2
export ROCBLAS_LOG_BENCH_PATH=${OUT_DIR}/rocblas_bench.csv
rm -f ${ROCBLAS_LOG_BENCH_PATH}
$CMD --max-steps 3 --warmup-steps 0
rm -f ${OUT_DIR}/*.db ${OUT_DIR}/*.json ${OUT_DIR}/*.txt

# rocblas-bench
TOOL=/root/rocblas/build/release/clients/staging/rocblas-bench
if [ ! -e rocblas-bench ]; then
	ln -s ${TOOL} .
fi
unset ROCBLAS_LAYER
NUM_GEMM_PER_ITER=38
tail -$NUM_GEMM_PER_ITER $ROCBLAS_LOG_BENCH_PATH > /tmp/rb.csv
sh /tmp/rb.csv | tee /tmp/rb_res.txt
sed -E -n '/(^N,|^T,)/p' /tmp/rb_res.txt > $OUT_DIR/rocblas_bench_res.csv
echo "File $OUT_DIR/rocblas_bench_res.csv is generated."

cp /tmp/rb.csv /data/resnet_rb.csv -v

