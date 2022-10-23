#!/bin/bash

for ((i=2016; i<=2025; i++))
do
    python main.py \
    --env_name "HalfCheetah-v2" \
    --T 1000 \
    --K 100 \
    --max_steps 12000000 \
    --stg_name "pges" \
    --pop_size 20 \
    --lrate 0.001 \
    --sigma 0.05 \
    --alpha 0.5 \
    --sub_dims 10 \
    --seed $i \
    --num_worker 6
done

for ((i=2016; i<=2025; i++))
do
    python main.py \
    --env_name "Ant-v2" \
    --T 1000 \
    --K 100 \
    --max_steps 60000000 \
    --stg_name "pges" \
    --pop_size 100 \
    --lrate 0.0005 \
    --sigma 0.02 \
    --alpha 0.5 \
    --sub_dims 20 \
    --seed $i \
    --num_worker 6
done

for ((i=2016; i<=2025; i++))
do
    python main.py \
    --env_name "Swimmer-v2" \
    --T 1000 \
    --K 100 \
    --max_steps 4000000 \
    --stg_name "pges" \
    --pop_size 10 \
    --lrate 0.002 \
    --sigma 0.15 \
    --alpha 0.5 \
    --sub_dims 1 \
    --seed $i \
    --num_worker 6
done

for ((i=2016; i<=2025; i++))
do
    python main.py \
    --env_name "Hopper-v2" \
    --T 1000 \
    --K 100 \
    --max_steps 4000000 \
    --stg_name "pges" \
    --pop_size 10 \
    --lrate 0.002 \
    --sigma 0.15 \
    --alpha 0.5 \
    --sub_dims 1 \
    --seed $i \
    --num_worker 6
done

for ((i=2016; i<=2025; i++))
do
    python main.py \
    --env_name "Walker2d-v2" \
    --T 1000 \
    --K 100 \
    --max_steps 12000000 \
    --stg_name "pges" \
    --pop_size 20 \
    --lrate 0.001 \
    --sigma 0.05 \
    --alpha 0.5 \
    --sub_dims 10 \
    --seed $i \
    --num_worker 6
done
