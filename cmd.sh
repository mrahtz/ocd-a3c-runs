function lsf_submit {
    local cmd=$*
    bsub -W 15:00 -n 16 -R "rusage[mem=500]" -R "affinity[core(1):distribute=pack(socket=1)]" $cmd
}

for n_workers in 1 2 4 8 16; do
   lsf_submit python train.py PongNoFrameskip-v4 --n_workers $n_workers --run_name Pong-${n_workers}workers --n_steps 50e6 --lr_schedule linear --lr_decay_to_zero_by_n_steps 50e6
done
