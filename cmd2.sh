function lsf_submit {
    local cmd=$*
    bsub -W 15:00 -n 16 -R "rusage[mem=500]" -R "affinity[core(1):distribute=pack(socket=1)]" $cmd
}

lsf_submit python train.py BeamRiderNoFrameskip-v4 --n_workers 16 --run_name BeamRider_lr1e-3_clip0.5 --n_steps 50e6 --lr_schedule linear --lr_decay_to_zero_by_n_steps 50e6 --initial_lr 1e-3 --max_grad_norm 0.5
