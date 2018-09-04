function lsf_submit {
    local cmd=$*
    bsub -W 24:00 -n 16 -R "rusage[mem=500]" -R "affinity[core(1):distribute=pack(socket=1)]" $cmd
}

for env in Pong SpaceInvaders Qbert BeamRider Breakout; do
    for seed in 0 1 2; do
       lsf_submit python train.py ${env}NoFrameskip-v4 --seed $seed --n_workers 16 --run_name ${env}-${seed} --n_steps 50e6 --lr_schedule linear --lr_decay_to_zero_by_n_steps 50e6
    done
done
