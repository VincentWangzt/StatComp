for target in 'banana' 'multimodal' 'x_shaped'; do
    python mcmc_baseline.py --target $target \
        --num-samples 100000 \
        --burn-in 50000 \
        --num-steps 10 
done