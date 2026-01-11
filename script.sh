for seed in 42 43 44; do
    for target in banana multimodal x_shaped; do
        for runner in rsivi aisivi sivi uivi_new; do
            config="configs/${runner}_${target}.yaml"
            echo "Running with config: $config and seed: $seed"
            echo "Running with config: $config and seed: $seed" >> output.txt
            python src.py --config=$config seed=$seed 2>&1 >> output.txt
        done
    done
done