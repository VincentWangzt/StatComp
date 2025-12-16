for target in 'multimodal' 'x_shaped'; do
    # timeout 1h python src.py --config configs/uivi_${target}.yaml
    timeout 30m python src.py --config configs/reverse_uivi_${target}.yaml
done