set -e

# python visualization/scripts/organize_data.py
python visualization/scripts/compute_kl_divergence.py
python visualization/scripts/plot_kl_divergence.py
python visualization/scripts/plot_scatter.py
python visualization/scripts/plot_density.py
python visualization/scripts/compute_W2_distance.py
python visualization/scripts/plot_W2_distance.py