# Official Re-Evaluation By Target

Runs summarized: 128

## banana

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| SIVI | on | 0.009844 +/- 0.002464 | **0.007024 +/- 6.100e-04** | 0.259208 +/- 0.001677 | **5.109e-04 +/- 2.284e-05** | 2.665e-04 +/- 1.673e-04 | N/A | N/A | 355.588s / e20000 |
| SIVI | off | 0.009137 +/- 0.002835 | **0.007588 +/- 7.225e-04** | 0.281884 +/- 0.001449 | 5.800e-04 +/- 2.707e-05 | **7.687e-05 +/- 1.361e-04** | N/A | N/A | 355.679s / e20000 |
| UIVI | on | -0.023784 +/- 5.907e-04 | 0.022624 +/- 6.121e-04 | 0.532009 +/- 0.0017 | 5.712e-04 +/- 1.837e-05 | 0.006277 +/- 2.678e-04 | N/A | N/A | 934.485s / e10000 |
| UIVI | off | -0.02801 +/- 6.650e-04 | 0.025212 +/- 7.690e-04 | 0.565903 +/- 0.001757 | 6.213e-04 +/- 2.193e-05 | 0.006694 +/- 2.342e-04 | N/A | N/A | 943.870s / e10000 |
| RSIVI | on | -0.982237 +/- 0.009221 | 0.751999 +/- 0.001127 | 0.47698 +/- 0.001094 | 0.026516 +/- 1.971e-04 | 2.2919 +/- 0.011521 | N/A | N/A | 330.661s / e10000 |
| AISIVI | on | -0.012581 +/- 0.017743 | 0.08453 +/- 6.858e-04 | 0.372979 +/- 0.001531 | 0.00409 +/- 1.037e-04 | 0.144563 +/- 0.001892 | N/A | N/A | 455.579s / e10000 |
| AISIVI | off | 0.035382 +/- 0.008936 | 0.008132 +/- 5.298e-04 | 0.292267 +/- 0.001277 | **5.232e-04 +/- 2.168e-05** | 2.126e-04 +/- 1.344e-04 | N/A | N/A | 439.354s / e10000 |
| KSIVI-custom | on | -0.157252 +/- 6.524e-04 | 0.134274 +/- 5.023e-04 | 0.996408 +/- 0.001465 | 0.008169 +/- 1.260e-04 | 0.006037 +/- 1.770e-04 | N/A | N/A | 442.276s / e50000 |
| KSIVI-custom | off | -0.170895 +/- 6.678e-04 | 0.147474 +/- 8.081e-04 | 1.0183 +/- 0.001603 | 0.009729 +/- 1.611e-04 | 0.006383 +/- 1.962e-04 | N/A | N/A | 443.205s / e50000 |
| KSIVI-standard-CG | on | -0.110898 +/- 0.003045 | 0.112212 +/- 7.344e-04 | 0.947806 +/- 0.001864 | 0.007162 +/- 1.247e-04 | 0.003408 +/- 1.621e-04 | N/A | N/A | 443.827s / e50000 |
| KSIVI-standard-CG | off | -0.021304 +/- 0.005382 | 0.071199 +/- 6.266e-04 | 0.827933 +/- 0.001869 | 0.003875 +/- 9.678e-05 | 0.001719 +/- 1.746e-04 | N/A | N/A | 437.778s / e50000 |
| DSIVI-default | on | 0.152008 +/- 0.041223 | **0.007133 +/- 6.764e-04** | 0.247681 +/- 0.001594 | **5.231e-04 +/- 2.032e-05** | 0.002905 +/- 2.999e-04 | N/A | N/A | 316.180s / e10000 |
| DSIVI-default | off | 0.269734 +/- 0.029837 | **0.007858 +/- 6.608e-04** | 0.269698 +/- 0.0015 | **4.931e-04 +/- 2.189e-05** | 2.069e-04 +/- 1.450e-04 | N/A | N/A | 299.576s / e10000 |
| DSIVI-bs4096-rbs2048 | on | 0.204953 +/- 0.034275 | **0.00723 +/- 7.291e-04** | 0.248447 +/- 0.001504 | **5.337e-04 +/- 2.535e-05** | **-1.987e-04 +/- 1.258e-04** | N/A | N/A | 295.071s / e10000 |
| DSIVI-bs4096-rbs2048 | off | **0.379628 +/- 0.038057** | **0.006397 +/- 6.358e-04** | 0.245055 +/- 0.00163 | **4.959e-04 +/- 2.060e-05** | **1.619e-04 +/- 1.543e-04** | N/A | N/A | 294.532s / e10000 |
| DSIVI-bs4096-rbs4096 | on | 0.088644 +/- 0.017896 | 0.008595 +/- 7.380e-04 | **0.240565 +/- 0.001438** | **5.357e-04 +/- 2.548e-05** | 3.378e-04 +/- 1.673e-04 | N/A | N/A | 295.752s / e10000 |
| DSIVI-bs4096-rbs4096 | off | 0.06128 +/- 0.013054 | **0.006951 +/- 6.254e-04** | 0.257849 +/- 0.00137 | **5.283e-04 +/- 2.213e-05** | **1.723e-04 +/- 1.487e-04** | N/A | N/A | 294.838s / e10000 |

## multimodal

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| SIVI | on | -7.017e-04 +/- 8.942e-05 | **8.960e-04 +/- 5.686e-04** | **0.028667 +/- 6.271e-04** | 4.146e-04 +/- 1.448e-05 | 5.030e-05 +/- 2.044e-05 | N/A | N/A | 385.430s / e20000 |
| SIVI | off | -9.206e-04 +/- 1.228e-04 | **0.001778 +/- 6.753e-04** | 0.030242 +/- 6.160e-04 | 4.370e-04 +/- 1.455e-05 | 1.177e-04 +/- 2.088e-05 | N/A | N/A | 365.712s / e20000 |
| UIVI | on | -0.031943 +/- 5.725e-04 | 0.03453 +/- 6.348e-04 | 0.106359 +/- 5.675e-04 | 0.003183 +/- 6.619e-05 | 0.008148 +/- 8.313e-05 | N/A | N/A | 1068.851s / e10000 |
| UIVI | off | -0.036022 +/- 7.271e-04 | 0.038844 +/- 7.286e-04 | 0.127887 +/- 0.001014 | 0.003809 +/- 1.012e-04 | 0.008491 +/- 1.122e-04 | N/A | N/A | 946.411s / e10000 |
| RSIVI | on | -0.006327 +/- 2.837e-04 | 0.00764 +/- 6.597e-04 | 0.053144 +/- 8.623e-04 | 7.580e-04 +/- 2.686e-05 | 0.001149 +/- 4.451e-05 | N/A | N/A | 362.851s / e10000 |
| AISIVI | on | 1.951e-04 +/- 6.860e-05 | **0.001104 +/- 5.147e-04** | **0.027861 +/- 7.351e-04** | **3.783e-04 +/- 1.188e-05** | **-6.356e-06 +/- 1.719e-05** | N/A | N/A | 438.848s / e10000 |
| AISIVI | off | -0.026763 +/- 0.001048 | 0.041327 +/- 7.164e-04 | 0.168663 +/- 0.00128 | 0.002326 +/- 6.513e-05 | 0.004424 +/- 6.688e-05 | N/A | N/A | 438.872s / e10000 |
| KSIVI-custom | on | -0.002432 +/- 2.371e-04 | 0.004955 +/- 7.737e-04 | 0.034435 +/- 7.434e-04 | 5.043e-04 +/- 2.009e-05 | 5.271e-04 +/- 2.671e-05 | N/A | N/A | 535.835s / e50000 |
| KSIVI-custom | off | -0.665004 +/- 5.227e-04 | 0.667978 +/- 6.341e-04 | 1.6180 +/- 0.002518 | 0.091277 +/- 4.127e-04 | 0.004417 +/- 6.652e-05 | N/A | N/A | 437.926s / e50000 |
| KSIVI-standard-CG | on | -0.007762 +/- 2.770e-04 | 0.009563 +/- 5.756e-04 | 0.040946 +/- 5.917e-04 | 6.992e-04 +/- 2.324e-05 | 0.001183 +/- 4.086e-05 | N/A | N/A | 499.801s / e50000 |
| KSIVI-standard-CG | off | -0.57978 +/- 0.004185 | 0.512648 +/- 5.639e-04 | 1.4243 +/- 0.002854 | 0.072225 +/- 3.684e-04 | 0.004335 +/- 8.572e-05 | N/A | N/A | 473.440s / e50000 |
| DSIVI-default | on | 1.024e-04 +/- 7.064e-05 | **0.001205 +/- 6.541e-04** | **0.027947 +/- 7.190e-04** | **3.876e-04 +/- 1.337e-05** | **1.584e-05 +/- 2.016e-05** | N/A | N/A | 301.508s / e10000 |
| DSIVI-default | off | 0.001423 +/- 1.669e-04 | **0.001995 +/- 5.709e-04** | 0.037889 +/- 7.293e-04 | 4.432e-04 +/- 1.719e-05 | 1.648e-04 +/- 2.321e-05 | N/A | N/A | 313.431s / e10000 |
| DSIVI-bs4096-rbs2048 | on | 0.006539 +/- 5.881e-04 | 0.002558 +/- 6.493e-04 | 0.031863 +/- 7.329e-04 | **3.984e-04 +/- 1.364e-05** | 5.380e-05 +/- 2.142e-05 | N/A | N/A | 323.857s / e10000 |
| DSIVI-bs4096-rbs2048 | off | **0.007842 +/- 4.609e-04** | 0.004999 +/- 6.170e-04 | 0.074558 +/- 0.001141 | 5.404e-04 +/- 1.884e-05 | 5.939e-04 +/- 2.939e-05 | N/A | N/A | 350.135s / e10000 |
| DSIVI-bs4096-rbs4096 | on | 1.278e-04 +/- 7.026e-05 | **6.367e-04 +/- 6.731e-04** | **0.02891 +/- 9.537e-04** | **3.655e-04 +/- 1.235e-05** | **9.667e-06 +/- 1.870e-05** | N/A | N/A | 328.740s / e10000 |
| DSIVI-bs4096-rbs4096 | off | **0.008033 +/- 3.437e-04** | 0.002873 +/- 5.637e-04 | 0.048634 +/- 8.324e-04 | 4.307e-04 +/- 1.645e-05 | 1.688e-04 +/- 2.269e-05 | N/A | N/A | 333.821s / e10000 |

## x_shaped

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| SIVI | on | 5.368e-04 +/- 1.523e-04 | **0.003855 +/- 7.214e-04** | **0.046068 +/- 7.009e-04** | **4.251e-04 +/- 1.834e-05** | 1.482e-04 +/- 3.937e-05 | N/A | N/A | 363.381s / e20000 |
| SIVI | off | 1.201e-04 +/- 1.624e-04 | **0.002963 +/- 6.000e-04** | 0.049689 +/- 6.862e-04 | **4.416e-04 +/- 1.776e-05** | 2.411e-04 +/- 3.944e-05 | N/A | N/A | 372.616s / e20000 |
| UIVI | on | -0.134239 +/- 0.001433 | 0.13679 +/- 8.174e-04 | 0.337866 +/- 6.537e-04 | 0.006293 +/- 1.139e-04 | 0.02242 +/- 2.424e-04 | N/A | N/A | 997.658s / e10000 |
| UIVI | off | -0.017036 +/- 5.200e-04 | 0.021742 +/- 5.506e-04 | 0.142153 +/- 9.607e-04 | 0.001601 +/- 5.363e-05 | 0.003214 +/- 8.259e-05 | N/A | N/A | 1014.579s / e10000 |
| RSIVI | on | -0.06717 +/- 0.003301 | 0.089742 +/- 7.141e-04 | 0.28711 +/- 8.130e-04 | 0.003435 +/- 8.262e-05 | 0.010446 +/- 1.710e-04 | N/A | N/A | 370.734s / e10000 |
| AISIVI | on | 0.041112 +/- 0.008362 | 0.106645 +/- 7.270e-04 | 0.297219 +/- 7.807e-04 | 0.004531 +/- 7.580e-05 | 0.013228 +/- 1.708e-04 | N/A | N/A | 416.295s / e10000 |
| AISIVI | off | **0.107457 +/- 0.016772** | 0.109264 +/- 8.306e-04 | 0.292835 +/- 5.287e-04 | 0.004179 +/- 8.753e-05 | 0.014258 +/- 1.841e-04 | N/A | N/A | 419.527s / e10000 |
| KSIVI-custom | on | 6.006e-04 +/- 2.504e-04 | **0.002365 +/- 7.081e-04** | 0.051539 +/- 9.655e-04 | **4.170e-04 +/- 1.536e-05** | **-2.298e-06 +/- 3.034e-05** | N/A | N/A | 535.886s / e50000 |
| KSIVI-custom | off | 0.001694 +/- 2.297e-04 | 0.004322 +/- 6.316e-04 | 0.051845 +/- 8.315e-04 | **4.364e-04 +/- 1.617e-05** | **-2.190e-05 +/- 3.742e-05** | N/A | N/A | 557.230s / e50000 |
| KSIVI-standard-CG | on | -0.12936 +/- 0.002815 | 0.152456 +/- 7.111e-04 | 0.457809 +/- 0.001144 | 0.005206 +/- 5.994e-05 | 0.006113 +/- 6.662e-05 | N/A | N/A | 534.813s / e50000 |
| KSIVI-standard-CG | off | -0.410857 +/- 0.002827 | 0.434455 +/- 9.773e-04 | 0.889375 +/- 0.001464 | 0.020212 +/- 1.566e-04 | 0.02197 +/- 1.673e-04 | N/A | N/A | 495.797s / e50000 |
| DSIVI-default | on | 0.057878 +/- 0.003389 | 0.005027 +/- 6.519e-04 | 0.053795 +/- 7.259e-04 | 4.476e-04 +/- 1.946e-05 | 2.893e-04 +/- 3.850e-05 | N/A | N/A | 350.293s / e10000 |
| DSIVI-default | off | 0.022949 +/- 0.002395 | 0.004816 +/- 7.809e-04 | 0.051494 +/- 8.191e-04 | **4.223e-04 +/- 1.641e-05** | 1.148e-04 +/- 4.236e-05 | N/A | N/A | 302.193s / e10000 |
| DSIVI-bs4096-rbs2048 | on | 0.04933 +/- 0.003904 | 0.004437 +/- 6.864e-04 | 0.056104 +/- 8.594e-04 | 4.543e-04 +/- 2.144e-05 | 1.330e-04 +/- 4.152e-05 | N/A | N/A | 311.241s / e10000 |
| DSIVI-bs4096-rbs2048 | off | -0.088129 +/- 0.032013 | 0.226311 +/- 7.218e-04 | 0.420994 +/- 5.994e-04 | 0.012269 +/- 1.608e-04 | 0.032664 +/- 3.472e-04 | N/A | N/A | 297.456s / e10000 |
| DSIVI-bs4096-rbs4096 | on | 0.026123 +/- 0.003373 | **0.003904 +/- 6.868e-04** | 0.048281 +/- 7.288e-04 | **4.344e-04 +/- 2.114e-05** | **5.279e-05 +/- 3.445e-05** | N/A | N/A | 306.002s / e10000 |
| DSIVI-bs4096-rbs4096 | off | 0.022112 +/- 0.0024 | **0.002745 +/- 6.541e-04** | 0.052619 +/- 8.185e-04 | **3.999e-04 +/- 1.424e-05** | 8.732e-05 +/- 3.378e-05 | N/A | N/A | 297.167s / e10000 |

## student_uc

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| SIVI | on | -2.6679 +/- 8.582e-04 | 0.02536 +/- 5.607e-04 | 0.600001 +/- 0.008852 | **4.829e-04 +/- 1.356e-05** | 0.003932 +/- 3.722e-04 | N/A | N/A | 334.213s / e20000 |
| SIVI | off | -2.6734 +/- 2.346e-04 | **0.002462 +/- 6.456e-04** | 0.132038 +/- 0.00123 | **4.885e-04 +/- 1.501e-05** | **0.002663 +/- 4.764e-04** | N/A | N/A | 327.906s / e20000 |
| UIVI | on | -2.7092 +/- 8.710e-04 | 0.022562 +/- 6.674e-04 | 0.356687 +/- 0.007681 | 0.001168 +/- 2.410e-05 | 0.054692 +/- 0.001476 | N/A | N/A | 933.675s / e10000 |
| UIVI | off | -2.6806 +/- 2.661e-04 | **0.003742 +/- 7.856e-04** | **0.111185 +/- 0.001196** | 7.313e-04 +/- 1.985e-05 | 0.015372 +/- 8.484e-04 | N/A | N/A | 927.183s / e10000 |
| KSIVI-custom | on | -9.5824 +/- 0.004891 | 8.1494 +/- 0.001217 | 10.9125 +/- 0.018974 | 0.102641 +/- 1.127e-05 | 0.080082 +/- 1.886e-04 | N/A | N/A | 418.437s / e50000 |
| KSIVI-custom | off | -10.9888 +/- 0.004446 | 9.0156 +/- 9.083e-04 | 15.9579 +/- 0.027965 | 0.101967 +/- 3.287e-06 | 0.053757 +/- 1.375e-04 | N/A | N/A | 361.303s / e50000 |
| KSIVI-standard-CG | on | -8.5273 +/- 0.006154 | 6.1656 +/- 0.001462 | 5.2325 +/- 0.009423 | 0.102653 +/- 3.119e-05 | 0.190703 +/- 4.770e-04 | N/A | N/A | 406.495s / e50000 |
| KSIVI-standard-CG | off | -14.1909 +/- 0.019903 | 8.0631 +/- 0.001246 | 28.1787 +/- 0.048955 | 0.100506 +/- 5.179e-06 | 0.018043 +/- 4.414e-05 | N/A | N/A | 402.341s / e50000 |
| DSIVI-default | on | 6.213e+04 +/- 1.004e+04 | 12.4366 +/- 0.001755 | 578.486 +/- 0.982386 | 0.100308 +/- 1.416e-05 | 0.133574 +/- 0.001508 | N/A | N/A | 62.6960s / e2000 |
| DSIVI-default | off | -2.6688 +/- 4.208e-04 | 0.012331 +/- 6.272e-04 | 0.186727 +/- 0.001517 | 5.522e-04 +/- 1.695e-05 | **0.002942 +/- 3.238e-04** | N/A | N/A | 62.8590s / e2000 |
| DSIVI-bs4096-rbs2048 | on | **2.185e+05 +/- 4706.742** | 8.4254 +/- 0.001238 | 557.498 +/- 1.0022 | 0.100406 +/- 2.175e-07 | 0.004222 +/- 6.960e-05 | N/A | N/A | 60.5750s / e2000 |
| DSIVI-bs4096-rbs2048 | off | -2.6620 +/- 9.130e-04 | 0.011208 +/- 6.059e-04 | 0.164929 +/- 0.001326 | **4.911e-04 +/- 1.633e-05** | **0.003695 +/- 3.901e-04** | N/A | N/A | 66.0790s / e2000 |
| DSIVI-bs4096-rbs4096 | on | 473.746 +/- 23.4762 | 5.5627 +/- 0.00259 | 62.6902 +/- 0.135902 | 0.088012 +/- 1.080e-04 | 0.038044 +/- 7.963e-04 | N/A | N/A | 62.1340s / e2000 |
| DSIVI-bs4096-rbs4096 | off | -2.6574 +/- 0.001398 | 0.021133 +/- 6.402e-04 | 0.224266 +/- 0.001544 | 5.990e-04 +/- 1.684e-05 | 0.009915 +/- 8.220e-04 | N/A | N/A | 60.7440s / e2000 |

## Langevin_post

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| SIVI | on | -175.861 +/- 0.010849 | N/A | 0.034676 +/- 3.930e-05 | 0.022504 +/- 3.084e-05 | 0.097592 +/- 1.032e-04 | N/A | N/A | 1049.584s / e20000 |
| SIVI | off | -176.091 +/- 0.010526 | N/A | 0.034748 +/- 3.914e-05 | 0.022709 +/- 3.288e-05 | 0.097845 +/- 1.019e-04 | N/A | N/A | 1036.152s / e20000 |
| UIVI | on | -90.5355 +/- 0.078628 | N/A | 0.007707 +/- 1.969e-05 | **0.001226 +/- 7.068e-07** | **0.006524 +/- 7.452e-05** | N/A | N/A | 970.946s / e10000 |
| UIVI | off | -112.796 +/- 0.094866 | N/A | **0.007196 +/- 1.510e-05** | 0.00123 +/- 6.334e-07 | 0.00963 +/- 1.025e-04 | N/A | N/A | 968.588s / e10000 |
| RSIVI | on | -70.8295 +/- 0.132251 | N/A | 0.02278 +/- 6.361e-05 | 0.001732 +/- 2.421e-06 | 0.015913 +/- 1.071e-04 | N/A | N/A | 990.040s / e10000 |
| KSIVI-custom | on | -587.089 +/- 0.154212 | N/A | 0.214441 +/- 2.648e-04 | 0.003467 +/- 2.562e-10 | 5.3506 +/- 0.005009 | N/A | N/A | 1112.781s / e100000 |
| KSIVI-custom | off | -860.580 +/- 0.266912 | N/A | 0.266185 +/- 3.613e-04 | 0.003467 +/- 3.408e-11 | 8.2462 +/- 0.007377 | N/A | N/A | 1110.867s / e100000 |
| KSIVI-standard-CG | on | -141.723 +/- 0.039147 | N/A | 0.020582 +/- 3.211e-05 | 0.005575 +/- 1.358e-05 | 0.053201 +/- 8.318e-05 | N/A | N/A | 1119.631s / e100000 |
| KSIVI-standard-CG | off | -145.140 +/- 0.053408 | N/A | 0.023099 +/- 4.440e-05 | 0.007885 +/- 1.866e-05 | 0.083668 +/- 9.796e-05 | N/A | N/A | 1099.545s / e100000 |
| DSIVI-default | on | **56.1210 +/- 0.245413** | N/A | 0.007271 +/- 1.512e-05 | 0.001259 +/- 9.135e-07 | 0.032348 +/- 2.001e-04 | N/A | N/A | 868.077s / e10000 |
| DSIVI-default | off | -59.2143 +/- 0.39792 | N/A | 0.007703 +/- 1.694e-05 | 0.001262 +/- 9.567e-07 | 0.052952 +/- 2.766e-04 | N/A | N/A | 914.963s / e10000 |
| DSIVI-bs4096-rbs2048 | on | 51.1508 +/- 0.42811 | N/A | 0.007615 +/- 1.389e-05 | 0.001286 +/- 1.364e-06 | 0.056571 +/- 3.551e-04 | N/A | N/A | 483.774s / e10000 |
| DSIVI-bs4096-rbs2048 | off | -10.1850 +/- 1.2211 | N/A | 0.00819 +/- 1.771e-05 | 0.001316 +/- 1.521e-06 | 0.05881 +/- 3.463e-04 | N/A | N/A | 519.267s / e10000 |
| DSIVI-bs4096-rbs4096 | on | **56.3140 +/- 0.322321** | N/A | 0.007773 +/- 1.805e-05 | 0.00127 +/- 1.301e-06 | 0.04182 +/- 2.118e-04 | N/A | N/A | 603.891s / e10000 |
| DSIVI-bs4096-rbs4096 | off | -15.7057 +/- 1.1352 | N/A | 0.008063 +/- 2.135e-05 | 0.001293 +/- 1.290e-06 | 0.065582 +/- 2.858e-04 | N/A | N/A | 606.912s / e10000 |

## LRwaveform

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| SIVI | on | 3.9933 +/- 0.008023 | N/A | N/A | N/A | 0.126645 +/- 6.054e-04 | N/A | N/A | 506.145s / e20000 |
| SIVI | off | 4.0131 +/- 0.009089 | N/A | N/A | N/A | 0.130757 +/- 5.587e-04 | N/A | N/A | 494.881s / e20000 |
| UIVI | on | 36.7131 +/- 0.04617 | N/A | N/A | N/A | **0.003396 +/- 6.874e-05** | N/A | N/A | 977.497s / e10000 |
| UIVI | off | 34.9448 +/- 0.038615 | N/A | N/A | N/A | 0.003909 +/- 8.947e-05 | N/A | N/A | 975.101s / e10000 |
| RSIVI | on | 102.671 +/- 0.182603 | N/A | N/A | N/A | 0.012552 +/- 1.347e-04 | N/A | N/A | 349.652s / e10000 |
| RSIVI | off | 61.6137 +/- 0.124882 | N/A | N/A | N/A | 0.005937 +/- 8.005e-05 | N/A | N/A | 377.850s / e10000 |
| AISIVI | off | 473.919 +/- 4.2967 | N/A | N/A | N/A | 0.040408 +/- 2.796e-04 | N/A | N/A | 348.532s / e10000 |
| KSIVI-custom | on | 28.9310 +/- 0.060269 | N/A | N/A | N/A | 0.1143 +/- 1.785e-04 | N/A | N/A | 138.921s / e20000 |
| KSIVI-custom | off | 16.2931 +/- 0.048556 | N/A | N/A | N/A | 0.17703 +/- 2.435e-04 | N/A | N/A | 154.558s / e20000 |
| KSIVI-standard-CG | on | 52.7968 +/- 0.132226 | N/A | N/A | N/A | 0.013046 +/- 5.504e-05 | N/A | N/A | 143.309s / e20000 |
| KSIVI-standard-CG | off | 26.7568 +/- 0.140573 | N/A | N/A | N/A | 0.072251 +/- 2.039e-04 | N/A | N/A | 152.897s / e20000 |
| DSIVI-default | on | 309.263 +/- 2.0344 | N/A | N/A | N/A | 0.071564 +/- 4.152e-04 | N/A | N/A | 90.3890s / e2000 |
| DSIVI-default | off | 243.617 +/- 1.0156 | N/A | N/A | N/A | 0.020611 +/- 3.106e-04 | N/A | N/A | 97.1140s / e2000 |
| DSIVI-bs4096-rbs2048 | on | **947.852 +/- 14.4338** | N/A | N/A | N/A | 0.076582 +/- 5.133e-04 | N/A | N/A | 75.2180s / e2000 |
| DSIVI-bs4096-rbs2048 | off | 290.332 +/- 2.1953 | N/A | N/A | N/A | 0.025117 +/- 3.484e-04 | N/A | N/A | 78.5190s / e2000 |
| DSIVI-bs4096-rbs4096 | on | 637.452 +/- 12.6531 | N/A | N/A | N/A | 0.077081 +/- 3.413e-04 | N/A | N/A | 81.9580s / e2000 |
| DSIVI-bs4096-rbs4096 | off | 274.807 +/- 1.4189 | N/A | N/A | N/A | 0.03676 +/- 4.632e-04 | N/A | N/A | 85.9880s / e2000 |

## Bnn_boston

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| UIVI | on | -1031.232 +/- 0.837239 | N/A | N/A | N/A | **0.284029 +/- 0.040334** | 3.6711 +/- 0.008825 | 3.3488 +/- 0.001024 | 1001.112s / e10000 |
| UIVI | off | **-1021.842 +/- 0.451484** | N/A | N/A | N/A | **0.308638 +/- 0.049937** | 3.6034 +/- 0.007805 | 3.3460 +/- 8.969e-04 | 1010.682s / e10000 |
| KSIVI-custom | on | -3605.556 +/- 2.8624 | N/A | N/A | N/A | 10.5178 +/- 0.034182 | **2.6334 +/- 8.268e-04** | **2.5084 +/- 2.132e-04** | 243.275s / e20000 |
| KSIVI-custom | off | -3605.556 +/- 2.8624 | N/A | N/A | N/A | 10.5178 +/- 0.034182 | **2.6334 +/- 8.268e-04** | **2.5084 +/- 2.132e-04** | 240.358s / e20000 |
| KSIVI-standard-CG | on | -2083.909 +/- 0.894375 | N/A | N/A | N/A | **0.347857 +/- 0.001616** | 2.7427 +/- 0.001314 | 2.5524 +/- 2.636e-04 | 276.102s / e20000 |
| KSIVI-standard-CG | off | -2072.626 +/- 1.1029 | N/A | N/A | N/A | 0.409093 +/- 0.002066 | 2.7760 +/- 0.001142 | 2.5455 +/- 2.223e-04 | 277.268s / e20000 |

## Bnn_concrete

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| SIVI | on | -2370.747 +/- 0.280428 | N/A | N/A | N/A | **0.123306 +/- 0.022035** | 12.9248 +/- 0.010722 | 4.1790 +/- 9.312e-04 | 1710.721s / e20000 |
| SIVI | off | -2370.559 +/- 0.274174 | N/A | N/A | N/A | 0.135258 +/- 0.024597 | 12.8321 +/- 0.011156 | 4.1777 +/- 9.571e-04 | 1716.910s / e20000 |
| UIVI | on | -2320.436 +/- 0.306458 | N/A | N/A | N/A | 0.148187 +/- 0.023116 | 10.2488 +/- 0.009573 | **4.1220 +/- 8.927e-04** | 1044.070s / e10000 |
| UIVI | off | **-2284.584 +/- 0.327069** | N/A | N/A | N/A | **0.080016 +/- 0.010778** | **10.1693 +/- 0.009107** | **4.1213 +/- 8.879e-04** | 1062.257s / e10000 |
| RSIVI | off | -1.864e+04 +/- 286.176 | N/A | N/A | N/A | 1.473e+04 +/- 405.007 | 143.667 +/- 0.988239 | 6.2747 +/- 0.002731 | 839.306s / e10000 |

## Bnn_power

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| SIVI | on | **-1.938e+04 +/- 0.577314** | N/A | N/A | N/A | **2.4096 +/- 0.506047** | **4.1511 +/- 6.265e-04** | **3.0521 +/- 5.610e-04** | 1152.349s / e20000 |
| SIVI | off | **-1.938e+04 +/- 0.576771** | N/A | N/A | N/A | **2.5320 +/- 0.525237** | 4.1531 +/- 6.358e-04 | **3.0530 +/- 5.695e-04** | 1157.732s / e20000 |
| UIVI | on | **-1.938e+04 +/- 0.576531** | N/A | N/A | N/A | **2.2842 +/- 0.458875** | 4.1648 +/- 9.319e-04 | 3.0574 +/- 5.745e-04 | 1031.420s / e10000 |
| UIVI | off | **-1.938e+04 +/- 0.586508** | N/A | N/A | N/A | 10.5377 +/- 1.2586 | 4.2914 +/- 0.00243 | 3.0653 +/- 6.135e-04 | 1020.840s / e10000 |
| RSIVI | on | -8.356e+08 +/- 2.308e+07 | N/A | N/A | N/A | 1.865e+14 +/- 1.360e+13 | 1.108e+04 +/- 51.5265 | 8.4864 +/- 0.015722 | 616.377s / e10000 |

## Bnn_protein

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| SIVI | on | **-1.025e+05 +/- 7.9953** | N/A | N/A | N/A | 432.443 +/- 66.6160 | 5.1908 +/- 2.239e-04 | 3.0627 +/- 3.731e-05 | 1879.212s / e20000 |
| SIVI | off | **-1.025e+05 +/- 8.0217** | N/A | N/A | N/A | 593.993 +/- 84.2611 | 5.2038 +/- 2.839e-04 | 3.0648 +/- 4.649e-05 | 1861.093s / e20000 |
| UIVI | on | **-1.026e+05 +/- 7.6177** | N/A | N/A | N/A | 2188.949 +/- 242.873 | 5.2544 +/- 4.336e-04 | 3.0744 +/- 7.233e-05 | 1042.916s / e10000 |
| UIVI | off | **-1.025e+05 +/- 7.7906** | N/A | N/A | N/A | **251.142 +/- 37.1528** | **5.1812 +/- 1.608e-04** | **3.0618 +/- 2.613e-05** | 985.611s / e10000 |

## Bnn_winered

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| UIVI | on | **-1991.778 +/- 28.4303** | N/A | N/A | N/A | **2532.761 +/- 667.922** | 0.609076 +/- 3.447e-05 | 0.920905 +/- 6.525e-05 | 1016.299s / e10000 |
| UIVI | off | **-2000.104 +/- 28.7193** | N/A | N/A | N/A | **1652.320 +/- 463.702** | **0.599755 +/- 3.361e-05** | **0.903441 +/- 5.582e-05** | 1044.161s / e10000 |
| RSIVI | on | -3.709e+09 +/- 2.826e+08 | N/A | N/A | N/A | 6.575e+16 +/- 1.066e+16 | 340.699 +/- 1.6689 | 4.9722 +/- 0.014843 | 1018.058s / e10000 |
| RSIVI | off | -1.716e+08 +/- 7.401e+06 | N/A | N/A | N/A | 1.059e+14 +/- 9.495e+12 | 152.425 +/- 0.669764 | 3.8289 +/- 0.005102 | 1031.483s / e10000 |
| AISIVI | on | -9.203e+07 +/- 2.418e+06 | N/A | N/A | N/A | 3.513e+13 +/- 2.076e+12 | 36.6328 +/- 0.17045 | 3.0879 +/- 0.002634 | 968.575s / e10000 |

## Bnn_yacht

Best cells are bolded when they are not significantly worse than the top mean at the 95% level using combined standard error.

| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |
|---------|--------|------|----|----|-----|-----|------|-----|-------------------|
| SIVI | on | -422.124 +/- 0.255489 | N/A | N/A | N/A | **1.1302 +/- 0.214748** | **2.7478 +/- 0.004078** | **2.7943 +/- 0.001693** | 1441.429s / e20000 |
| SIVI | off | -422.336 +/- 0.257884 | N/A | N/A | N/A | **1.0093 +/- 0.159364** | **2.7587 +/- 0.004052** | **2.7926 +/- 0.001571** | 1434.516s / e20000 |
| UIVI | on | **-421.659 +/- 0.295355** | N/A | N/A | N/A | 6.5273 +/- 0.684106 | 2.8679 +/- 0.005186 | 2.8161 +/- 0.001493 | 1010.898s / e10000 |
| UIVI | off | **-421.281 +/- 0.27004** | N/A | N/A | N/A | 2.8315 +/- 0.413118 | 2.7688 +/- 0.00481 | 2.8043 +/- 0.001662 | 1026.588s / e10000 |
| RSIVI | on | -1.546e+05 +/- 2788.795 | N/A | N/A | N/A | 1.406e+07 +/- 6.334e+05 | 184.251 +/- 0.98572 | 6.0539 +/- 0.007411 | 722.713s / e10000 |
| AISIVI | off | -4.291e+04 +/- 994.057 | N/A | N/A | N/A | 818.873 +/- 17.1604 | 29.4767 +/- 0.51159 | 5.3379 +/- 0.005143 | 674.475s / e10000 |

## Skipped Runs

| Target | Variant | Anneal | Reason |
|--------|---------|--------|--------|
| Bnn_boston | DSIVI-bs4096-rbs2048 | off | checkpoints_dir_missing |
| Bnn_boston | DSIVI-bs4096-rbs2048 | on | checkpoints_dir_missing |
| Bnn_boston | DSIVI-bs4096-rbs4096 | off | checkpoints_dir_missing |
| Bnn_boston | DSIVI-bs4096-rbs4096 | on | checkpoints_dir_missing |
| Bnn_boston | DSIVI-default | off | checkpoints_dir_missing |
| Bnn_boston | DSIVI-default | on | checkpoints_dir_missing |
| Bnn_concrete | DSIVI-bs4096-rbs2048 | off | checkpoints_dir_missing |
| Bnn_concrete | DSIVI-bs4096-rbs2048 | on | checkpoints_dir_missing |
| Bnn_concrete | DSIVI-bs4096-rbs4096 | off | checkpoints_dir_missing |
| Bnn_concrete | DSIVI-bs4096-rbs4096 | on | checkpoints_dir_missing |
| Bnn_concrete | DSIVI-default | off | checkpoints_dir_missing |
| Bnn_concrete | DSIVI-default | on | checkpoints_dir_missing |
| Bnn_concrete | KSIVI-custom | off | checkpoints_dir_missing |
| Bnn_concrete | KSIVI-custom | on | checkpoints_dir_missing |
| Bnn_concrete | KSIVI-standard-CG | off | checkpoints_dir_missing |
| Bnn_concrete | KSIVI-standard-CG | on | checkpoints_dir_missing |
| Bnn_power | DSIVI-bs4096-rbs2048 | off | checkpoints_dir_missing |
| Bnn_power | DSIVI-bs4096-rbs2048 | on | checkpoints_dir_missing |
| Bnn_power | DSIVI-bs4096-rbs4096 | off | checkpoints_dir_missing |
| Bnn_power | DSIVI-bs4096-rbs4096 | on | checkpoints_dir_missing |
| Bnn_power | DSIVI-default | off | checkpoints_dir_missing |
| Bnn_power | DSIVI-default | on | checkpoints_dir_missing |
| Bnn_power | KSIVI-custom | off | checkpoints_dir_missing |
| Bnn_power | KSIVI-custom | on | checkpoints_dir_missing |
| Bnn_power | KSIVI-standard-CG | off | checkpoints_dir_missing |
| Bnn_power | KSIVI-standard-CG | on | checkpoints_dir_missing |
| Bnn_protein | DSIVI-bs4096-rbs2048 | off | checkpoints_dir_missing |
| Bnn_protein | DSIVI-bs4096-rbs2048 | on | checkpoints_dir_missing |
| Bnn_protein | DSIVI-bs4096-rbs4096 | off | checkpoints_dir_missing |
| Bnn_protein | DSIVI-bs4096-rbs4096 | on | checkpoints_dir_missing |
| Bnn_protein | DSIVI-default | off | checkpoints_dir_missing |
| Bnn_protein | DSIVI-default | on | checkpoints_dir_missing |
| Bnn_protein | KSIVI-custom | off | checkpoints_dir_missing |
| Bnn_protein | KSIVI-custom | on | checkpoints_dir_missing |
| Bnn_protein | KSIVI-standard-CG | off | checkpoints_dir_missing |
| Bnn_protein | KSIVI-standard-CG | on | checkpoints_dir_missing |
| Bnn_winered | DSIVI-bs4096-rbs2048 | off | checkpoints_dir_missing |
| Bnn_winered | DSIVI-bs4096-rbs2048 | on | checkpoints_dir_missing |
| Bnn_winered | DSIVI-bs4096-rbs4096 | off | checkpoints_dir_missing |
| Bnn_winered | DSIVI-bs4096-rbs4096 | on | checkpoints_dir_missing |
| Bnn_winered | DSIVI-default | off | checkpoints_dir_missing |
| Bnn_winered | DSIVI-default | on | checkpoints_dir_missing |
| Bnn_winered | KSIVI-custom | off | checkpoints_dir_missing |
| Bnn_winered | KSIVI-custom | on | checkpoints_dir_missing |
| Bnn_winered | KSIVI-standard-CG | off | checkpoints_dir_missing |
| Bnn_winered | KSIVI-standard-CG | on | checkpoints_dir_missing |
| Bnn_yacht | DSIVI-bs4096-rbs2048 | off | checkpoints_dir_missing |
| Bnn_yacht | DSIVI-bs4096-rbs2048 | on | checkpoints_dir_missing |
| Bnn_yacht | DSIVI-bs4096-rbs4096 | off | checkpoints_dir_missing |
| Bnn_yacht | DSIVI-bs4096-rbs4096 | on | checkpoints_dir_missing |
| Bnn_yacht | DSIVI-default | off | checkpoints_dir_missing |
| Bnn_yacht | DSIVI-default | on | checkpoints_dir_missing |
| Bnn_yacht | KSIVI-custom | off | checkpoints_dir_missing |
| Bnn_yacht | KSIVI-custom | on | checkpoints_dir_missing |
| Bnn_yacht | KSIVI-standard-CG | off | checkpoints_dir_missing |
| Bnn_yacht | KSIVI-standard-CG | on | checkpoints_dir_missing |
