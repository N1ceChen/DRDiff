
# DRDiff
This is the implementation code for "Downscaling of satellite passive microwave brightness temperature through super-resolution reconstruction"

# Project Structure
```text
DRDiff/
├── datasets.py             # Data preprocessing
├── diffusion/
│   └── Res_Diffusion.py    # Diffusion settings
├── arch/
│   ├── losses.py           # Training losses
│   ├── module.py           # U-Net modules
│   ├── tools.py            # Evaluation metrics
│   └── unet.py             # Prediction network
├── config/
│   └── config.yaml         # Training settings
├── model/
│   └── DRDiff_model.py     # DRDiff trainer
├── main.py                 # Model training
└── eval.py                 # Model evaluation and downscaling
```

# Citation
If you find this work useful, please cite:

**Paper:** [Downscaling of satellite passive microwave brightness temperature through super-resolution reconstruction](https://doi.org/10.1016/j.rse.2026.115405)
```bibtex
@article{CHEN2026115405,
title = {Downscaling of satellite passive microwave brightness temperature through super-resolution reconstruction},
journal = {Remote Sensing of Environment},
volume = {339},
pages = {115405},
year = {2026},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2026.115405},
author = {Jiaxin Chen and Ji Zhou and Tao Zhang and Shaojie Zhao and Ruyin Cao and Jin Ma and Wenbin Tang and Lin Feng},
}
