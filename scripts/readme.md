# Model Scripts

All model scripts are contained here:

* `process-images.py`: Process raw scale images (crop, pad, normalize)
* `predict-ages-multimodal.py`: Model that predicts ages from processed images and accompanying metadata. This is the preferred model as of September 2025.
* `predict-ages-image-only.py`: Model that predicts ages from processed images only.

All three scripts use the the provided `configurations.yml` file for configuration. Edit this file accordingly.

The models themselves, which consist of weights and hyperparameters obtained during training, are stored as `.pth` files in the `models` directory. Be sure to point to the correct model file in `configurations.yml`.

Finally, additional scripts to train new models are provided in the `training-scripts` directory.

Basic instructions for running these scripts are provided in the `README.md` file in the root directory of this repository, wherein a link to the official documentation pages containing more detailed information and instructions is also provided.