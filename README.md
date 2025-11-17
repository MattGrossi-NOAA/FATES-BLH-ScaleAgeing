# Menhaden Ageing Model

This Menhaden Ageing Model provides an innovative method for automatically estimating Menhaden fish age using scale images and fish length, weight, and month of catch (hereafter, “metadata.”) Built upon state-of-the-art deep learning algorithms, the model enables rapid generation of fish age predictions by simply pointing to a directory containing configuration file that instructs the model where to find the data and metadata, how to preprocess the images, and where to write the output. See the [official documentation pages](https://sefsc.github.io/FATES-BLH-ScaleAgeing/) for more thorough instructions.

This repo consists of four branches:
- `main`: The main branch containing the most stable version of the code and a series of Jupyter notebooks demonstrating the workflow.
- `dev`: A development branch where new features and updates are tested before being merged into the main branch. Any modification to and testing of the model or accompanying notebooks should be done here (or a new development branch created from `main`, if desired) and merged into `main` once changes are verified to be stable.
- `docs`: A branch dedicated to hosting the documentation for the project. This should be updated as needed whenever changes to the model are made in `main` to ensure the model documentation remains up-to-date.
- `gh-pages`: A branch used for GitHub Pages to serve the project's documentation website, which happens automatically using GitHub Actions whenever changes are pushed to the `docs` branch. There is no need to modify or even clone this branch.

*This repo consists of four primary branches, each serving a distinct purpose to facilitate development, documentation, and model deployment.*
| Branch     | Description                                                    |
|------------|----------------------------------------------------------------|
| `main`     | Contains the most stable version of the code and a series of Jupyter notebooks demonstrating the workflow. |
| `dev`      | A development branch where new features and updates are tested before being merged into the main branch. Any modification to and testing of the model or accompanying notebooks should be done here (or a new development branch created from `main`, if desired) and merged into `main` once changes are verified to be stable. |
| `docs`     | Hosts the documentation for the project. This should be updated as needed whenever changes to the model are made in `main` to ensure the model documentation remains up-to-date. |
| `gh-pages` | Orphan branch for GitHub Pages to serve the project's documentation website, updated automatically. |

## Prerequisites

This model is built using Python 3.8. The recommended way to install the required packages is to use [Conda](https://docs.conda.io/en/latest/) to create a new environment from the provided `environment.yml` file. It also has the option to implement [Segment Anything Model](https://arxiv.org/abs/2408.00714) for image segmentation, which requires downloading a model checkpoint. See the [docs page](https://sefsc.github.io/FATES-BLH-ScaleAgeing/content/setup.html#download-segmentation-model) for more information.

## Usage

First, create a conda virtual environment using the provided `environment.yml` file, which will install all necessary dependencies, and activate it. Next, edit the `config.yaml` file to specify the paths to your data, metadata, and output directory. See the [docs](https://sefsc.github.io/FATES-BLH-ScaleAgeing/content/configuration.html) for details on each configuration option.

Raw images need to be processed before passing them to the ageing model:

```bash
python process-images.py --config-path configurations.yml
```

This utility takes an image that may contain multiple scales, identifies the center-most scale, crops it out into a square, and resizes it to the dimensions expected by the model. The processed images are saved to a new directory specified in the configuration file.

Finally, run the model to generate age predictions on the processed images:

```bash
python predict-ages-multimodal.py --config-path configurations.yml
```

Deactivate the conda environment when done.

<hr>

### Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project content is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

### License

This content was created by U.S. Government employees as part of their official duties. This content is not subject to copyright in the United States (17 U.S.C. §105) and is in the public domain within the United States of America. Additionally, copyright is waived worldwide through the CC0 1.0 Universal public domain dedication.
