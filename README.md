# Menhaden Ageing Model

This Menhaden Ageing Model provides an innovative method for automatically estimating Menhaden fish age using scale images and fish length, weight, and month of catch (hereafter, “metadata.”) Built upon state-of-the-art deep learning algorithms, the model enables rapid generation of fish age predictions by simply pointing to a directory containing configuration file that instructs the model where to find the data and metadata, how to preprocess the images, and where to write the output. See the [official documentation pages](https://sefsc.github.io/FATES-BLH-ScaleAgeing/) for more thorough instructions.

This repo consists of four primary branches, each serving a distinct purpose to facilitate development, documentation, and model deployment:
- `main`: The main branch containing the most stable version of the code and a series of Jupyter notebooks demonstrating the workflow.
- `dev`: A development branch where new features and updates are tested before being merged into the main branch. Any modification to and testing of the model or accompanying notebooks should be done here (or a new development branch created from `main`, if desired) and merged into `main` once changes are verified to be stable.
- `docs`: A branch dedicated to hosting the documentation for the project. This should be updated as needed whenever changes to the model are made in `main` to ensure the model documentation remains up-to-date.
- `gh-pages`: A branch used for GitHub Pages to serve the project's documentation website. This site is updated automatically using GitHub Actions whenever changes are pushed to the `docs` branch. There is no need to clone or manually this branch.

## Prerequisites

This model is built using Python 3.8 but currently works with versions up to Python 3.10. A Python virtual environment and package dependencies can be managed using either [Conda](https://docs.conda.io/en/latest/) and the provided `environment.yml` file, or [pip](https://pypi.org/project/pip/) with the provided `requirements.txt` file. The model has the option to implement [Segment Anything Model](https://arxiv.org/abs/2408.00714) for image segmentation, which requires downloading a model checkpoint. See the [docs page](https://sefsc.github.io/FATES-BLH-ScaleAgeing/content/setup.html#download-segmentation-model) for more information.

## Usage

The easiest way to launch the model is with the provided `menhaden-age-model` utility, a Windows Batch (`.bat`) file designed to streamline Python environment management and script execution. Behind the scenes, this utility checks the workstation for an existing Python installation[^1], creates and manages a Python virtual environment and necessary package dependencies, and handles launching the necessary Python scripts that contain the ageing model and associated data processing routines.

[^1]: If no compatible Python version is found, the user will be prompted to download and install Python 3.10. Be sure to select the option to add Python to your PATH environment variables during the installation process.

Upon launching `menhaden-age-model`, the user is asked to select a process from the following options:

- **Process raw images (crop, pad, normalize, etc.)**: Process raw scale images prior to passing them through an age prediction model. The image processing routine takes an image that may contain multiple scales, identifies and crops the center-most scale, resizes it to square dimensions expected by the model, adds necessary padding around the cropped scale, and optionally applies image pixel normalization or other image processing techniques.
- **Predict ages using images only**: Runs the "image-only" ageing model, which predicts ages using only the cropped scale images.
- **Predict ages using images and metadata**: Runs the so-called "multimodal" ageing model, which combines sample metadata (*e.g.*, fish length, weight, month of catch) with processed scale images to predict age.

Regardless of which pipeline is selected, the user will then be prompted to select the desired configuration `YAML` file. See the [docs](https://sefsc.github.io/FATES-BLH-ScaleAgeing/content/configuration.html) for more information and details on each configuration option.

Command line options for using the Menhaden ageing model are also available. See the [docs](https://sefsc.github.io/FATES-BLH-ScaleAgeing/content/usage.html) for more information.

<hr>

### Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project content is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

### License

This content was created by U.S. Government employees as part of their official duties. This content is not subject to copyright in the United States (17 U.S.C. §105) and is in the public domain within the United States of America. Additionally, copyright is waived worldwide through the CC0 1.0 Universal public domain dedication. The United States/Department of Commerce reserve all rights to seek and obtain copyright protection in countries other than the United States for Software authored in its entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the Software outside of the United States.
