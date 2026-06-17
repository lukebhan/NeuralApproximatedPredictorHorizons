<div align="center">
  <a href="https://ccsd.ucsd.edu/home">
    <img align="left" src="media/CCSD.png" width="400" height="60" alt="ccsd">
  </a>
  <a href="https://ucsd.edu/">
    <img align="right" src="media/ucsd(1).png" width="260" alt="ucsd">
  </a>
</div>

<br> <br>

# Neural Operator Predictors for Delay-Compensated Nonlinear Stabilization
<div align="center">
 <a href="#"><img alt="Main result from the paper" src="media/manipulatorFig.jpg" width="100%"/></a>
</div>

<br> 

[Paper](https://arxiv.org/pdf/2411.18964) |
[GitHub](https://github.com/lukebhan/NeuralOperatorPredictorFeedback) |
[arXiv](https://arxiv.org/pdf/2411.18964) |
Published in L4DC 2025 (Oral) + Submitted to IEEE TAC.

## Introduction

This repository is the official implementation for the papers titled "Neural Operator Predictors for Delay-Compensated Nonlinear Stabilization" and "Neural Operators for Predictor Feedback Control of
Nonlinear Delay Systems". Please refer to the example below for a quick notebook reproducing the paper results. We have included pretrained datasets and models as well as the code to train
one's own models and dataset if interested. 

Note, since the original L4DC publication, these result have been improve and updated to the latest packages as of May 2025. Thus, the direct numerical values may differ slightly from those published in
the paper, but the overall result remains the same. 

## Getting Started - Installation, Package Versioning, Datasets, and Models.
- To get started, please setup your virtual envrionment ([Virtual Env tutorial](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)) and install the corresponding packages and versions given in `requirements.txt`.
- Additionally, we have published both the dataset and models on huggingface. Please clone the repositories below following the instructions on huggingface and place the resulting
files in the dataset and models folders of this repository (See below for structure). 
  - [Hugging face: Dataset](https://huggingface.co/datasets/lukebhan/NeuralOperatorsPredictorFeedbackDataset)
  - [Hugging face: Models](https://huggingface.co/lukebhan/NeuralOperatorsPredictorFeedbackModels)
 
    
<br>

   >The repository directory should look like this:
  ```
  NeuralOpertorPredictorFeedback/
  ├── examples/
  ├── src/
  ├── config/
  ├── datasets/ # Place the cloned datasets here
  │   ├── ManipulatorDatasets/    
  │   ├── UnicycleConstDelayDataset/   
  │   ├── UnicycleTimeVaryingDelayDataset/
  ├── models/ # Place the cloned models here
  │   ├── ManipulatorModels/    
  │   ├── UnicycleConstDelayModels/   
  │   ├── UnicycleTimeVaryingDelayModels/
  ├── media/ 
  •   •   •
  •   •   •
  ```


## Unicycle Example - Constant and Time-Varying Delays
See `examples` folder for two notebooks that have implementations of neural operator predictors for both constant and time-varying delays for the
unicycle system. These models are quick to train and thus can be run with any modern GPU system (and higher end laptops) although we do provide pretrained models for 
those who are interested. The Jupyter-notebooks should be self-explanatory as they have section headers guiding one through each portion of the code. 

## Manipulator Example - Constant Delays
See `examples` folder for a third notebook that replicates the manipulator results for a 5DOF manipulator that contains the same parameters as a single arm of a Baxter manipulator without the last two joints. To generate a dataset for the manipulator, it takes significant computation time as the example's forward dynamics are expensive. Furthermore, for the models, they can be trained either through the notebook or using the  `train_model.py` file
given that one modifies the configuration. Without significant computing resources, it is not recommended that the user develops their own models for the manipulator. 

## Assistance / Troubleshooting
If you have any issues with any of the notebooks or models in this repo, feel free to create an issue in this github repo or email lbhan@ucsd.edu and I am more than happy to assist! 

### Citation 
If you found this work useful or interesting for your own research, we would appreciate if you could cite our work:
```
@misc{bhan2025neuraloperatorspredictorfeedback,
      title={Neural Operators for Predictor Feedback Control of Nonlinear Delay Systems}, 
      author={Luke Bhan and Peijia Qin and Miroslav Krstic and Yuanyuan Shi},
      year={2025},
      eprint={2411.18964},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2411.18964}, 
}
```

### Licensing
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

