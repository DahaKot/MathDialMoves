# MathDialMoves

MathDial is a dataset that consists of 2861 teacher-student dialogues about math word problems.
[ArXiv paper](https://arxiv.org/abs/2305.14536)


MathDialMoves is the project that aims to build upon the MathDial dataset using its teacher strategies labels.

The link to the GitHub repository: https://github.com/DahaKot/MathDialMoves.
The link to the data files: https://drive.google.com/drive/folders/1vUc778R1PfRF85A4Dhj-vI89lXK-sgsu?usp=sharing.


# Usage
See args_parser.py for the script arguments. The most important argument is the run_name. It is used to get the dataset name (train_{run_name}.csv and test_{run_name}.csv) and to create a folder in logs.

- to generate your own data - use data_preparation.py
- to run the classification on the prepared data - run classification.py. Classification script will finetune your model and plot a confusion matrix
- to generate teacher utterances - use generation.ipynb. The evaluation is in the separate notebook - generation_evaluation.ipynb

# Disclaimer

The work on the repository continues, so consult the actual README file in the repo. At the moment there is a gap between GitHub repository and actual code due to some GitHub problems (several commits contain API key that should not be public), so all the code at the moment is avalable at: https://drive.google.com/drive/folders/1X4Lk8BK-VgA-Y8Q6CfPbmm7TKccpRDjG?usp=sharing.
