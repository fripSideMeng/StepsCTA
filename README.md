# StepsCTA

## Third-Party Licenses

This project includes third-party code governed by the following licenses:

1. **Megagon Labs' Doduo Code** (Licensed under Apache License 2.0)  
   - The original source is available at: [https://github.com/megagonlabs/doduo](https://github.com/megagonlabs/doduo).  
   - A copy of this license is provided in `LICENSES/Apache_2.0_Megagon`.

2. **Alibaba-MIIL's ASL** (Licensed under MIT License)  
   - The original source is available at: [https://github.com/Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL).  
   - A copy of this license is provided in `LICENSES/MIT_Alibaba_MIIL`.

For the reproduction of baselines, we follow the official codes provided by DODUO and Watchog at [https://github.com/megagonlabs/doduo](https://github.com/megagonlabs/doduo) and [https://github.com/megagonlabs/watchog](https://github.com/megagonlabs/watchog), respectively.

## Installation

Make sure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt
```

This project requires **Python 3.9.0**.

## Datasets and Models

The datasets splits can be downloaded from the following link:

- [Download Link](https://drive.google.com/file/d/1lFBm0nr5q923l60GQ-iCZIoq4VUF8Kth)

After downloading, create data/ folder in the project root and put all the pickle files there.

The models at 2% labeled ratio can be downloaded from:

- [Download Link](https://drive.google.com/file/d/1eYPGXETsLyl1Y4lU0f3wWPaMbQRh4vtg/edit)

Create bert/ and distil-bert/ folder in the project root for the teacher and the student respectively. Put either the provided trained models, or pre-trained models from HuggingFace there. We will add support for more flexible path specification from command line arguments later.

## Running the Code

### Training the Teacher Model

#### On WikiTable Dataset:
```bash
python -u distil_teacher_feedback_wiki.py --batch_size 16 --lr 5e-5 --tasks turl --true_ratios turl=<labeled ratio> --epoch 30 --teacher_confidence <confidence> --output_hidden_states --feedback
```

#### On VizNet Dataset:
```bash
python -u distil_teacher_feedback_viz.py --temp 0.5 --teacher_confidence <confidence> --epoch 30 --batch_size 16 --lr 5e-5 --tasks sato --true_ratios sato=<labeled ratio> --output_hidden_states --teacher_dropout 0.1 --pl_loss <loss func type>
```

### Training the Student Model

#### On WikiTable Dataset:
```bash
torchrun --nproc_per_node=<Number of GPUs> --nnodes=1 distil_student_wiki.py --batch_size 16 --lr 5e-5 --tasks turl --true_ratios turl=<labeled ratio> --epoch 15 --teacher_confidence <confidence> --output_hidden_states
```

#### On VizNet Dataset:
```bash
torchrun --nproc_per_node=<Number of GPUs> --nnodes=1 distil_student_viz.py --batch_size 16 --lr 5e-5 --tasks sato --true_ratios sato=<labeled ratio> --epoch 15 --teacher_confidence <confidence> --train_temp 0.5 --pl_loss <loss func type> --embed_loss <hidden state alignment loss func type> --output_hidden_states
```

## Configuration Parameters

- `<confidence>`: Confidence threshold (configurable).
- `<labeled ratio>`: Can be `0.02`, `0.05`, or `0.1`.
- `<loss func type>`: Can be `ce` (Cross Entropy), `kl` (KL Divergence), or `mse` (MSE Loss).
- `<hidden state alignment loss func type>`: Can be `mse` (MSE Loss) or `cos` (Cosine Similarity Loss).

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.