# GAN Architecture Comparison

A PyTorch comparison of **Vanilla GAN, DCGAN, and Conditional GAN (CGAN)** on **MNIST** and **Fashion-MNIST**. The project focuses on how architectural differences and class conditioning affect generated 28×28 grayscale images.

## Models

| Model | Generator / Discriminator | Key idea |
|---|---|---|
| Vanilla GAN | Fully connected layers | Baseline adversarial image generation |
| DCGAN | ConvTranspose2d / Conv2d | Learns spatial image features with convolutional layers |
| CGAN | Fully connected layers + class condition | Generates images conditioned on class labels |

### CGAN implementation note

In this implementation, each scalar class label is expanded to the same width as the latent vector for the Generator and to the flattened image width for the Discriminator. The expanded condition is then concatenated with the corresponding input. It does **not** use one-hot encoding or a learned label embedding.

## Experiment Setup

- Datasets: MNIST, Fashion-MNIST
- Image size: 28 × 28, grayscale
- Latent dimension: 100
- Batch size: 100
- Optimizer: Adam
- Generator learning rate: 0.0002
- Discriminator learning rate: 0.0002
- Adam betas: (0.5, 0.999)
- Maximum epochs: 40
- Loss: Binary Cross Entropy

## Generated Samples

Generated samples from the original coursework are included in the `results/` directory for all six dataset/model combinations.

The submitted coursework does not include a quantitative generation metric such as FID, so this repository presents the comparison as an architectural and qualitative experiment rather than claiming a numerical winner.

## Project Structure

```text
gan-architecture-comparison/
├── config/              # MNIST/Fashion-MNIST × 3 model configs
├── models/
│   ├── gan.py           # Vanilla GAN + shared GAN losses
│   ├── dcgan.py         # DCGAN
│   └── cgan.py          # Conditional GAN
├── scripts/
│   └── train_gan.py
├── results/             # Six generated sample images
├── .gitignore
├── README.md
└── requirements.txt
```

## Run

From the repository root:

```bash
python scripts/train_gan.py config/mnist_vanilla.yaml
python scripts/train_gan.py config/mnist_dcgan.yaml
python scripts/train_gan.py config/mnist_cgan.yaml
```

Replace `mnist` with `fashion` to run the Fashion-MNIST experiments. Torchvision downloads the selected dataset automatically.

## Portfolio Refactoring

The original coursework used a course-specific `DL_Lecture` import path. This portfolio version replaces it with local module imports, provides separate relative configuration files for all six experiments, and uses the actual current batch size when sampling latent vectors so the training loop also handles a smaller final batch.
