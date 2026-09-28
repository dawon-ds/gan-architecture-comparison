import os
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.cgan import conditional_D, conditional_G
from models.dcgan import dcgan_D, dcgan_G
from models.gan import D_Loss, G_Loss, vanilla_D, vanilla_G


def main():
    params_filename = sys.argv[1] if len(sys.argv) >= 2 else "config/fashion_dcgan.yaml"
    with open(params_filename, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    seed = params.get("random_seed", 54321)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    if params["task"] == "MNIST":
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    elif params["task"] == "Fashion":
        train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    else:
        raise ValueError(f"Unsupported task: {params['task']}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
    )

    if params["model"] == "Vanilla":
        G = vanilla_G(params["z_dim"], params["img_size"]).to(device)
        D = vanilla_D(params["img_size"]).to(device)
    elif params["model"] == "CGAN":
        G = conditional_G(params["z_dim"], params["img_size"]).to(device)
        D = conditional_D(params["img_size"]).to(device)
    elif params["model"] == "DCGAN":
        G = dcgan_G(params["z_dim"], params["img_size"]).to(device)
        D = dcgan_D(params["img_size"]).to(device)
    else:
        raise ValueError(f"Unsupported model: {params['model']}")

    g_loss_fn = G_Loss(device)
    d_loss_fn = D_Loss(device)

    g_optim = torch.optim.Adam(
        G.parameters(), lr=params["lr_G"], betas=(params["beta1"], params["beta2"])
    )
    d_optim = torch.optim.Adam(
        D.parameters(), lr=params["lr_D"], betas=(params["beta1"], params["beta2"])
    )

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join("runs", timestamp))
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    images_dir = os.path.join(out_dir, "images")
    summary_dir = os.path.join(out_dir, "summaries")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    writer = SummaryWriter(summary_dir)

    if params["model"] in ["Vanilla", "DCGAN"]:
        eval_z = torch.randn(params["num_show_img"], params["z_dim"], device=device)
        eval_c = None
    else:
        eval_c = torch.tensor(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
            dtype=torch.float32,
            device=device,
        )
        eval_z = torch.randn(eval_c.shape[0], params["z_dim"], device=device)

    def to_img(x):
        return torch.clamp((x + 1) / 2, 0, 1)

    start_time = time.time()
    global_steps = 0

    for epoch in range(params["max_epochs"]):
        for images, labels in train_loader:
            G.train()
            D.train()
            images = images.to(device)
            labels = labels.float().to(device)
            batch_size = images.size(0)

            z = torch.randn(batch_size, params["z_dim"], device=device)
            if params["model"] in ["Vanilla", "DCGAN"]:
                fake_images = G(z)
                d_real = D(images)
                d_fake = D(fake_images.detach())
            else:
                fake_images = G(z, labels)
                d_real = D(images, labels)
                d_fake = D(fake_images.detach(), labels)

            d_loss, d_real_loss, d_fake_loss = d_loss_fn(d_real, d_fake)
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            z = torch.randn(batch_size, params["z_dim"], device=device)
            if params["model"] in ["Vanilla", "DCGAN"]:
                fake_images = G(z)
                g_fake = D(fake_images)
            else:
                fake_images = G(z, labels)
                g_fake = D(fake_images, labels)

            g_loss = g_loss_fn(g_fake)
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

            writer.add_scalar("Batch/G_Loss", g_loss.item(), global_steps)
            writer.add_scalar("Batch/D_Loss", d_loss.item(), global_steps)
            global_steps += 1

            if global_steps % 300 == 0:
                print(
                    f"Epoch [{epoch + 1}], Step [{global_steps}], "
                    f"G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}"
                )
                G.eval()
                with torch.no_grad():
                    samples = G(eval_z) if eval_c is None else G(eval_z, eval_c)
                save_image(to_img(samples), os.path.join(images_dir, f"gen_imgs_{global_steps}.jpg"))

        torch.save(
            {"epoch": epoch + 1, "model_state_dict": G.state_dict()},
            os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_G.pth"),
        )
        torch.save(
            {"epoch": epoch + 1, "model_state_dict": D.state_dict()},
            os.path.join(checkpoint_dir, f"epoch_{epoch + 1}_D.pth"),
        )

        training_time = (time.time() - start_time) / 60
        print("========================================")
        print(f"epoch: {epoch + 1} / global_steps: {global_steps}")
        print(
            f"LOSS value G: {g_loss.item():.4f} / "
            f"D(r, f): {d_loss.item():.4f} "
            f"({d_real_loss.item():.4f}, {d_fake_loss.item():.4f})"
        )
        print(f"training_time: {training_time:.2f} minutes")

    writer.close()


if __name__ == "__main__":
    main()
