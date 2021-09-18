import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm


class Critic(nn.Module):
    def __init__(self, img_channels, features_d):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, img_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(img_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, img):
        return self.critic(img)


class Generator(nn.Module):
    def __init__(self, z_dim, img_channels, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0),
            self._block(features_g*16, features_g*8, 4, 2, 1),
            self._block(features_g*8, features_g*4, 4, 2, 1),
            self._block(features_g*4, features_g*2, 4, 2, 1),
            nn.ConvTranspose2d(features_g*2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, img_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(img_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        )
    ]
)

datasets = datasets.ImageFolder(root="datasets/celeb_dataset", transform=transforms)
loader = DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

optim_critic = optim.RMSprop(critic.parameters(), lr=LR)
optim_gen = optim.RMSprop(gen.parameters(), lr=LR)

writer_fake = SummaryWriter(f"runs/WGAN/fake")
writer_real = SummaryWriter(f"runs/WGAN/real")

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
gen.train()
critic.train()
step = 0

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(tqdm(loader)):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            optim_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        # train Generator
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            gen.eval()
            critic.eval()
            print(f"\nEpoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(loader)}"
                  f" Loss Critic: {loss_critic:.4f} Loss Gen {loss_gen:.4f}")
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_real = torchvision.utils.make_grid(real[:32], normalize=True)
                writer_real.add_image("real", img_real, global_step=step)
                writer_fake.add_image("fake", img_fake, global_step=step)

            step += 1
            gen.train()
            critic.train()


torch.save(gen.state_dict(), "model/WGAN/gen.pth")
torch.save(critic.state_dict(), "model/WGAN/critic.pth")
