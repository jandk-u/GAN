import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)


# create block
def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.02, inplace=True)
    )


# create noise vector
def get_noise(n_sample, z_dim, device="cpu"):
    return torch.randn(n_sample, z_dim, device=device)


# create model
# model Generator
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784,hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim*2),
            get_generator_block(hidden_dim*2, hidden_dim*4),
            get_generator_block(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen


# model Discriminator
class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc


# hyperparameters
n_epochs = 20
z_dim = 64
batch_size = 64
lr = 3e-4
display_step = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# init data
# download = True if dataset doesn't exist
dataloader = DataLoader(
    MNIST('./datasets', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)


disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr)
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr)
criterion = nn.BCEWithLogitsLoss()
fixed_noise = get_noise(batch_size, z_dim)
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

mean_disc_loss = 0
mean_gen_loss = 0

# training
for epoch in range(n_epochs):
    for batch_idx, (real, _) in enumerate(tqdm(dataloader)):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train Discriminator: max log(D(real)) + log(1-D(G(z)))
        noise = get_noise(batch_size, z_dim)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        disc_opt.step()

        # Train Generator min log(1-D(G(z)))
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        gen_opt.step()

        if batch_idx %100 == 0:
            print(f"\nEpoch [{epoch}/{n_epochs}] Batch {batch_idx}/{len(dataloader)} \
                        Loss disc: {loss_disc:.4f}, Loss gen: {loss_gen:.4f}")

            with torch.no_grad():
                noise = get_noise(batch_size, z_dim)
                fake = gen(noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                writer_fake.add_image("MNIST fake images", img_grid_fake, global_step=step)
                writer_real.add_image("MNIST real images", img_grid_real, global_step=step)
                step += 1


# save sate dict
torch.save(gen.state_dict(), "model/GAN/gen.pth")
torch.save(disc.state_dict(), "model/GAN/disc.pth")