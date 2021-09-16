import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
import torch.optim as optim
import numpy as np

if torch.cuda.is_available():
    print("cuda is available")

TRUTH_WEIGHT = 5 # The weight applied to the truth prediction in comparison to the classification prediction

trainset = MNIST('.', train = True, download = True, transform = transforms.ToTensor())
testset = MNIST('.', train = False, download = True, transform = transforms.ToTensor())
trainloader = DataLoader(trainset, batch_size=128, shuffle = True)
testloader = DataLoader(testset, batch_size=128, shuffle = True)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


device = "cuda" if torch.cuda.is_available() else "cpu"

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)




generator = Generator(64, 28*28).cuda()
discriminator = Discriminator(28*28).cuda()

def test_generator():
    print()
    print("TEST GENERATOR")
    z_noise = torch.zeros(1,64).cuda()

    print("z_noise.shape", z_noise.shape)
    # z_noise = torch.randn(1,64,1,1).cuda()
    output = generator.forward(z_noise).cuda()
    print("fake_image_shape",output.shape)
    # print("fake_image:", output)
    plt.imshow(output.reshape(28,28).detach().cpu().numpy())
    plt.show()

# test_generator()

def test_transform_image():
    for image, _ in trainloader:
        image = image[0].squeeze()
        image = image.reshape(-1, 784)
        image = image.reshape(28,28)
        print(image.shape)
        plt.imshow(image)
        plt.show()

# test_transform_image()




def test_discriminator():
    print()
    print("TEST DISCRIMINATOR")
    test_layer = nn.Conv2d(1, 256, stride = 2, kernel_size = 3, padding = 0)
    for image, label in trainloader:
      image = image.cuda()
      image = image.view(-1, 28*28)
      print("entry image shape:",image[0].shape)
      # print("entry image:",image[0])
      x = discriminator(image)
      print("discriminator result: ",x)
      break

# test_discriminator()


def test_images():
    for image, label in trainloader:
      image_grid = make_grid(image[0:25], nrow = 5, )
      print("image_grid.shape",image_grid.shape)
      print("image.shape",image.shape)
      plt.imshow(image_grid.permute(1,2,0))
      print(image[0].shape)
      break
    plt.show()

# test_images()

def generate_noise_vector(n_images):
    z_noise = torch.randn(n_images,64).cuda()
    return z_noise



def test_generate_noise_vector():
    print()
    print("TEST_GENERATE_NOISE_VECTOR")

    z_noise = generate_noise_vector(128)
    print(z_noise.shape)
    print(generator(z_noise).shape)
    output_images = generator(z_noise).reshape(-1,28,28)
    plt.imshow(output_images[0].cpu().detach().numpy())
    plt.show()
    # print(generator(z_noise))

# test_generate_noise_vector()

def test_generator_discriminator():
    print("TEST GENERATOR DISCRIMINATOR")
    z_noise = torch.randn(10,64).cuda()
    output = generator.forward(z_noise)
    classification = discriminator(output)
    print(classification)

# test_generator_discriminator()

criterion = nn.BCELoss()
# print("criterion_TEST", criterion(torch.Tensor([0.9, 0.8,0.7]).float(),torch.tensor([0.5,0.2,0.4]).float()))
lr = 3e-4
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = optim.Adam(generator.parameters(), lr=lr)

def test_generator_discriminator_criterion():
    print()
    print("TEST GENERATOR DISCRIMINATOR CRITERION")
    z_noise = generate_noise_vector(128)
    fake_img_flat = generator(z_noise)
    # logits_prediction = discriminator(output)[0]
    prediction = discriminator(fake_img_flat)
    print("truth_pred:", prediction)
    # classification_loss = criterion(logits_prediction, torch.tensor([1]).cuda())
    loss = criterion(prediction,torch.tensor([[1]]*128).float().cuda() )
    print("loss", loss)
# test_generator_discriminator_criterion()


def calculate_loss(prediction, fake):
    if fake:
        loss = criterion(prediction, torch.zeros_like(prediction).float().cuda())
        return loss

    else:
        loss = criterion(prediction, torch.ones_like(prediction).float().cuda())
        return loss



#### TRAINING LOOP

def train(n_batchs):
    # for epoch in range(2):  # loop over the dataset multiple times
    running_lossG = 0.0
    running_lossD = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        real, _ = data
        real = real.cuda()
        real = real.view(-1, 28*28)
        noise = generate_noise_vector(128)
        fake = generator(noise)
        # zero the parameter gradients


        ### TRAIN DISC
        prediction_real = discriminator(real)
        loss_realD = criterion(prediction_real, torch.ones_like(prediction_real))


        prediction_fake = discriminator(fake)
        loss_fakeD = criterion(prediction_fake, torch.zeros_like(prediction_fake))
        lossD = loss_fakeD + loss_realD
        optimizer_discriminator.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_discriminator.step()


        ### TRAIN GEN
        prediction_fake = discriminator(fake)
        lossG = criterion(prediction_fake, torch.ones_like(prediction_fake))
        optimizer_generator.zero_grad()
        lossG.backward()

        optimizer_generator.step()
        # print statistics
        running_lossG += lossG.item()
        running_lossD += lossD.item()

        if i == n_batchs:
            break
    # print('discriminator loss:', running_loss / n_batchs)
    final_lossG = running_lossG/n_batchs
    final_lossD = running_lossD/n_batchs
    return final_lossG, final_lossD

# torch.save(discriminator.state_dict(), 'discriminator')
# discriminator.load_state_dict(torch.load('discriminator'))

def test_training_works():
    print()
    print("TEST_TRAINING_WORKS")
    LossG, LossD = train(20)
    print("discriminator_loss", LossG, LossD)
# test_training_works()


def test_gan_fake_input():
    print()
    print("TEST_GAN_FAKE_INPUT")
    noise_vector = generate_noise_vector(1)
    image = generator(noise_vector)
    predictions = discriminator(image)
    print(predictions)
    plt.imshow((image.reshape(-1,28,28))[0].cpu().detach().numpy(), cmap="gray")
    plt.show()
# test_gan_fake_input()

def test_gan_real():
    for i, data in enumerate(testloader, 0):
        images, label = data
        images = images.cuda()
        images = images.view(-1, 784)
        predictions = discriminator(images)
        print("prediction.shape", predictions.shape)
        print("predictions",predictions)
        break
# test_gan_real()

def check_entry_images():
    for data, label in trainloader:
        print(data.shape)
        print(label)
        plt.imshow(torch.squeeze(data[0]).numpy())
        plt.show()
        break

def check_params(model):
    for param in model.parameters():
        print(param[0])
        print(param.shape)
        break
# check_params(discriminator)


if __name__ == '__main__':

    LENGTH_CYCLE = 20
    SEED = generate_noise_vector(25)
    def generate_fake_image_grid(path):
        z_noise = SEED
        fake_images = generator(z_noise).reshape(-1,1, 28,28).cpu().detach()
        image_grid = make_grid(fake_images[0:25], nrow = 5, )
        # print(image_grid.shape, "image_grid.shape")
        plt.imshow(image_grid.permute(1,2,0))
        plt.savefig(path)

    # generate_fake_image_grid()

    def full_train(n_cycle):
        gen_loss, disc_loss = train(1)

        for i in range(n_cycle):
            gen_loss, disc_loss = train(LENGTH_CYCLE)
            print("step",  i, "gen_loss:", gen_loss, "disc_loss:", disc_loss)
            if i % 50 == 49:
                print("save")
                generate_fake_image_grid("result_images2/image_grid"+str(i//50))
                torch.save(discriminator.state_dict(), 'weights/discriminator'+ "_CYCLE_"+str(LENGTH_CYCLE))
                torch.save(generator.state_dict(), 'weights/generator' + "_CYCLE_"+str(LENGTH_CYCLE))

    # discriminator.load_state_dict(torch.load('weights/discriminator2_gen20_disc2'))
    # generator.load_state_dict(torch.load('generator_weights/generator2_gen20_disc2'))

    # discriminator.load_state_dict(torch.load('weights/discriminator'))
    # generator.load_state_dict(torch.load('generator_weights/generator'))
    #


    def load_models():
        try:
            discriminator.load_state_dict(torch.load('weights/discriminator'+ "_CYCLE_"+str(LENGTH_CYCLE)))
            generator.load_state_dict(torch.load('generator_weights/generator'+ "_CYCLE_"+str(LENGTH_CYCLE)))
            print("loaded models")
        except:
            print("couldn't load models, training from scratch")
    # load_models()

    full_train(5000)


