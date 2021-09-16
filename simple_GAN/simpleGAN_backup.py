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


#
# class Generator(nn.Module):
#   def __init__(self):
#     super().__init__()
#
#     self.block1 = self.make_gen_block(64, 1024)
#     self.block2 = self.make_gen_block(1024, 512, stride=1)
#     self.block3 = self.make_gen_block(512, 256,kernel_size=3)
#     self.block4 = self.make_gen_block(256, 128,kernel_size=3, stride = 1)
#     self.block5 = self.make_gen_block(128, 1, kernel_size =4, padding = 2, final = True)
#
#
#   def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final = False, padding=0):
#     if not final:
#       return nn.Sequential(
#           nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride = stride, padding = padding),
#           nn.BatchNorm2d(output_channels),
#           nn.LeakyReLU(),
#       )
#     else:
#       return nn.Sequential(
#           nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride = stride),
#           nn.Tanh(),
#       )
#
#   def forward(self, input):
#     # print("input generator", input.shape)
#
#     # print(x.shape)
#
#     # print("shape after fc0",x.shape)
#
#     # print(x.shape)
#
#     x = self.block1(input)
#     x = self.block2(x)
#     x = self.block3(x)
#     x = self.block4(x)
#     x = self.block5(x)
#     # print("final shape", x.shape)
#     return x

generator = Generator(64, 28*28).cuda()

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

# class Discriminator(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.layer1 =  self.make_disc_block(1, 256)
#     self.layer2 =  self.make_disc_block(256, 512)
#     self.layer3 =  self.make_disc_block(512, 1024, stride=1)
#     self.layer4 =  self.make_disc_block(1024, 64, final = True)
#     self.fc1 = nn.Linear(64,1)
#
#
#   def make_disc_block(self, input_channels, output_channels, stride = 2, kernel_size = 3, padding = 0, final = False):
#     if not final:
#       return nn.Sequential(
#           nn.Conv2d(input_channels, output_channels, stride = stride, kernel_size = kernel_size, padding = padding),
#           nn.BatchNorm2d(output_channels),
#           nn.LeakyReLU()
#       )
#     else:
#       return nn.Sequential(
#           nn.Conv2d(input_channels, output_channels, stride = stride, kernel_size = kernel_size, padding = padding),
#           nn.Sigmoid()
#       )
#   def forward(self, input):
#     x = self.layer1(input)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)
#     # print("x.shape after layer4:",x.shape)
#     x = torch.flatten(x, 1)
#     # print("x.shape after flatten:",x.shape)
#     x = self.fc1(x)
#     # print("x.shape after fc1:",x.shape)
#
#     return torch.sigmoid(x)


discriminator = Discriminator(28*28).cuda()

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
      plt.imshow(image_grid.permute(1,2,0))
      print(image[0].shape)
      break
    plt.show()

test_images()



def generate_noise_vector(n_images):
    z_noise = torch.randn(n_images,64,1,1).cuda()
    for i in range(n_images):
        # print("z_noise_shape", z_noise.shape)
        z_noise[i,:,:,:] = torch.randn(64,1,1).cuda()
    return z_noise

def test_generate_noise_vector():
    print()
    print("TEST_GENERATE_NOISE_VECTOR")

    z_noise = generate_noise_vector(128)
    print(z_noise.shape)
    plt.imshow(torch.squeeze(generator(z_noise)).cpu().detach().numpy()[0])
    plt.show()
    # print(generator(z_noise))

# test_generate_noise_vector()

def test_generator_discriminator():
    print("TEST GENERATOR DISCRIMINATOR")
    z_noise = torch.randn(1,64,1,1).cuda()
    output = generator.forward(z_noise)
    classification = discriminator(output)
    print(classification)

# test_generator_discriminator()

truth_criterion = nn.BCELoss()
# print("TRUTH_CRITERION_TEST", truth_criterion(torch.Tensor([0.9, 0.8,0.7]).float(),torch.tensor([0.5,0.2,0.4]).float()))
optimizer_discriminator = optim.Adam(discriminator.parameters())
optimizer_generator = optim.SGD(generator.parameters(), lr= 1)

def test_generator_discriminator_criterion():
    print()
    print("TEST GENERATOR DISCRIMINATOR CRITERION")
    z_noise = generate_noise_vector(128)
    fake_img = generator(z_noise)
    # logits_prediction = discriminator(output)[0]
    truth_prediction = discriminator(fake_img)
    print("truth_pred:", truth_prediction)
    # classification_loss = criterion(logits_prediction, torch.tensor([1]).cuda())
    truth_loss = truth_criterion(truth_prediction,torch.tensor([[1]]*128).float().cuda() )
    print("truth_loss", truth_loss)
# test_generator_discriminator_criterion()


def calculate_loss(truth_prediction, fake):
    if fake:
        truth_loss = truth_criterion(truth_prediction, torch.zeros_like(truth_prediction).float().cuda())
        return truth_loss

    else:
        truth_loss = truth_criterion(truth_prediction, torch.ones_like(truth_prediction).float().cuda())
        return truth_loss



#### TRAINING LOOP DISCRIMINATOR

def train_discriminator(n_batchs):
    # for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()

        # zero the parameter gradients
        optimizer_discriminator.zero_grad()

        # forward + backward + optimize
        truth_prediction = discriminator(inputs)
        loss = calculate_loss(truth_prediction, fake = False)

        loss.backward()
        optimizer_discriminator.step()

        # print statistics
        running_loss += loss.item()

        if i == n_batchs:
            break
    # print('discriminator loss:', running_loss / n_batchs)
    discriminator_loss = running_loss/n_batchs
    return discriminator_loss

# torch.save(discriminator.state_dict(), 'discriminator')
# discriminator.load_state_dict(torch.load('discriminator'))

def test_training_works():
    print()
    print("TEST_TRAINING_WORKS")
    discriminator_loss = train_discriminator(1)
    print("discriminator_loss", discriminator_loss)
# test_training_works()


def train_generator_and_disc(n_batchs):
    print()
    # print("TEST DISCRIMINATOR_GENERATOR")
    # optimizer = optim.Adam(list(generator.parameters()) ) ## + list(discriminator.parameters())

    running_loss = 0.0
    SIZE_BATCHES = 128
    for i in range(n_batchs):
        # print("labels.shape",labels.shape)
        # get the inputs; data is a list of [inputs, labels]
        batch_z_noise = generate_noise_vector(SIZE_BATCHES)
        batch_fake_image = generator(batch_z_noise)

        # zero the parameter gradients
        optimizer_generator.zero_grad()

        # forward + backward + optimize
        truth_prediction = discriminator(batch_fake_image)
        truth_prediction_disc = discriminator(batch_fake_image.detach())
        # print("torch.tensor(labels).cuda()", torch.tensor(labels).long().cuda())
        # print("logits:", logits.shape, "truth_pred:", truth_prediction.shape, "labels:", labels.shape)
        loss_gen = calculate_loss(truth_prediction, fake = False)
        loss_gen.backward(retain_graph=True)
        optimizer_generator.step()

        loss_disc = calculate_loss(truth_prediction_disc, fake = True)
        optimizer_discriminator.zero_grad()
        loss_disc.backward()
        optimizer_discriminator.step()


        # print statistics
        running_loss += loss_gen.item()

    # print("generator loss:", running_loss/n_batchs)
    generator_loss = running_loss/n_batchs
    return generator_loss

def train_generator_only(n_batchs):
    # print()
    # print("TRAIN GENERATOR ONLY")
    # optimizer = optim.Adam(list(generator.parameters()) ) ## + list(discriminator.parameters())

    running_loss = 0.0
    SIZE_BATCHES = 128
    for i in range(n_batchs):
        # print("labels.shape",labels.shape)
        # get the inputs; data is a list of [inputs, labels]
        optimizer_generator.zero_grad()
        batch_z_noise = generate_noise_vector(SIZE_BATCHES)
        # print("batch_z_noise:", batch_z_noise)
        batch_fake_image = generator(batch_z_noise)
        # print("batch_fake_image:", batch_fake_image)

        # zero the parameter gradients


        # forward + backward + optimize
        truth_prediction = discriminator(batch_fake_image)
        # print("truth_prediction.shape:", truth_prediction.shape)
        # print("truth_prediction:", truth_prediction)
        # print("torch.tensor(labels).cuda()", torch.tensor(labels).long().cuda())
        # print("logits:", logits.shape, "truth_pred:", truth_prediction.shape, "labels:", labels.shape)
        loss_gen = calculate_loss(truth_prediction, fake = False)
        loss_gen.backward()
        optimizer_generator.step()



        # print statistics
        running_loss += loss_gen.item()

    # print("generator loss:", running_loss/n_batchs)
    generator_loss = running_loss/n_batchs
    return generator_loss

def test_train_generator_only():
    print()
    print("TEST_TRAIN_GENERATOR")
    print("Generator loss: ",train_generator_only(2))

# test_train_generator_only()


def test_gan_fake_input():
    print()
    print("TEST_GAN_FAKE_INPUT")
    noise_vector = generate_noise_vector(1)
    image = generator(noise_vector)
    predictions = discriminator(image)
    print(predictions)
    plt.imshow(torch.squeeze(image).cpu().detach().numpy(), cmap = "gray")
    plt.show()
# test_gan_fake_input()

def test_gan_real():
    for i, data in enumerate(testloader, 0):
        images, label = data
        images = images.cuda()
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

import copy

def test_if_discriminator_changes_during_generator_training():
    disc_params_before = 0
    disc_params_after = 0
    for param in discriminator.parameters():
        disc_params_before = copy.deepcopy(param)
        # print(disc_params_before[0])
        break
    train_generator(30)
    for param in discriminator.parameters():
        disc_params_after = copy.deepcopy(param)
        # print(disc_params_after[0])
        break

    if torch.all(torch.eq(disc_params_before, disc_params_after)):
        print("discriminator did not change during generator training")
    else:
        print("discriminator did change during generator training" )

# test_if_discriminator_changes_during_generator_training()

def test_if_discriminator_changes_during_discriminator_training():
    disc_params_before = 0
    disc_params_after = 0
    for param in discriminator.parameters():
        disc_params_before = copy.deepcopy(param)
        # print(disc_params_before[0])
        break
    train_discriminator(30)
    for param in discriminator.parameters():
        disc_params_after = copy.deepcopy(param)
        # print(disc_params_after[0])
        break

    if torch.all(torch.eq(disc_params_before, disc_params_after)):
        print("discriminator did not change during discriminator training")
    else:
        print("discriminator did change during discriminator training" )
    # print(disc_params_before)
    # print(disc_params_after)

# test_if_discriminator_changes_during_discriminator_training()

if __name__ == '__main__':

    def full_train2(n_cycle, length_cycle_gen, length_cycle_disc):
        gen_loss = train_generator_only(1)
        disc_loss = 0
        for i in range(n_cycle):
            disc_loss = train_discriminator(length_cycle_disc)
            gen_loss = train_generator_only(length_cycle_gen)
            print("step",  i, "gen_loss:", gen_loss, "disc_loss:", disc_loss)
            if i % 50 == 49:
                print("save")
                torch.save(discriminator.state_dict(), 'weights/discriminator'+"_gen"+str(length_cycle_gen) + "_disc" + str(length_cycle_disc))
                torch.save(generator.state_dict(), 'generator_weights/generator' + "_gen"+str(length_cycle_gen) + "_disc" + str(length_cycle_disc))

    # discriminator.load_state_dict(torch.load('weights/discriminator2_gen20_disc2'))
    # generator.load_state_dict(torch.load('generator_weights/generator2_gen20_disc2'))

    # discriminator.load_state_dict(torch.load('weights/discriminator'))
    # generator.load_state_dict(torch.load('generator_weights/generator'))
    #

    LENGTH_CYCLE_GEN = 20
    LENGTH_CYCLE_DISC = 20

    def load_models():
        try:
            discriminator.load_state_dict(torch.load('weights/discriminator_gen'+ str(LENGTH_CYCLE_GEN) + '_disc' + str(LENGTH_CYCLE_DISC)))
            generator.load_state_dict(torch.load('generator_weights/generator_gen'+ str(LENGTH_CYCLE_GEN) + '_disc' + str(LENGTH_CYCLE_DISC)))
            print("loaded models")
        except:
            print("couldn't load models, training from scratch")
    # load_models()

    full_train2(50, length_cycle_gen = LENGTH_CYCLE_GEN, length_cycle_disc= LENGTH_CYCLE_DISC)
    # train_disc_only(20)
    # train_gen_only(20)

    def test_gan_generation():

        for i in range(10):
            test_gan_fake_input()

    # test_gan_generation()
    #
    #
    # torch.save(discriminator.state_dict(), 'discriminator2')
    # torch.save(generator.state_dict(), 'generator2')
