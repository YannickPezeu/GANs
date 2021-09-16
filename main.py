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

class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc0 = nn.Linear(640, 64)
    self.block1 = self.make_gen_block(64, 1024)
    self.block2 = self.make_gen_block(1024, 512, stride=1)
    self.block3 = self.make_gen_block(512, 256,kernel_size=3)
    self.block4 = self.make_gen_block(256, 128,kernel_size=3, stride = 1)
    self.block5 = self.make_gen_block(128, 1, kernel_size =4, padding = 2, final = True)


  def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final = False, padding=0):
    if not final:
      return nn.Sequential(
          nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride = stride, padding = padding),
          nn.BatchNorm2d(output_channels),
          nn.LeakyReLU(),
      )
    else:
      return nn.Sequential(
          nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride = stride),
          nn.Tanh(),
      )

  def forward(self, input):
    # print("input generator", input.shape)
    x = torch.flatten(input, 1)
    # print(x.shape)
    x = self.fc0(x)
    # print("shape after fc0",x.shape)
    x = x[...,None, None]
    # print(x.shape)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    # print("final shape", x.shape)
    return x

def test_generator():
    print()
    print("TEST GENERATOR")
    z_noise = torch.randn(10,64,1,1).cuda()
    z_noise[0,:,:,:] = torch.randn(1,64,1,1).cuda()
    print("z_noise", z_noise)
    # z_noise = torch.randn(1,64,1,1).cuda()
    output = generator.forward(z_noise).cuda()
    print(output.shape)
    plt.imshow(torch.squeeze(output).cpu().detach().numpy())
    plt.show()

# test_generator()

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 =  self.make_disc_block(1, 256)
    self.layer2 =  self.make_disc_block(256, 512)
    self.layer3 =  self.make_disc_block(512, 1024, stride=1)
    self.layer4 =  self.make_disc_block(1024, 64, final = True)
    self.fc1 = nn.Linear(64,256)
    self.fc2 = nn.Linear(256,10)
    self.fc3 = nn.Linear(256, 1)

  def make_disc_block(self, input_channels, output_channels, stride = 2, kernel_size = 3, padding = 0, final = False):
    if not final:
      return nn.Sequential(
          nn.Conv2d(input_channels, output_channels, stride = stride, kernel_size = kernel_size, padding = padding),
          nn.BatchNorm2d(output_channels),
          nn.LeakyReLU()
      )
    else:
      return nn.Sequential(
          nn.Conv2d(input_channels, output_channels, stride = stride, kernel_size = kernel_size, padding = padding),
          nn.Sigmoid()
      )
  def forward(self, input):
    x = self.layer1(input)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    # print("x.shape after layer4:",x.shape)
    x = torch.flatten(x, 1)
    # print("x.shape after flatten:",x.shape)
    x = self.fc1(x)
    # print("x.shape after fc1:",x.shape)
    digit_prediction = self.fc2(x)
    truth_prediction = self.fc3(x)
    truth_prediction = torch.sigmoid(truth_prediction)
    return digit_prediction, torch.squeeze(truth_prediction)

generator = Generator().cuda()
discriminator = Discriminator().cuda()

def test_discriminator():
    print()
    print("TEST DISCRIMINATOR")
    test_layer = nn.Conv2d(1, 256, stride = 2, kernel_size = 3, padding = 0)
    for image, label in trainloader:
      image = image.cuda()
      print("entry image shape:",image[0].shape)
      x = discriminator(image[0][None,...])
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

# test_images()



def generate_noise_vector(labels):
    z_noise = torch.randn(len(labels),64,10,1).cuda()
    for i,label in enumerate(labels):
        z_noise[i,:,label,:] = torch.randn(1,64,1).cuda()
    return z_noise

def test_generate_noise_vector():
    print()
    print("test_generate_noise_vector()")
    labels = np.random.randint(10, size = 10)
    z_noise = generate_noise_vector(labels)
    print(z_noise.shape)
    print(generator(z_noise))

# test_generate_noise_vector()

def test_generator_discriminator():
    print("TEST GENERATOR DISCRIMINATOR")
    z_noise = torch.randn(1,64,10,1).cuda()
    output = generator.forward(z_noise)
    classification = discriminator(output)
    print(classification)

# test_generator_discriminator()

criterion = nn.CrossEntropyLoss()
truth_criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters())
optimizer_generator = optim.Adam(list(generator.parameters()))

def test_generator_discriminator_criterion():
    z_noise = torch.randn(1,64,10,1).cuda()
    output = generator(z_noise)
    logits_prediction = discriminator(output)[0]
    truth_prediction = discriminator(output)[1]

    classification_loss = criterion(logits_prediction, torch.tensor([1]).cuda())
    truth_loss = truth_criterion(truth_prediction,torch.tensor(1).float().cuda() )
    print("classification_loss",classification_loss)
    print("truth_loss",truth_loss)
    full_loss = classification_loss + 5* truth_loss
    print("full_loss", full_loss)
# test_generator_discriminator_criterion()


def calculate_full_lossD(class_prediction, truth_prediction, labels, truth_label, mode):
    if mode == "disc_only":
        classification_loss = criterion(class_prediction, labels)
        real_truth_loss = truth_criterion(truth_prediction, torch.ones_like(truth_prediction).float().cuda())
        full_loss = classification_loss + real_truth_loss * TRUTH_WEIGHT
        return full_loss
    elif mode == "disc_and_gen":
        truth_loss = truth_criterion(truth_prediction, torch.tensor(truth_label).float().cuda())
        return truth_loss * TRUTH_WEIGHT
    else:
        raise ValueError("mode unknown in calculate_full_lossD. Choose disc_only or disc_and_gen")

def calculate_full_loss_gen(logits, truth_prediction, labels, truth_label, ):
    classification_loss = criterion(logits, labels)
    truth_loss = truth_criterion(truth_prediction, torch.tensor(truth_label).float().cuda())
    full_loss = classification_loss - truth_loss * TRUTH_WEIGHT
    return full_loss

#### TRAINING LOOP DISCRIMINATOR

def train_discriminator(n_batchs):
    # for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        real_img, labels = data
        real_img = real_img.cuda()
        labels = labels.cuda()
        z_noise = generate_noise_vector(labels)
        fake_img = generator(z_noise)
        
        # Calculate Loss
        real_class_prediction, real_truth_prediction = discriminator(real_img)
        fake_class_prediction, fake_truth_prediction = discriminator(fake_img)

        class_prediction_loss = criterion(real_class_prediction, labels)
        real_truth_prediction_loss = truth_criterion(real_truth_prediction, torch.ones_like(real_truth_prediction))
        fake_truth_prediction_loss = truth_criterion(fake_truth_prediction, torch.ones_like(fake_truth_prediction))

        full_lossD = class_prediction_loss + real_truth_prediction_loss + fake_truth_prediction_loss
        
        # zero the parameter gradients
        optimizerD.zero_grad()
        full_lossD.backward()
        optimizerD.step()

        # print statistics
        running_loss += full_lossD.item()


        if i == n_batchs:
            break
    # print('discriminator loss:', running_loss / n_batchs)
    discriminator_loss =running_loss/n_batchs
    return discriminator_loss

# torch.save(discriminator.state_dict(), 'discriminator')
# discriminator.load_state_dict(torch.load('discriminator'))

def test_training_works():
    print()
    print("TEST_TRAINING_WORKS")
    discriminator_loss = train_discriminator(1)
    print("discriminator_loss", discriminator_loss)
# test_training_works()


def train_generator(n_batchs):
    print()
    # print("TEST DISCRIMINATOR_GENERATOR")
    # optimizer = optim.Adam(list(generator.parameters()) ) ## + list(discriminator.parameters())

    running_loss = 0.0
    SIZE_BATCHES = 128
    for i in range(n_batchs):
        labels = np.random.randint(10, size=SIZE_BATCHES)
        labels = torch.Tensor(labels).long().cuda()
        # get the inputs; data is a list of [inputs, labels]
        batch_z_noise = generate_noise_vector(labels)
        batch_fake_image = generator(batch_z_noise)

        # zero the parameter gradients
        optimizer_generator.zero_grad()

        # forward + backward + optimize
        class_prediction, truth_prediction = discriminator(batch_fake_image)
        # print("torch.tensor(labels).cuda()", torch.tensor(labels).long().cuda())
        # print("logits:", logits.shape, "truth_pred:", truth_prediction.shape, "labels:", labels.shape)

        truth_lossG = truth_criterion(truth_prediction, torch.ones_like(truth_prediction))
        class_lossG = criterion(class_prediction, labels)
        full_loss_gen = 10 * truth_lossG + class_lossG
        full_loss_gen.backward(retain_graph=True)
        optimizer_generator.step()

        # print statistics
        running_loss += full_loss_gen.item()

    # print("generator loss:", running_loss/n_batchs)
    generator_loss = running_loss/n_batchs
    return generator_loss

def test_train_generator():
    print("TEST_TRAIN_GENERATOR")
    train_generator(2)

# test_train_generator()


def test_gan_fake_input(n):
    print("TEST_GAN_FAKE_INPUT")
    noise_vector = generate_noise_vector([n])
    image = generator(noise_vector)
    predictions = discriminator(image)
    print(predictions)
    plt.imshow(torch.squeeze(image).cpu().detach().numpy())
    plt.show()

# test_gan_fake_input(1)

def test_gan_real():
    for i, data in enumerate(testloader, 0):
        images, label = data
        images = images.cuda()
        predictions = discriminator(images)
        print("prediction.shape", predictions.shape)
        print("comparison",torch.argmax(predictions, dim = 1), label)
        print(np.sum(torch.argmax(predictions, dim = 1).cpu().detach().numpy()== label.detach().numpy()))
        # plt.imshow(torch.squeeze(image).cpu().detach().numpy())
        # plt.show()
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

    def full_train_basic(n_cycle, length_cycle):

        gen_loss = train_generator(1)
        disc_loss = train_discriminator(1)
        for i in range(n_cycle):
            disc_loss = train_discriminator(length_cycle)
            gen_loss = train_generator(length_cycle)
            print("gen_loss:", gen_loss, "disc_loss:", disc_loss)
            if i % 50 == 49:
                torch.save(discriminator.state_dict(), 'weights/discriminator_basic')
                torch.save(generator.state_dict(), 'generator_weights/generator_basic')

    def train_disc_only(n_cycle):
        disc_loss = train_discriminator(1)
        for i in range(n_cycle):
            disc_loss = train_discriminator(10)
            print("disc_loss:", disc_loss)
            if i % 50 == 49:
                print("save")
                torch.save(discriminator.state_dict(), 'discriminator_only/discriminator_only')

    def train_gen_only(n_cycle):
        gen_loss = train_generator(1)
        for i in range(n_cycle):
            gen_loss = train_generator(10)
            print("gen_loss:", gen_loss)
            if i % 50 == 49:
                print("save")
                torch.save(discriminator.state_dict(), 'generator_only/generator_only')


    def generate_fake_image_grid(path):
            z_noise = generate_noise_vector([0]*25)
            fake_images = generator(z_noise).reshape(-1,1, 28,28).cpu().detach()
            image_grid = make_grid(fake_images[0:25], nrow = 5,)
            # print(image_grid.shape, "image_grid.shape")
            plt.imshow(image_grid.permute(1,2,0))
            plt.savefig(path)



    def full_train2(n_cycle, length_cycle_gen, length_cycle_disc):

        gen_loss = train_generator(1)
        disc_loss = train_discriminator(1)
        for i in range(n_cycle):
            disc_loss = train_discriminator(length_cycle_disc)
            # gen_loss = train_generator(length_cycle_gen)
            print("gen_loss:", gen_loss, "disc_loss:", disc_loss)
            if i % 50 == 49:
                print("save")
                generate_fake_image_grid("main/image_grid")
                torch.save(discriminator.state_dict(), 'weights/discriminator'+"_gen"+str(length_cycle_gen) + "_disc" + str(length_cycle_disc))
                torch.save(generator.state_dict(), 'weights/generator' + "_gen"+str(length_cycle_gen) + "_disc" + str(length_cycle_disc))



    # discriminator.load_state_dict(torch.load('weights/discriminator2_gen20_disc2'))
    # generator.load_state_dict(torch.load('generator_weights/generator2_gen20_disc2'))

    # discriminator.load_state_dict(torch.load('weights/discriminator'))
    # generator.load_state_dict(torch.load('generator_weights/generator'))
    #

    LENGTH_CYCLE_GEN = 20
    LENGTH_CYCLE_DISC = 1

    try:

        discriminator.load_state_dict(torch.load('weights/discriminator'+"_gen"+str(LENGTH_CYCLE_GEN) + "_disc" + str(LENGTH_CYCLE_DISC)))
        generator.load_state_dict(torch.load('weights/generator'+"_gen"+str(LENGTH_CYCLE_GEN) + "_disc" + str(LENGTH_CYCLE_DISC)))
        print("loaded models")
    except:
        print("couldn't load models, training from scratch")

    full_train2(100, length_cycle_gen = LENGTH_CYCLE_GEN, length_cycle_disc= LENGTH_CYCLE_DISC)
    # train_disc_only(20)
    # train_gen_only(20)

    def test_gan_generation():

        for i in range(10):
            test_gan_fake_input(i)

    test_gan_generation()
    #
    #
    # torch.save(discriminator.state_dict(), 'discriminator2')
    # torch.save(generator.state_dict(), 'generator2')
