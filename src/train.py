<<<<<<< Updated upstream
"""

"""
import os
import time

import torch
import src.networks as networks
import torchvision.transforms as transforms

from src.videodata import VideoDataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader


class Train:

    def __init__(self, args):

        self.num_ds = args.num_ds
        self.lambda_l1 = args.lambda_l1
        self.batch_size = args.batch_size
        self.num_blocks = args.num_blocks
        self.block_type = args.block_type

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        if args.block_type == 'sisr':
            resblocks = networks.SISR_Resblocks(args.num_blocks)

        self.net_g = networks.Generator(resblocks).to(self.device)
        self.l1_loss = torch.nn.L1Loss()

        self.optimizer = torch.optim.Adam(
            self.net_g.parameters(), lr=args.lr_g)

        self.step = 0
        self.resize_func = transforms.Resize(args.patch_size)
        self.resize2_func = transforms.Resize((1080, 1440))

        train_dir = os.path.join(args.data_dir, 'train')
        test_dir = os.path.join(args.data_dir, 'test')

        self.train_dataset = VideoDataset(
            root_dir=train_dir,
            train=True,
            cache_size=args.cache_size,
            patch_size=args.patch_size,
            num_ds=args.num_ds)
        self.train_data_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True)

        self.test_dataset = VideoDataset(
            root_dir=test_dir,
            train=False,
            cache_size=args.cache_size,
            patch_size=args.patch_size,
            num_ds=args.num_ds)
        self.test_data_loader = DataLoader(
                self.test_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=False)

        self.static_test_batch = []
        for batch_y in self.test_data_loader:
            self.static_test_batch.append(batch_y.to(self.device))
            break

    def step_g(self, batch_x, batch_y):

        self.optimizer.zero_grad()

        batch_g = self.net_g(batch_x)
        loss = self.lambda_l1 * self.l1_loss(batch_g, batch_y)

        loss.backward()

        self.optimizer.step()

        return loss, batch_g

    def train(self):

        num_epochs = self.train_dataset.get_epochs_per_dataset()
        print("{} epochs to loop over entire dataset once".format(num_epochs))

        # How many times we're going to loop over our entire dataset
        for loop in range(100):

            for epoch in range(num_epochs):

                for batch_data in self.train_data_loader:
                    batch_x, batch_y = batch_data

                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # shift our data's range from [0, 1] to [-1, 1]
                    batch_x = (batch_x * 2.) - 1.
                    batch_y = (batch_y * 2.) - 1.

                    s = time.time()
                    loss, batch_g = self.step_g(batch_x, batch_y)
                    loss = round(loss.cpu().item(), 3)
                    self.step += 1

                    #print('| epoch:', epoch, '/', num_epochs, '| step:', self.step, '| loss:', loss, '| time:', round(time.time()-s, 2))
                    print("| itr:{} | epoch:{}/{} | step:{} | loss:{} | time:{} |".
                          format(loop, epoch, num_epochs, self.step, loss, round(time.time()-s, 2)))
                    # for debugging purposes, cap our epochs to whatever number of steps
                    #if not self.step % 50:
                    #    break

                # Swap memory cache and train on the new stuff
                self.train_dataset.swap()

                i = 0
                print('Saving out test images')
                for test_batch_y in self.static_test_batch:
                    _, _, yh, yw = test_batch_y.shape
                    x_size = (yh // self.num_ds, yw // self.num_ds)
                    resize_func = transforms.Resize(x_size)
                    test_batch_x = resize_func(test_batch_y)

                    test_batch_g = self.net_g(test_batch_x)

                    for bx, by, bg in zip(test_batch_x, test_batch_y, test_batch_g):
                        bx = (bx + 1.) / 2.
                        by = (by + 1.) / 2.
                        bg = (bg + 1.) / 2.
                        bx = self.resize2_func(bx)
                        canvas = torch.cat([bx, by, bg], axis=1)
                        save_image(
                            canvas, os.path.join('test', str(self.step).zfill(3)+'_'+str(i)+'.png'))
                        break
                    i += 1
                    break

            # make sure our superlist is indeed empty
            assert self.dataset.data_decoder.get_num_chunks() == 0, "Still have some chunks left unloaded"
            # Rebuild our superlist and send er again
            self.dataset.data_decoder.build_chunk_superlist()
=======
"""

"""
import gc
import os
import copy
import time

import torch
import src.networks as networks
import torchvision.transforms as transforms

from src.videodata import VideoDataset
from torchvision.utils import save_image
from torch.utils.data import DataLoader


class Train:

    def build_gan_loss(self):

        # try this
        # Discriminator loss
        #errD = BCE_stable(y_pred - y_pred_fake, y)
        #errD.backward()

        # Generator loss (You may want to resample again from real and fake data)
        #errG = BCE_stable(y_pred_fake - y_pred, y)
        #errG.backward()
        # end try this

        criterion = torch.nn.BCEWithLogitsLoss()

        #def loss_g(d_fake):
        def loss_g(d_fake, d_real):
            y_real = torch.ones(d_real.shape).to(self.device)
            #loss = criterion(d_fake, y_pred_fake)
            loss = criterion(d_fake - d_real, y_real)
            return loss

        # takes discriminated fake and real
        def loss_d(d_fake, d_real):
            y_real = torch.ones(d_real.shape).to(self.device)

            #loss_real = criterion(d_fake, y_real)
            #loss_fake = criterion(d_fake, y_fake)
            loss = criterion(d_real - d_fake, y_real)

            #return loss_real + loss_fake
            return loss

        return loss_g, loss_d

    def __init__(self, args):

        self.num_ds = args.num_ds
        self.lambda_l1 = args.lambda_l1
        self.lambda_gan = args.lambda_gan # for now, take an arg later
        self.lambda_feature = args.lambda_feature
        self.batch_size = args.batch_size
        self.num_blocks = args.num_blocks
        self.block_type = args.block_type
        self.model_dir = args.model_dir
        self.test_training = args.test
        self.patch_ds_factor = args.patch_ds_factor
        self.cache_size = args.cache_size
        self.patch_size = args.patch_size
        self.num_ds = args.num_ds

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        if args.block_type == 'sisr':
            resblocks = networks.SISR_Resblocks(args.num_blocks)
        elif args.block_type == 'rrdb':
            resblocks = networks.RRDB_Resblocks(args.num_blocks)

        # define the generator and descriminiator
        self.net_g = networks.Generator(resblocks).to(self.device)
        self.net_d = networks.VGGStyleDiscriminator128().to(self.device)
        self.l1_loss = torch.nn.L1Loss()

        # feature extractor
        self.net_f = networks.VGG19FeatureNet().to(self.device)

        self.step = 0
        self.optimizer_g = torch.optim.Adam(
            self.net_g.parameters(), lr=args.lr_g)

        self.optimizer_d = torch.optim.Adam(
            self.net_d.parameters(), lr=1e-3)

        # If we're resuming a training session, this is the place to do it
        if args.resume_training:
            self.load()

        self.loss_g, self.loss_d = self.build_gan_loss()

        #self.resize_func = transforms.Resize(args.patch_size)
        #self.patch_downscale_func = transforms.Resize((1080, 1440))

        self.train_dir = os.path.join(args.data_dir, 'train')
        self.test_dir = os.path.join(args.data_dir, 'test')

        self.train_dataset = VideoDataset(
            root_dir=self.train_dir,
            train=True,
            cache_size=self.cache_size,
            patch_size=self.patch_size,
            num_ds=self.num_ds,
            patch_ds_factor=self.patch_ds_factor)
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True)

        if self.test_training:
            self.static_test_batch = []
            self.get_test_frame(self.patch_size, self.num_ds)

    def load(self):
        checkpoint = torch.load(os.path.join(self.model_dir, 'model.pth'))
        self.net_g.load_state_dict(checkpoint['gen_model_state'])
        self.optimizer_g.load_state_dict(checkpoint['gen_optim_state'])
        self.net_d.load_state_dict(checkpoint['disc_model_state'])
        self.optimizer_d.load_state_dict(checkpoint['disc_optim_state'])
        self.step = checkpoint['step_num']
        print("Loaded saved model at step {} and set model/optim states".format(checkpoint['step_num']))

    def save(self):
        print("trying to save model...")

        print("step num is", self.step)
        gen_model_state_dict = self.net_g.state_dict()
        gen_optim_state_dict = self.optimizer_g.state_dict()
        disc_model_state_dict = self.net_d.state_dict()
        disc_optim_state_dict = self.optimizer_d.state_dict()
        torch.save({
            'step_num': self.step,
            'gen_model_state': gen_model_state_dict,
            'gen_optim_state': gen_optim_state_dict,
            'disc_model_state': disc_model_state_dict,
            'disc_optim_state': disc_optim_state_dict
        }, os.path.join(self.model_dir, 'model.pth'))

    # initialize our dataloader here because it caches it's cache_size in ram,
    # and we don't want that hanging around
    def get_test_frame(self, patch_size, num_ds):
        self.test_dataset = VideoDataset(
            root_dir=self.test_dir,
            train=False,
            cache_size=0.5,
            patch_size=patch_size,
            num_ds=num_ds)
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False)

        # Get 1 frame
        for batch_y in self.test_data_loader:
            #self.static_test_batch.append(batch_y.to(self.device))
            self.static_test_batch.append(batch_y)
            break


    # This is in a separate function so we can create (and destroy) our net on the cpu so
    # that we don't blast video ram with a full image convolution
    def test(self):
        i = 0
        print('Saving out test images')
        # Copy the net over to our cpu so we don't blast our vram
        cpunet_g = copy.deepcopy(self.net_g).to('cpu')
        print("Duplicated model on cpu...")
        for test_batch_y in self.static_test_batch:
            _, _, yh, yw = test_batch_y.shape
            x_size = (yh // self.num_ds, yw // self.num_ds)
            resize_func = transforms.Resize(x_size)
            test_batch_x = resize_func(test_batch_y)

            #test_batch_g = self.net_g(test_batch_x)
            test_batch_g = cpunet_g(test_batch_x)

            for bx, by, bg in zip(test_batch_x, test_batch_y, test_batch_g):
                #bx = (bx + 1.) / 2.
                #by = (by + 1.) / 2.
                #bg = (bg + 1.) / 2.
                bx = self.resize2_func(bx)
                canvas = torch.cat([bx, by, bg], axis=1)
                save_image(
                    canvas, os.path.join('test', str(self.step).zfill(3)+'_'+str(i)+'.png'))
                break
            i += 1
            break

    def checkpoint(self, input, gt, output):
        dims = gt.size()[3]

        input = transforms.Resize(dims)(input)
        # Bring this back to 0 to 1 from -1 to 1
        input = (input + 1.) / 2.
        gt = (gt + 1.) / 2.
        output = (output + 1.) / 2.
        canvas = torch.cat([input[:1], gt[:1], output[:1]], axis=3)
        save_image(canvas, os.path.join('test', str(self.step).zfill(3)+'.png'))

    # batch x is the small image, batch_y is the original size
    def step_g(self, batch_x, batch_y, discriminator=True):

        self.optimizer_g.zero_grad()

        # run our downsized batch_y through generator
        batch_g = self.net_g(batch_x)

        # this is the pixel-wise loss
        l1_loss = self.lambda_l1 * self.l1_loss(batch_g, batch_y)

        total_loss = l1_loss

        # add the loss from the feature net
        total_loss += self.lambda_feature * self.l1_loss(self.net_f(batch_g), self.net_f(batch_y))

        # calculate loss
        if discriminator:
            # run our generated samples through our discriminator (not racist)
            d_fake = self.net_d(batch_g)
            d_real = self.net_d(batch_y)
            g_loss = self.lambda_gan * self.loss_g(d_fake, d_real)

            total_loss += g_loss

        total_loss.backward()

        self.optimizer_g.step()

        return total_loss, batch_g.detach() # detaches from the autograd in step_g so the tensor can be worked on

    # take our truth, and the downsized data we'll use to resemple net_g
    def step_d(self, batch_truth, batch_x):

        self.optimizer_d.zero_grad()

        # resample our lie from the generator
        batch_lie = self.net_g(batch_x)

        # run our real and fake data through the descriminator
        d_real = self.net_d(batch_truth)
        d_fake = self.net_d(batch_lie)
        # calculate the loss of the real images, just using dumb l1 for now to get a psnr-oriented model
        #loss = self.lambda_l1 * self.l1_loss(descriminiator_output, all_data_labels)
        loss = self.lambda_gan * self.loss_d(d_fake, d_real)

        loss.backward()

        self.optimizer_d.step()

        return loss

    def train(self):

        num_epochs = self.train_dataset.get_epochs_per_dataset()
        print("{} epochs to loop over entire dataset once".format(num_epochs))

        last_d_value = 0

        print("here we go aGAN ;)")
        # How many times we're going to loop over our entire dataset
        for loop in range(100):

            for epoch in range(num_epochs):

                for batch_data in self.train_data_loader:
                    batch_x, batch_y = batch_data


                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    # shift our data's range from [0, 1] to [-1, 1] (should really move this to dataloader)
                    batch_x = (batch_x * 2.) - 1.
                    batch_y = (batch_y * 2.) - 1.

                    # start our batch timer
                    s = time.time()
                    # Train our generator with just l1 for a while
                    #if self.step < 1000:
                    #    loss_g, batch_g = self.step_g(batch_x, batch_y, discriminator=False)
                    #    #loss_d = self.step_d(batch_truth=batch_y, batch_lie=batch_g)
                    #    loss_d = loss_g
                    #else:
                    #    if self.step == 1000:
                    #        # reinitilize our optimizer so it doesnt' blast
                    #        print("RESTARTING OPTIMIZER")
                    #        self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=1e-4)
                    #    loss_g, batch_g = self.step_g(batch_x, batch_y, discriminator=True)
                    #    loss_d = self.step_d(batch_truth=batch_y, batch_lie=batch_g)

                    loss_g, batch_g = self.step_g(batch_x, batch_y, discriminator=True)
                    # don't use ^ batch_g for now, we'll re-sample in the discriminator step
                    #loss_d = self.step_d(batch_truth=batch_y, batch_lie=batch_g)
                    loss_d = self.step_d(batch_truth=batch_y, batch_x=batch_x)
                    last_d_value = loss_d

                    self.step += 1

                    # round our losses
                    loss_d = round(loss_d.cpu().item(), 3)
                    loss_g = round(loss_g.cpu().item(), 3)
                    # print our update
                    print("| itr:{} | epoch:{}/{} | step:{} | d_loss:{} | g_loss:{} | time:{} |".
                          format(loop, epoch, num_epochs, self.step, loss_d, loss_g, round(time.time()-s, 2)))
                    # for debugging purposes, do whatever X amount of steps
                    if not self.step % 100:
                        print("checkpoint!")
                        self.checkpoint(batch_x, batch_y, batch_g)
                    #    break

                if (last_d_value <= .100) & (self.step >= 1000):
                    print("Something's wrong... it looks like our gan blew up. Exiting")
                    exit(1)
                else:
                    # Swap memory cache and train on the new stuff
                    self.train_dataset.swap()

                    # test our model
                    if self.test_training:
                        self.test()

                    # Save our model
                    self.save()

            # make sure our superlist is indeed empty
            print("Num chunks left:", self.train_dataset.data_decoder.get_num_chunks())
            #assert self.train_dataset.data_decoder.get_num_chunks() == 0, "Still have some chunks left unloaded"
            # Rebuild our superlist and send er again
            self.train_dataset.data_decoder.build_chunk_superlist()
>>>>>>> Stashed changes
