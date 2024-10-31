import torch
import torch.nn as nn
import numpy as np
from options import HiDDenConfiguration
from model.encoder_decoder import EncoderDecoder
from model.discriminator import Discriminator
from noise_layers.noiser import Noiser

class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger):
        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.AdamW(self.encoder_decoder.parameters())
        self.optimizer_discrim = torch.optim.AdamW(self.discriminator.parameters())

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

    def train_on_batch(self, batch: list):
        images, messages = batch
        batch_size = images.shape[0]

        images = images.to(self.device)
        messages = messages.to(self.device)

        self.encoder_decoder.train()
        self.discriminator.train()
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device, dtype=torch.float)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            d_loss_on_cover.backward()

            # train on fake
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
            decoded_messages = decoded_messages.to(self.device)
            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            g_loss_enc = self.mse_loss(encoded_images, images)

            # print(f"Device of images: {images.device}")
            # print(f"Device of messages: {messages.device}")
            # print(f"Device of encoded_images: {encoded_images.device}")
            # print(f"Device of decoded_messages: {decoded_messages.device}")
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec

            g_loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # # if TensorboardX logging is enabled, save some of the tensors.
        # if self.tb_logger is not None:
        #     encoder_final = self.encoder_decoder.encoder._modules['final_layer']
        #     self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
        #     decoder_final = self.encoder_decoder.decoder._modules['linear']
        #     self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
        #     discrim_final = self.discriminator._modules['linear']
        #     self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        images, messages = batch

        batch_size = images.shape[0]

        images = images.to(self.device)
        messages = messages.to(self.device)

        self.encoder_decoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device, dtype=torch.float)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)

            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            d_on_encoded = self.discriminator(encoded_images)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)

            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)

            
            g_loss_enc = self.mse_loss(encoded_images, images)

            decoded_messages = decoded_messages.to(self.device)
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))