import torch
import torch.nn as nn
import torch.fft
from model.conv_bn_relu import ConvBNRelu
from options import HiDDenConfiguration

class Encoder(nn.Module):
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]
        for _ in range(config.encoder_blocks - 1):
            layers.append(ConvBNRelu(self.conv_channels, self.conv_channels))

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length, self.conv_channels)
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

        # Fourier space mask configuration
        self.watermark_radius = config.watermark_radius

    def apply_tree_ring_watermark(self, image_fft, message):

        # low_band, mid_band, high_band = self.split_frequency_bands(image_fft)
        # Check and print shapes for diagnostic purposes
        batch_size, num_channels, height, width = image_fft.shape
        #image_fft = low_band
        # print(f"image_fft shape: {image_fft.shape}")
        # print(f"message shape: {message.shape}")

        if message.dim() == 1:
            message = message.unsqueeze(0)
        
        message_length = message.shape[1]
        center = (height // 2, width // 2)
        
        # Embed the message in each channel separately
        for batch_idx in range(batch_size):
            index = 0  # Reset the message index for each batch item
            for channel in range(num_channels):
                for i in range(-self.watermark_radius, self.watermark_radius + 1):
                    for j in range(-self.watermark_radius, self.watermark_radius + 1):
                        if index >= message_length:
                            break
                        if not (0 <= center[0] + i < height and 0 <= center[1] + j < width):
                            continue  # Skip if out of bounds
                        distance = (i ** 2 + j ** 2) ** 0.5
                        if distance <= self.watermark_radius:
                            value = float(message[batch_idx, index])
                            # Embed into the Fourier space at the center position
                            image_fft[batch_idx, channel, center[0] + i, center[1] + j] = complex(value, value)
                            index += 1
        return image_fft

        def split_frequency_bands(self, frequency_image):
            # Split the frequency image into low, mid, and high-frequency bands
            h, w = frequency_image.shape[-2:]
            center_h, center_w = h // 2, w // 2

            # Define band ranges
            low_radius = min(h, w) // 4
            high_radius = min(h, w) // 2

            # Create masks for different frequency bands
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            distance = ((y - center_h) ** 2 + (x - center_w) ** 2).sqrt()

            low_mask = distance <= low_radius
            high_mask = distance > high_radius
            mid_mask = (distance > low_radius) & (distance <= high_radius)

            # Apply masks to get different frequency bands
            low_band = frequency_image * low_mask
            mid_band = frequency_image * mid_mask
            high_band = frequency_image * high_mask

            return low_band, mid_band, high_band


    def forward(self, image, message):
        
        #  # Encode watermark in each band
        # low_encoded = self.low_band_encoder(low_band)
        # mid_encoded = self.mid_band_encoder(mid_band)
        # high_encoded = self.high_band_encoder(high_band)

        encoded_image = self.conv_layers(image)
        #encoded_image = self.conv_layers(low_encoded)
        
        # Transform to Fourier space and embed the message as a watermark
        image_fft = torch.fft.fft2(encoded_image)
        watermarked_fft = self.apply_tree_ring_watermark(image_fft, message)
        encoded_image = torch.fft.ifft2(watermarked_fft).real  # back to spatial domain

        # Continue with remaining encoding steps
        expanded_message = message.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.H, self.W)
        concat = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w
