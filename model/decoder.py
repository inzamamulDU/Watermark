import torch
import torch.nn as nn
import torch.fft
from model.conv_bn_relu import ConvBNRelu
from options import HiDDenConfiguration

class Decoder(nn.Module):
    """
    Decoder module. Extracts the message embedded in the Fourier space of the image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Decoder, self).__init__()
        self.channels = config.decoder_channels
        self.message_length = config.message_length
        self.watermark_radius = config.watermark_radius  # Set watermark radius from config

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))
        
        layers.append(ConvBNRelu(self.channels, self.message_length))
        
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

    def extract_message_from_fourier(self, image_fft):
        # Extract message bits from the Fourier space by iterating over channels
        batch_size, num_channels, height, width = image_fft.shape
        center = (height // 2, width // 2)
        message = []

        for batch_idx in range(batch_size):
            message_bits = []
            for channel in range(num_channels):
                for i in range(-self.watermark_radius, self.watermark_radius + 1):
                    for j in range(-self.watermark_radius, self.watermark_radius + 1):
                        if len(message_bits) >= self.message_length:
                            break
                        if not (0 <= center[0] + i < height and 0 <= center[1] + j < width):
                            continue  # Skip if out of bounds
                        distance = (i ** 2 + j ** 2) ** 0.5
                        if distance <= self.watermark_radius:
                            value = image_fft[batch_idx, channel, center[0] + i, center[1] + j].real
                            message_bits.append(value)
            message.append(torch.tensor(message_bits).unsqueeze(0))
        return torch.cat(message, dim=0)


    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        
        # Convert to Fourier space to extract embedded message
        image_fft = torch.fft.fft2(x)
        extracted_message = self.extract_message_from_fourier(image_fft)
        return extracted_message
