import torch
import cv2
import numpy as np
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

    def low_pass_filter(self, image_batch_tensor, radius=50):
        """
        Applies a low-pass filter to a batch of RGB images by keeping only low-frequency components.
        
        Parameters:
        image_batch_tensor (torch.Tensor): The input batch of RGB images as a PyTorch tensor of shape (batch_size, 3, H, W) on CUDA.
        radius (int): The radius for the low-pass filter in the frequency domain.

        Returns:
        low_freq_batch_tensor (torch.Tensor): The batch of RGB images containing only low-frequency components, as a PyTorch tensor on CUDA.
        """
        # List to store the low-frequency processed images
        low_freq_batch = []

        # Process each image in the batch independently
        for image_tensor in image_batch_tensor:
            # Convert individual image tensor from CUDA to CPU and NumPy array for OpenCV processing
            image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()  # Convert to H x W x C (for RGB)

            # Split into channels
            channels = cv2.split(image_np)
            low_freq_channels = []

            for channel in channels:
                # Step 1: Apply FFT to the channel
                freq_domain = np.fft.fft2(channel)
                freq_domain_shifted = np.fft.fftshift(freq_domain)

                # Step 2: Create a low-pass filter (a circular mask in the center)
                rows, cols = channel.shape
                center_row, center_col = rows // 2, cols // 2
                mask = np.zeros((rows, cols), np.uint8)
                cv2.circle(mask, (center_col, center_row), radius, 1, thickness=-1)

                # Step 3: Apply the mask to the frequency domain
                low_freq_shifted = freq_domain_shifted * mask

                # Step 4: Shift back and apply Inverse FFT
                low_freq = np.fft.ifftshift(low_freq_shifted)
                spatial_domain = np.fft.ifft2(low_freq)

                # Step 5: Take only the real part and normalize the values
                spatial_domain = np.real(spatial_domain)
                spatial_domain = np.clip(spatial_domain, 0, 255).astype(np.uint8)

                low_freq_channels.append(spatial_domain)

            # Combine the channels back into an RGB image in NumPy format
            low_freq_img_np = cv2.merge(low_freq_channels)

            # Convert the result back to a PyTorch tensor and normalize
            low_freq_tensor = torch.from_numpy(low_freq_img_np).permute(2, 0, 1).float().cuda() / 255.0  # Normalize to [0, 1]
            
            # Append to the list of processed images in batch
            low_freq_batch.append(low_freq_tensor)

        # Stack all images in the batch to create a single batch tensor
        low_freq_batch_tensor = torch.stack(low_freq_batch)

        return low_freq_batch_tensor



    def forward(self, image, message):

        low_pass_image = self.low_pass_filter(image)
        encoded_image = self.conv_layers(low_pass_image)
        
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
