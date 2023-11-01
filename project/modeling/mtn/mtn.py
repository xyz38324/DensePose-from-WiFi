import torch.nn as nn
import torch
from detectron2.config import configurable
from .build import ModalityTranslationNetwork_REGISTRY
ModalityTranslationNetwork_REGISTRY.register()
class ModalityTranslationNetwork(nn.Module):
    '''
    input data should be [5,     30,       3,       3]
                         [batch,frequency,emitter,receiiver]
    convert csi matrix to spatial domain(3*720*720)

    '''

    def __init__(self):
        super(ModalityTranslationNetwork, self).__init__()
        
        # Define the MLPs for amplitude and phase
        self.amplitude_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(150 * 3 * 3, 1024),
            #nn.ReLU(),
            nn.Linear(1024, 600)
        )
        
        self.phase_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(150 * 3 * 3, 1024),
            #nn.ReLU(),
            nn.Linear(1024,600)
        )
        
        # Define the fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(600 * 2, 1024),
            #nn.ReLU(),
            nn.Linear(1024, 576),
        )
        
        # Define the convolution blocks
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), 
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 6x6
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output size: 64*6x6
            #nn.ReLU(),
        )
        
        # Define the deconvolution layers
        self.deconv_layers = nn.Sequential(           
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32x12x12
            #nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 16x24x24
            #nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),   # Output: 8x48x48
            #nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 4x96x96
            #nn.ReLU(),
            nn.ConvTranspose2d(4, 3, kernel_size=3, stride=4, padding=1, output_padding=1),    # Output: 3x384x384
            #nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # Output: 3x768x768
            #nn.ReLU(),
            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1)     # Output: 3x1536x1536
        )

       
    def forward(self, amplitude_tensor, phase_tensor):
        # Encode the amplitude and phase tensors
        amplitude_features = self.amplitude_encoder(amplitude_tensor)
        phase_features = self.phase_encoder(phase_tensor)
        
        # Concatenate and fuse the features
        fused_features = torch.cat((amplitude_features, phase_features), dim=1)
        fused_features = self.fusion_mlp(fused_features)
        
        # Reshape and process through convolution blocks
        reshaped_features = fused_features.view(-1, 1, 24, 24)
        conv_output = self.conv_blocks(reshaped_features)
        
        # Process through deconvolution layers
        deconv_output = self.deconv_layers(conv_output)
        # For a center crop
        start_dim3 = (deconv_output.size(2) - 720) // 2  # Center crop in dim3
        start_dim4 = (deconv_output.size(3) - 1080) // 2  # Center crop in dim4

        cropped_tensor = deconv_output[:, :, start_dim3:start_dim3+720, start_dim4:start_dim4+1080]

        
        return cropped_tensor
