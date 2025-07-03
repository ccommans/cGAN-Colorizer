import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Import your model definition
class UNetGenerator(nn.Module):
    def __init__(self, in_ch=2, out_ch=3, features=64):
        super().__init__()
        # Encoder
        self.down1 = self._block(in_ch, features)        # 2 -> 64, 256->128
        self.down2 = self._block(features, features*2)   # 64->128, 128->64
        self.down3 = self._block(features*2, features*4) # 128->256, 64->32
        self.down4 = self._block(features*4, features*8) # 256->512, 32->16
        # Bottleneck: reduce spatial to 8x8, then keep channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*16, kernel_size=4, stride=2, padding=1),  # 512->1024, 16->8
            nn.ReLU(),
            nn.Conv2d(features*16, features*8, kernel_size=3, stride=1, padding=1),   # 1024->512, 8->8
            nn.ReLU(),
        )
        # Decoder
        self.up1 = self._upblock(features*8, features*8)   # 512->512, 8->16
        self.up2 = self._upblock(features*16, features*4)  # (512+512)->256, 16->32
        self.up3 = self._upblock(features*8, features*2)   # (256+256)->128, 32->64
        self.up4 = self._upblock(features*4, features)     # (128+128)->64, 64->128
        # Final layer to restore size
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_ch, kernel_size=4, stride=2, padding=1),  # (64+64)->3, 128->256
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def _upblock(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        e1 = self.down1(x)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)

        b = self.bottleneck(e4)

        d1 = self.up1(b)                # 8->16
        d1 = torch.cat([d1, e4], dim=1) # ->1024 channels
        d2 = self.up2(d1)               # 16->32
        d2 = torch.cat([d2, e3], dim=1) # ->512 channels
        d3 = self.up3(d2)               # 32->64
        d3 = torch.cat([d3, e2], dim=1) # ->256 channels
        d4 = self.up4(d3)               # 64->128
        d4 = torch.cat([d4, e1], dim=1) # ->128 channels

        return self.final(d4)          # 128->256

@st.cache_resource
def load_model(checkpoint_path: str, device: torch.device):
    model = UNetGenerator(in_ch=2, out_ch=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    model.eval()
    return model

def preprocess_image(img: Image.Image, image_size=256):
    img_gray = img.convert("L")
    original_size = img.size
    img_resized = img_gray.resize((image_size, image_size))
    to_tensor = transforms.ToTensor()
    gray_tensor = to_tensor(img_resized)
    gray_np = np.array(img_resized)
    edge_np = cv2.Canny(gray_np, 100, 200) / 255.0
    edge_tensor = torch.from_numpy(edge_np).unsqueeze(0).float()
    inp = torch.cat([gray_tensor, edge_tensor], dim=0)
    inp = (inp - 0.5) / 0.5
    return inp.unsqueeze(0), original_size

def postprocess_and_display(output_tensor, original_size=None):
    out = output_tensor.squeeze().cpu().detach()
    out = (out * 0.5 + 0.5).clamp(0, 1)
    img = transforms.ToPILImage()(out)
    if original_size:
        img = img.resize(original_size)
    return img

def main():
    st.title("cGAN Colorization Demo")
    st.write("Upload a grayscale image, and see it colorized!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "./G.pth"  # update if needed
    model = load_model(checkpoint, device)

    uploaded_file = st.file_uploader("Choose a grayscale image", type=["jpg","jpeg","png","bmp"])
    if uploaded_file is not None:
        input_img = Image.open(uploaded_file)
        st.image(input_img, caption="Input Grayscale", use_container_width=True)

        inp, orig_size = preprocess_image(input_img, image_size=128)
        inp = inp.to(device)
        with torch.no_grad():
            output = model(inp)
        result_img = postprocess_and_display(output, original_size=orig_size)

        st.image(result_img, caption="Colorized Output", use_container_width=True)

if __name__ == "__main__":
    main()