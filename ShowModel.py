import netron
import networks
import torch

# resU_net

ResUnet=networks.Resnet34_Unet(1,1)
input=torch.rand(1,1, 512, 512)
torch.onnx.export(ResUnet,input,"model_Structure/ResU_net.onnx")
netron.start("model_Structure/ResU_net.onnx")