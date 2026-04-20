import torch


print("Version de PyTorch :", torch.__version__)
print("CUDA disponible ?", torch.mps.is_available())
print("Nom du GPU :", torch.mps.get_device_name(0))
# Creer un petit tenseur sur GPU pour tester
x = torch.tensor([1.0, 2.0, 3.0]).cuda()
print("Tenseur sur GPU :", x)
print("Device du tenseur :", x.device)