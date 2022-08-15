import torch
import torchvision
import torch.nn as nn

# An instance of your model.
model = torchvision.models.mobilenet_v3_small()
model.classifier = nn.Sequential(
    nn.Linear(in_features=576, out_features=1024, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1024, out_features=101, bias=True))
model.load_state_dict(torch.load("./weights/ResNet50_epoch_89.pth")["model_state_dict"])

# Switch the model to eval model
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 128, 128)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Save the TorchScript model
traced_script_module.save("RotNetPyTorch_model.pt")
