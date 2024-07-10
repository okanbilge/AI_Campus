import torchvision.models as torchmodels
import timm
import torch


modelList = []
torch.hub._validate_https_requests = False

for model_name in ['alexnet', 'vgg16', 'resnet50', 'densenet121', 'shufflenet_v2_x1_0', 'resnext50_32x4d', 'wide_resnet50_2']:
        # Set a custom cache directory if needed


    model = getattr(torchmodels, model_name)(pretrained=False)
    
    if 'alexnet' in model_name:
        num_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_features, 2)
    elif 'vgg' in model_name:
        num_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_features, 2)
    elif 'resnet' in model_name:
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
    elif 'densenet' in model_name:
        num_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_features, 2)
    elif 'googlenet' in model_name:
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
    elif 'mobilenet' in model_name:
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, 2)
    elif 'shufflenet' in model_name:
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
    elif 'resnext' in model_name:
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
    elif 'wide_resnet' in model_name:
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
    
    modelList.append(model)

for model_name in ["vit_base_patch16_224", "vit_base_patch32_224"]:
    model = timm.create_model(model_name, pretrained=True)
    
    num_features = model.head.in_features
    model.head = torch.nn.Linear(num_features, 2)
    
    modelList.append(model)
    