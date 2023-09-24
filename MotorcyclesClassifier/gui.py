import gradio as gr
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image


def get_model(path):
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    num_ftrs = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1),

        nn.Linear(num_ftrs, num_ftrs),
        nn.BatchNorm1d(num_ftrs),
        nn.ReLU(),
        nn.Dropout(.2),

        nn.Linear(num_ftrs, 3)
    )

    model.load_state_dict(torch.load(path))
    return model


def predict(img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model('ConvNeXt_Base_finetuned.pth').to(device)

    labels = ['city', 'sport', 'tourist']
    img = transforms.Resize((224, 224))(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        probas = torch.softmax(model(img.to(device)), dim=1)[0]
    
    return {label: proba for label, proba in zip(labels, [proba.item() for proba in probas])}


if __name__ == '__main__':
    gr.Interface(fn=predict,
                 inputs=gr.Image(type='pil'),
                 outputs=gr.Label(num_top_classes=3),
                 examples=['im_city.jpg', 'im_sport.jpg', 'im_tourist.jpg']
    ).launch(share=True)
