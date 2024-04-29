from ultralytics import YOLO
import segmentation_models_pytorch as smp
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import albumentations as A
from torch.utils.data import DataLoader
from random import randint
import os
import collections
import torch.nn.functional as F
import streamlit as st
from torchvision.models import efficientnet_b3
from torchvision.models import EfficientNet_B3_Weights
from torchvision.models import mobilenet_v3_small
from torchvision.models import MobileNet_V3_Small_Weights
import ssl
from streamlit_gsheets import GSheetsConnection
from collections import Counter
from PIL import Image
from albumentations.augmentations.transforms import Normalize
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
import cv2 as cv

ssl._create_default_https_context = ssl._create_unverified_context

#################################################################


classes = {
    0: "Actinic keratosis",
    1: "Basal cell carcinoma",
    2: "Benign keratosis",
    3: "Dermatofibroma",
    4: "Melanocytic nevus",
    5: "Melanoma",
    6: "Squamous cell carcinoma",
    7: "Vascular lesion"
}

class2id = {
    "first degree": 0,
    "second degree": 1,
    "third degree": 2
}
id2class = {
    0: "first degree",
    1: "second degree",
    2: "third degree"

}


class MyEfficientnetB3(torch.nn.Module):
    def __init__(self, network):
        super(MyEfficientnetB3, self).__init__()
        self.fc1 = torch.nn.Linear(1536, 1024)

        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 128)
        self.fc4 = torch.nn.Linear(128, 8)
        self.network = network

        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.network(x)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x


class MyMobilenet(torch.nn.Module):
    def __init__(self, network):
        super(MyMobilenet, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 8)
        self.network = network

        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.network(x)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


class StudentModel(torch.nn.Module):
    def __init__(self, network):
        super(StudentModel, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 8)
        self.network = network

        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.network(x)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


transform_val = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

inv_transform = A.Compose([
    A.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                max_pixel_value=1)
])
load_transform = transforms.Compose([
    transforms.Resize((256, 256))
])

transform_vae = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


@st.cache_resource()
def load_models():
    # Model1
    efficientnet = efficientnet_b3(weights=None)
    efficientnet.classifier = torch.nn.Sequential(*list(efficientnet.classifier.children())[:-1])
    state = torch.load("./ClassificationModels/MyEfficientnet_256_100_long_best_model.pt", map_location='cpu')
    model1 = MyEfficientnetB3(efficientnet)
    model1.load_state_dict(state['state_dict'])
    model1.eval()
    # Model3
    mobilenet = mobilenet_v3_small(weights=None)
    mobilenet.classifier = torch.nn.Sequential(*list(mobilenet.classifier.children())[:-1])
    state = torch.load(
        "./ClassificationModels/MyMobilenet_256_100_long_best_model.pt",
        map_location='cpu')
    model3 = MyMobilenet(mobilenet)
    model3.load_state_dict(state['state_dict'])
    model3.eval()

    # Model4
    mobilenet = mobilenet_v3_small(weights=None)
    mobilenet.classifier = torch.nn.Sequential(*list(mobilenet.classifier.children())[:-1])
    state = torch.load("ClassificationModels/DistMobileNet_50_best_model.pt", map_location='cpu')
    model4 = StudentModel(mobilenet)
    model4.load_state_dict(state['state_dict'])

    # SegModel
    seg_model = smp.Unet(encoder_name="mobilenet_v2", encoder_depth=4, encoder_weights=None,
                         decoder_channels=(128, 64, 32, 16), decoder_use_batchnorm=True,
                         in_channels=3, classes=1, activation=None
                         )
    state = torch.load("SegmentaionModels/UnetMobileNet_best_model.pt", map_location='cpu')
    seg_model.load_state_dict(state['state_dict'])
    seg_model.eval()

    detection_model = YOLO("DetectionModels/yolov8xNewData100.pt")

    return model1, model3, model4, seg_model, detection_model


with st.spinner('Models is being loaded..'):
    model1, model3, model4, seg_model, detection_model = load_models()
    ensemble_model = [model1, model3, model4]
st.title('Medical application')

page_name = ['Identify a mole', 'Determine the location and degree of burn']
page = st.radio('Navigation', page_name)

if page == "Identify a mole" or page == "Determine the location and degree of burn":
    file = st.file_uploader("Please upload an image", type=["jpg", "png"])


def predict_cancer(img, models):
    for model in models:
        model.eval()

    img = img.convert('RGB')
    img = np.array(img)
    transformed = transform_val(image=img)
    t_img = torch.tensor(np.transpose(transformed['image'], (2, 0, 1)))
    t_img = torch.unsqueeze(t_img, 0)

    preds = []
    for model in models:
        y_predicted = model(t_img)
        class_id = torch.argmax(y_predicted, axis=1).detach().cpu().numpy()
        preds.append(class_id[0])
    pred = Counter(preds).most_common(1)[0][0]

    return classes[pred]


def predict_mask(network, img):
    network.eval()

    img = np.array(img)
    transformed = transform_val(image=img)
    t_img = torch.tensor(np.transpose(transformed['image'], (2, 0, 1)))
    t_img = torch.unsqueeze(t_img, 0)
    t_img = t_img

    mask_pred = network(t_img)

    return mask_pred


def return_box_img(img, model):
    img_np = np.array(img)
    results = model(img)
    boxes = results[0].boxes

    thickness = 4
    color = (255, 0, 0)

    for i in range(len(boxes.xyxy)):
        # Формируем координаты левого верхнего угла и правого нижнего
        start_points = (int(boxes.xyxy[i][0].cpu()), int(boxes.xyxy[i][1].cpu()))
        end_points = (int(boxes.xyxy[i][2].cpu()), int(boxes.xyxy[i][3].cpu()))
        # Строим прямоугольник
        img_np = cv.rectangle(img_np, start_points, end_points, color, thickness)

        # Пишем текст класса
        img_np = cv.putText(
            img_np,
            id2class[int(boxes.cls[i])],
            org=start_points,
            fontFace=cv.FONT_HERSHEY_PLAIN,
            fontScale=5.75,
            color=(0, 0, 255),
            thickness=6
        )

    return img_np


if page == "Identify a mole" or page == "Determine the location and degree of burn":
    if file is None:
        pass
    else:
        if "nav" not in st.session_state:
            if page == "Identify a mole":
                st.session_state["nav"] = 1
            elif page == "Determine the location and degree of burn":
                st.session_state["nav"] = 2

if page == "Identify a mole" or page == "Determine the location and degree of burn":
    if file is None:
        pass
    else:
        if page == "Identify a mole":
            if st.session_state["nav"] == 1:
                image = Image.open(file)
                if "no anomaly" == "no anomaly":
                    with st.spinner('Makeing predictions..'):
                        prediction = predict_cancer(image, ensemble_model)
                    st.image(load_transform(image), caption=prediction)

                    with st.spinner('Makeing mask..'):
                        mask = predict_mask(seg_model, image)
                    mask = torch.squeeze(mask)
                    mask = mask > -0.3
                    mask = mask.byte()
                    mask = mask.detach().to('cpu').numpy()

                    pil_img = Image.fromarray(mask * 255, mode="L")
                    st.image(pil_img, caption='lesion')

                    if prediction == "Melanoma":
                        st.write(
                            "Меланома(Melanoma) — это тип рака кожи, который может распространяться на другие участки тела. Основной причиной меланомы является ультрафиолетовый свет, который исходит от солнца и используется в соляриях.")
                    elif prediction == "Actinic keratosis":
                        st.write(
                            "Актинический кератоз(Actinic keratosis) – это грубое чешуйчатое пятно или шишка на коже. Это также известно как солнечный кератоз. Актинический кератоз очень распространен, и он есть у многих людей. Они вызваны ультрафиолетовым (УФ) повреждением кожи. Некоторые актинические кератозы могут перерасти в плоскоклеточный рак кожи.")
                    elif prediction == "Basal cell carcinoma":
                        st.write(
                            "Базальноклеточная карцинома(Basal cell carcinoma) — это разновидность рака кожи. Базальноклеточная карцинома начинается в базальных клетках — типе клеток кожи, которые производят новые клетки кожи по мере отмирания старых. Базальноклеточная карцинома часто проявляется в виде слегка прозрачной шишки на коже, хотя может принимать и другие формы.")
                    elif prediction == "Benign keratosis":
                        st.write(
                            "Себорейный кератоз (Benign keratosis) — распространенное нераковое (доброкачественное) новообразование кожи. С возрастом люди имеют тенденцию получать их больше. Себорейные кератозы обычно имеют коричневый, черный или светло-коричневый цвет. Наросты (поражения) выглядят восковыми или чешуйчатыми и слегка приподнятыми.")
                    elif prediction == "Dermatofibroma":
                        st.write(
                            "Дерматофиброма(Dermatofibroma) — это распространенное разрастание фиброзной ткани, расположенной в дерме (более глубоком из двух основных слоев кожи). Оно доброкачественное (безвредное) и не превратится в рак. Хотя дерматофибромы безвредны, по внешнему виду они могут быть похожи на другие опухоли кожи.")
                    elif prediction == "Melanocytic nevus":
                        st.write(
                            "Меланоцитарный невус(Melanocytic nevus) — медицинский термин, обозначающий родинку. Невусы могут появиться на любом участке тела. Они доброкачественные (не раковые) и обычно не требуют лечения. В очень небольшом проценте меланоцитарных невусов может развиться меланома.")
                    elif prediction == "Squamous cell carcinoma":
                        st.write(
                            "Плоскоклеточный рак кожи(Squamous cell carcinoma) — это тип рака, который начинается с разрастания клеток на коже. Это начинается в клетках, называемых плоскими клетками. Плоские клетки составляют средний и наружный слои кожи. Плоскоклеточный рак — распространенный тип рака кожи. 11 авг. 2023г.")
                    elif prediction == "Vascular lesion":
                        st.write(
                            "Сосудистые поражения(Vascular lesion) — это аномальные разрастания или пороки развития кровеносных сосудов, которые могут возникать в различных частях тела. Они могут быть врожденными или приобретенными и могут возникнуть в результате травмы, инфекции или других заболеваний.")
                else:
                    st.write("This isn't mole or poor quality image. Upload another image")
            else:
                st.session_state["nav"] = 1
        if page == "Determine the location and degree of burn":
            if st.session_state["nav"] == 2:
                image = Image.open(file)
                box_image = return_box_img(image, detection_model)
                st.image(np.array(box_image))
            else:
                st.session_state["nav"] = 2
st.write("Done")
