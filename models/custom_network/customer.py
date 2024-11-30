# custom_network.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLOv10
from torchvision.models import get_model_weights
from PIL import Image
import os
import shutil
import random
import shutil

class CustomNetwork:
    def __init__(self, mode, detector_cfg, classifier_cfg, training_cfg=None, prediction_cfg=None):
        self.mode = mode
        self.device = 'cpu'

        # 初始化组件
        self._init_detector(detector_cfg)
        self._init_classifier(classifier_cfg)
        self._init_processor()

        

        # 根据模式初始化其他参数
        if self.mode == 'train' and training_cfg is not None:
            self.training_cfg = training_cfg
            self.device = training_cfg.get('device', 'cpu')
        elif self.mode == 'predict' and prediction_cfg is not None:
            self.prediction_cfg = prediction_cfg
            self.device = prediction_cfg.get('device', 'cpu')
        else:
            raise ValueError("Invalid mode or missing configuration.")

    def _init_detector(self, detector_cfg):
        model_name = detector_cfg.get('model_name', 'yolov5s')
        pretrained = detector_cfg.get('pretrained', True)
        model_path = detector_cfg.get('model_path', None)
        
        self.expected_type_id = detector_cfg.get('expected_type_id', '80')
        if model_name.startswith('yolov10'):
            self.detector = YOLOv10(model_path,verbose=False)
        elif model_name.startswith('yolov5'):
            self.detector = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained,verbose=False)
        
        # to_device
        self.detector.to(self.device)

    def _init_classifier(self, classifier_cfg):
       
        model_name = classifier_cfg.get('model_name', 'resnet18')
        num_classes = classifier_cfg.get('num_classes', 10)
        pretrained = classifier_cfg.get('pretrained', False)
        model_path = classifier_cfg.get('model_path', None)
        weights = get_model_weights(model_name)

        self.classifier = getattr(models, model_name)(weights=weights)
        
        # 修改最后一层
        if model_name.startswith('resnet'):
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        elif model_name.startswith('vgg') or model_name.startswith('alexnet'):
            self.classifier.classifier[6] = nn.Linear(self.classifier.classifier[6].in_features, num_classes)
        # 可根据需要添加更多模型的适配

        # 如果提供了 model_path，则加载模型权重
        if model_path and pretrained:
            self.classifier.load_state_dict(torch.load(model_path))
        
        # to_device
        self.classifier.to(self.device)

    def _init_processor(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 如果需要，添加归一化
        ])

    def split_dataset(self, data_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        classes = os.listdir(data_dir)
        for cls in classes:
            cls_path = os.path.join(data_dir, cls)
            images = os.listdir(cls_path)
            random.shuffle(images)
            total = len(images)
            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)

            splits = {
                'train': images[:train_end],
                'val': images[train_end:val_end],
                'test': images[val_end:]
            }

            for split in splits:
                split_dir = os.path.join(output_dir, split, cls)
                os.makedirs(split_dir, exist_ok=True)
                for img_name in splits[split]:
                    src = os.path.join(cls_path, img_name)
                    dst = os.path.join(split_dir, img_name)
                    shutil.copy(src, dst)

    def process_detections(self, image, detections):
        cropped_images = []
        
        
        # Iterate over the detections (now using the 'data' field)
        for detection_all in detections:
            detection = detection_all.data
            if detection_all.id != self.expected_type_id:
                continue
            # Unpack detection values (x_min, y_min, x_max, y_max, conf, cls)
            x_min, y_min, x_max, y_max, conf, cls = detection
            
            # Crop the image using the bounding box
            cropped_img = image.crop((x_min, y_min, x_max, y_max))
            
            # Apply the transformation
            cropped_img = self.transform(cropped_img)
            
            # Append the processed image to the list
            cropped_images.append(cropped_img)
        
        # If there are cropped images, stack them into a tensor, otherwise return None
        if cropped_images:
            return torch.stack(cropped_images)
        else:
            return None


    def train(self):
        # 获取训练配置
        data_dir = self.training_cfg.get('data_dir')
        output_dir = self.training_cfg.get('temp_dir', 'temp_data')

        # 数据集划分
        if not os.path.exists(output_dir):
            self.split_dataset(data_dir, output_dir)
            print("Dataset splitted successfully.")
        else:
            shutil.rmtree(output_dir)  # 递归删除非空目录
            self.split_dataset(data_dir, output_dir)

        # 定义数据集和数据加载器
        train_loader, val_loader = self._get_data_loaders(output_dir, self.training_cfg.get('batch_size', 32))

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            self.classifier.parameters(),
            lr=self.training_cfg.get('learning_rate', 0.001),
            momentum=self.training_cfg.get('momentum', 0.9)
        )

        num_epochs = self.training_cfg.get('num_epochs', 20)

        # 训练循环
        for epoch in range(num_epochs):
            self.classifier.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                # 打印输出和标签的形状
                inputs = self._process_batch(images)
                if inputs is None:
                    continue

                optimizer.zero_grad()
                outputs = self.classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 计算并打印进度
                progress = (batch_idx + 1) / len(train_loader)
                if progress % 0.1 <= 1 / len(train_loader):
                    print(f'Epoch [{epoch+1}/{num_epochs}], Progress: {progress*100:.1f}%, Loss: {loss.item():.4f}')


            # # Validation step
            self.classifier.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = self.classifier(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")
            
        print("Training complete!")
            

        # 保存模型
        save_path = self.training_cfg.get('save_path', 'classifier.pth')
        torch.save(self.classifier.state_dict(), save_path)

    def predict(self):
        # 加载训练好的模型
        #self.classifier.load_state_dict(torch.load('classifier.pth',map_location="cpu", weights_only=True))
        self.classifier.eval()

        # 获取预测配置
        predict_dir = self.prediction_cfg.get('predict_dir')
        output_dir = self.prediction_cfg.get('output_dir')
        os.makedirs(output_dir, exist_ok=True)

        image_names = os.listdir(predict_dir)
        for img_name in image_names:
            img_path = os.path.join(predict_dir, img_name)
            image = Image.open(img_path).convert('RGB')

            # 检测
            results = self.detector(image)
            detections = results.xyxy[0]

            # 数据处理
            inputs = self.process_detections(image, detections)
            if inputs is None:
                continue

            # 预测
            outputs = self.classifier(inputs)
            _, preds = torch.max(outputs, 1)

            # 保存结果
            result_path = os.path.join(output_dir, img_name + '_result.txt')
            with open(result_path, 'w') as f:
                f.write(str(preds.numpy()))

    def _get_data_loaders(self, data_dir, batch_size):
        from torch.utils.data import DataLoader, Dataset

        class CustomDataset(Dataset):
            def __init__(self, data_dir, transform=None):
                self.data = []
                self.labels = []
                self.transform = transform
                self.classes = os.listdir(data_dir)
                self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

                for cls_name in self.classes:
                    cls_path = os.path.join(data_dir, cls_name)
                    img_names = os.listdir(cls_path)
                    for img_name in img_names:
                        img_path = os.path.join(cls_path, img_name)
                        self.data.append(img_path)
                        self.labels.append(self.class_to_idx[cls_name])

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                img_path = self.data[idx]
                label = self.labels[idx]
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return image, label

        train_dataset = CustomDataset(os.path.join(data_dir, 'train'), transform=self.transform)
        val_dataset = CustomDataset(os.path.join(data_dir, 'val'), transform=self.transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def _process_batch(self, images):
        inputs = []
        #print(len(images))
        for img_tensor in images:
            img = transforms.ToPILImage()(img_tensor)
            #print(self.detector.verbose)
            results = self.detector(img,verbose=False)
            
            
            # 使用 results.boxes 获取检测框
            detections = results[0].boxes
            
            if detections is not None and len(detections) > 0:
                detections = detections.cpu().numpy()  # 转换为 numpy 数组
                input_tensor = self.process_detections(img, detections)
                if input_tensor is not None and len(input_tensor) == 1:
                    inputs.append(input_tensor)
                else:
                    input_tensor = self.transform(img).unsqueeze(0)
                    inputs.append(input_tensor)
            else:
                # 如果没有检测到对象，将整个图像传入分类器
                input_tensor = self.transform(img).unsqueeze(0)  # 添加一个维度，确保 batch size 为 1
                inputs.append(input_tensor)
        
        if not inputs:
            return None
        
        # 合并所有图像的输入，形成一个大的 batch
        # 确保输入张量的维度一致
        inputs = torch.cat(inputs, dim=0)
        #print(f"Inputs shape: {inputs.shape}")
        
        # 返回输入时，确保返回的数据是一个批次
        return inputs




    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'predict':
            self.predict()
        else:
            print("Invalid mode. Please choose 'train' or 'predict'.")
