# main.py

from models.custom_network.customer import CustomNetwork
import yaml

def main():
    # 加载配置文件
    with open('configs/config.yaml', 'r',encoding='utf8') as f:
        config = yaml.safe_load(f)

    mode = config.get('mode', 'train')
    detector_cfg = config.get('detector', {})
    classifier_cfg = config.get('classifier', {})
    training_cfg = config.get('training', {}) if mode == 'train' else None
    prediction_cfg = config.get('prediction', {}) if mode == 'predict' else None

    # 初始化自定义网络
    network = CustomNetwork(
        mode=mode,
        detector_cfg=detector_cfg,
        classifier_cfg=classifier_cfg,
        training_cfg=training_cfg,
        prediction_cfg=prediction_cfg
    )
    # 运行
    network.run()

if __name__ == '__main__':
    main()
