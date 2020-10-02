# tf.keras
문법의 간소화를 통한 한가지만 남기는것이 목표

딥러닝이 발전함에 따라 다양한 네트워크가 개발됬지만 좋은 성능을 보이는 네트워크는 수십, 수백개의 레이어를 쌓은 경우가 대부분이고, 레이어가 늘어남에 따라 네트워크를 훈련시키는데 걸리는 시간도 증가합니다. 유명한 CNN중 하나인 ResNet-50은 8장 P100 GPU를 사용해 ImageNet의 사진을 잘 분류하도록 학습시키는데 29시간이 걸립니다. 페이스북이에서는 이걸 1시간으로 줄였지만 그 대신 더 많은 256장의 GPU를 사용했습니다. 최근 이미지 분류 문제에서 더 좋은 결과를 내고 있는 NASNet은 AutoML기법을 사용해 최적의 네트워크 구조를 스스로 학습하기 때문에 훨씬 더 많은 훈련 시간이 필요합니다.

다행스러운 점은 회사에서 일을 진행할떄는 연구자들이 자신이 만든 사전 훈련된 모델을 인터넷에 올려놓아 다른사람들이 사용할 수 있도록 해줍니다 이렇게 얻은 모델을 그대로 사용할 수 도 있고 전이학습이나 신경스타일 전이처럼 다른 과제를 위해 재가공해서 사용 할 수 도 있습니다.

# 텐서플로 허브

여기서 주로 사용하는 라이브러리인 텐서플로에서 제공하는 텐서플로 허브(TensorFlow Hub)는 재사용 가능한 모델을 쉽게 이용할 수 있는 라이브러리입니다.
텐서플로 허브 홈페이지에서는 이미지, 텍스트, 비디오 등의 분야에서 사전 훈련된 모델들을 검색가능합니다.

# 텐서플로 2.0을 이용한 턴서플로 허브 불러오는 방법

1. 텐서플로 허브에서 사전 훈련된 MobileNet 모델 불러오기
import tensorflow_hub as hub

mobile_net_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
model = tf.keras.Sequential([
  hub.kerasLyaer(handle = mobile_net_url, input_shape=(224,224,3), trainable=False)
])

model.summary()

# MobileNet 이란?

MobileNet은 계산 부담이 큰 컨볼루션 신경망을 연산 성능이 제한적인 모바일 환경에서도 작동 가능하도록 네트워크 구조를 경량화한것입니다.
MobileNet 버전2는 1을 개선했고 파라미터 수도 더 줄어들었습니다.

MobileNet은 ImageNet에 존재하는 1,000 종류의 이미지를 분류할 수 있으며, 이 가운데 어떤것에도 속하지 않는다고 판단될 때는 background에 해당 하는 인덱스 0을 반환합니다. 이미지의 분류는 수량(cock)과 암탉(hen)을 분류할 정도로 상세하고 화장지(toilet tissue)같은 사물도 포함되어있다.

MobileNet의 성능을 평가하기 위해 이미지를 학습시켰을 때 얼마나 적합한 라벨로 분류하는지 알아보겠습니다.
ImageNet의 데이터 주 ㅇ일부만 모아놓은 ImageNetV2를 사용하겠습니다. ImageNetV2는 아마존 매커니컬 터크를 이용해 다수의 참가자에게서 클래스 예측값을 받아서 선별한 데이터입니다. 여기서는 각클래스에서 가장 많은 선택을 받은 이미지 10장씩 모아놓은 10,000장의 이미지가 포함된 TopImages 데이터를 사용하겠습니다.

# imageNetV2-TopImages 불러오기

import os
import pathlib

content_data_url = '/content/sample_data'
data_root_orig = tf.keras.utils.get_file('imagenetV2', 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-topimages.tar.gz', cache_dir=content_data_url, extract=True)
data_root= pathlib.Path(content_data_url + '/datasets/imagenetv2-topimages')
print(data_root)

# 디렉터리 출력
for idx, item in enumerate(data_root.iterdir()):
  print(item)
  if idx == 9:
    break

# ImageNet 라벨 텍스트 불러오기

label_file = tf.keras.utils.get_file('label', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

label_text = None
with open(label_file, 'r') as f:
  label_text = f.read().split('\n')[:-1]
print(len(label_Text))
print(label_text[:10])
print(label_text[-10:])

# 이미지 확인

import PIL.Image as Image
import matplotlib.pyplot as plt
import random

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

#이미지를 랜덤하게 섞습니다
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print('image_count', image_count)

plt.figure(figsize=(12, 12))

for c in range(9):
  image_path = random.chice(all_image_paths)
  plt.subplot(3,3,c+1)
  plt.imshow(plt.imread(image_path))
  idx = int(image_path.split('/')[-2]) + 1
  plt.title(str(idx) + ', ' + label_text[idx])
  plt.axis('off')
plt.show()

# 텐서플로 허브 모델 사용법

텐서플로 허브에 올라와 있는 모델은 hub.KerasLayer()명령으로 tf.keras에서 사용 가능한 레이어로 변환 할 수 있습니다.


