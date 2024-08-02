import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import Layer # type: ignore

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # 이미지를 128x128 크기로 리사이징
    img = img.astype('float32') / 255.0  # 이미지를 0에서 1 사이 값으로 정규화
    return img

# 사용자 정의 L2 거리 계산 레이어
class L2DistanceLayer(Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

# Contrastive Loss 정의
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)  # y_true를 float32로 변환
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

# 유사도 예측 함수
def predict_similarity(model_path, img1_path, img2_path):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    # 모델 로드
    model = load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss, 'L2DistanceLayer': L2DistanceLayer})

    similarity = model.predict([img1, img2])[0][0]
    similarity_percentage = similarity * 100

    return similarity_percentage

# 예측 결과 출력
model_path = 'C:/Users/Owner/Desktop/2024 Makerthon Project/main/siamese_model_vgg16.keras'
similarity_percentage = predict_similarity(model_path, 'C:/Users/Owner/Desktop/2024 Makerthon Project/main/1.PNG', 'C:/Users/Owner/Desktop/2024 Makerthon Project/main/2.PNG')
print(f'유사도: {similarity_percentage:.2f}%')
