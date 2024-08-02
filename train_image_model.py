import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# 랜덤 시드 고정
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(str(image_path))  # 경로를 문자열로 변환
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    return img

# 데이터셋 생성 함수
def create_dataset(image_pairs, labels, batch_size):
    def generator():
        for (img1_path, img2_path), label in zip(image_pairs, labels):
            img1 = preprocess_image(img1_path)
            img2 = preprocess_image(img2_path)
            if img1 is not None and img2 is not None:
                yield (img1, img2), label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=((tf.float32, tf.float32), tf.int32),
        output_shapes=(((128, 128, 3), (128, 128, 3)), ())
    )

    dataset = dataset.shuffle(buffer_size=len(labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

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

# 모델 생성 함수 (드롭아웃 추가 및 Early Stopping 설정)
def create_siamese_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Transfer learning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # GlobalAveragePooling2D 레이어 추가
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout 레이어 추가
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout 레이어 추가
    x = Dense(64, activation='relu')(x)
    
    model = Model(inputs=base_model.input, outputs=x)

    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    output1 = model(input1)
    output2 = model(input2)

    l2_distance = L2DistanceLayer()([output1, output2])
    output = Dense(1, activation='sigmoid')(l2_distance)

    siamese_model = Model(inputs=[input1, input2], outputs=output)
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    siamese_model.compile(optimizer=sgd, loss=contrastive_loss, metrics=['accuracy'])

    return siamese_model

# Early Stopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

base_dir = Path(__file__).resolve().parent
cache_dir = base_dir / ".dataset"
anchor_images_path = cache_dir / "left"
positive_images_path = cache_dir / "right"

anchor_image_files = list(anchor_images_path.glob("*.jpg"))[:25]  # 확장자에 맞게 변경 가능
positive_image_files = list(positive_images_path.glob("*.jpg"))[:25]  # 확장자에 맞게 변경 가능


# 앵커-포지티브 이미지 쌍 생성
image_pairs = list(zip(anchor_image_files, positive_image_files))
similar_labels = [1] * len(image_pairs)  # 유사: 1

# 비슷하지 않은 이미지 쌍 생성
non_similar_pairs = []
for i in range(len(anchor_image_files)):
    for j in range(len(positive_image_files)):
        if i != j:
            non_similar_pairs.append((anchor_image_files[i], positive_image_files[j]))
non_similar_labels = [0] * len(non_similar_pairs)  # 비유사: 0

# 데이터셋 결합
all_pairs = image_pairs + non_similar_pairs
all_labels = similar_labels + non_similar_labels

# 데이터셋 분할
train_pairs, test_pairs, train_labels, test_labels = train_test_split(
    all_pairs, all_labels, test_size=0.2, random_state=random_seed
)

input_shape = (128, 128, 3)
batch_size = 32  # 배치 크기 설정
model_path = base_dir / 'siamese_model_vgg16.keras'

try:
    model = tf.keras.models.load_model(model_path, custom_objects={'contrastive_loss': contrastive_loss, 'L2DistanceLayer': L2DistanceLayer})
    print("이전 모델을 로드했습니다.")
except:
    print("새로운 모델을 생성합니다.")
    model = create_siamese_model(input_shape)

# 데이터셋 생성
train_dataset = create_dataset(train_pairs, train_labels, batch_size)
test_dataset = create_dataset(test_pairs, test_labels, batch_size)

# 모델 학습
print("모델 학습 시작")
for epoch in range(10):
    print(f'Epoch {epoch+1}/10')
    model.fit(train_dataset, epochs=1, validation_data=test_dataset)
    print(f'에폭 {epoch+1} 완료')
print("모델 학습 완료")

# 학습된 모델 저장
model.save(model_path)
print(f"모델 저장 완료: {model_path}")