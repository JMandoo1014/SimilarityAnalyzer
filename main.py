from PyQt5 import QtCore, QtWidgets, uic
import os
import sys
import tensorflow as tf
import numpy as np
import cv2
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.layers import Layer # type: ignore

def resource_path(relative_path) :
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

macro_form = resource_path('SimilarityAnalyzer.ui')

form_class = uic.loadUiType(macro_form)[0]

class App(QtWidgets.QMainWindow, form_class) :
    is_data_disabled = False

    def __init__(self):
        super().__init__()

        self.okt = Okt()
        self.vectorizer = TfidfVectorizer()

        self.setupUi(self)
        self.result.setReadOnly(True)

        self.text.clicked.connect(self.select_method)
        self.image.clicked.connect(self.select_method)

        self.OpenBtn1.clicked.connect(self.btn_fun_FileLoad)
        self.OpenBtn2.clicked.connect(self.btn_fun_FileLoad)

        self.StartBtn.clicked.connect(self.start)

        self.file1 = None
        self.file2 = None

    def log(self, msg) :
        global now
        self.result.append(msg)

    def disableElements(self, *elements):
        for element in elements:
            element.setEnabled(False)

    def select_method(self) :
        select_sender = self.sender()
        if select_sender == self.text:
            if select_sender.isChecked():
                self.image.setChecked(False)
        elif select_sender == self.image :
            if select_sender.isChecked() :
                self.text.setChecked(False)

    def btn_fun_FileLoad(self) :
        fname = QtWidgets.QFileDialog.getOpenFileName(self,'','','Text(*.txt);;PNG(*.png);;JPG(*.jpg *.jpeg)')
        if fname[0]:
            sender = self.sender()
            if sender == self.OpenBtn1 :
                self.Content1.setText(fname[0])
                self.file1 = fname[0]
                self.log("파일 업로드에 성공했습니다.")
            elif sender == self.OpenBtn2 :
                self.Content2.setText(fname[0])
                self.file2 = fname[0]
                self.log("파일 업로드에 성공했습니다.")
        else:
            self.file1 = None
            self.file2 = None

    def start(self) :
        try :
            if self.file1 is None or self.file2 is None:
                QtWidgets.QMessageBox.critical(self, '오류', '파일을 업로드 해 주십시오')
                return

            if self.text.isChecked() :
                self.disableElements(self.OpenBtn1, self.OpenBtn2, self.image, self.text, self.StartBtn)
                self.similarity_percentage_txt = self.compute_cosine_similarity(self.file1, self.file2) * 100
                self.log(f"유사도 : {self.similarity_percentage_txt:.2f}%")
            elif self.image.isChecked() :
                self.disableElements(self.OpenBtn1, self.OpenBtn2, self.image, self.text, self.StartBtn)
                self.predict_similarity('siamese_model_vgg16.keras')
            else :
                QtWidgets.QMessageBox.critical(self, '오류', '파일 형식을 정해주십시오.')
        except Exception as e :
            self.log(f'에러발생 : {e}')



#######################################################################################
    def preprocess_text(self, text):
        # 한글과 숫자를 포함한 문장을 전처리
        tokens = self.okt.pos(text)
        words = [token[0] for token in tokens if token[1] in ['Alpha', 'Number', 'Noun']]
        return ' '.join(words)
    
    def compute_cosine_similarity(self, text1, text2):
        if text1 and text2:
            try_encodings = ['utf-8', 'cp949', 'euc-kr']  # 시도할 인코딩 목록

            # 첫 번째 파일 읽기 시도
            for encoding in try_encodings:
                try:
                    with open(text1, 'r', encoding=encoding) as f1:
                        self.data1 = f1.read()
                        break  # 성공하면 반복 중단
                except UnicodeDecodeError:
                    continue  # 다음 인코딩 시도

            # 두 번째 파일 읽기 시도
            for encoding in try_encodings:
                try:
                    with open(text2, 'r', encoding=encoding) as f2:
                        self.data2 = f2.read()
                        break  # 성공하면 반복 중단
                except UnicodeDecodeError:
                    continue  # 다음 인코딩 시도

            # 텍스트 전처리
            processed_text1 = self.preprocess_text(self.data1)
            processed_text2 = self.preprocess_text(self.data2)
            
            # 벡터화
            vectorized = self.vectorizer.fit_transform([processed_text1, processed_text2])
            vectors = vectorized.toarray()
            
            # 코사인 유사도 계산
            self.similarity_txt = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))[0][0]
            return self.similarity_txt
        else:
            raise ValueError("파일 경로를 모두 입력해 주세요.")

########################################################################################

    # 이미지 전처리 함수
    def preprocess_image(self, image_path):
        self.img = cv2.imread(image_path)
        self.img = cv2.resize(self.img, (128, 128))
        self.img = self.img.astype('float32') / 255.0
        return self.img

    # 사용자 정의 L2 거리 계산 레이어
    class L2DistanceLayer(Layer):
        def call(self, inputs):
            x, y = inputs
            return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

    # Contrastive Loss 정의
    def contrastive_loss(self, y_true, y_pred):
        margin = 1.0
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    # 유사도 예측 함수
    def predict_similarity(self, model_path):
        self.img1 = self.preprocess_image(self.file1)
        self.img2 = self.preprocess_image(self.file2)
        self.img1 = np.expand_dims(self.img1, axis=0)
        self.img2 = np.expand_dims(self.img2, axis=0)

        self.model = load_model(model_path, custom_objects={'contrastive_loss': self.contrastive_loss, 'L2DistanceLayer': self.L2DistanceLayer})

        self.similarity_img = self.model.predict([self.img1, self.img2])[0][0]
        self.similarity_percentage_img = self.similarity_img * 100

        self.log(f"유사도 : {self.similarity_percentage_img:.2f}%")




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    myWindow = App()
    myWindow.show()
    app.exec_()