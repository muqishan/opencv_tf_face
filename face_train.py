from load_dataset import load_dataset, resize_image, IMAGE_SIZE
from setting import cascade_path,face_path,labels_dict,model_path  # 导入人脸识别分类器 ,数据集存放根目录
import tensorflow as tf
import numpy as np
import cv2


class Faces(object):
    def __init__(self):
        self.file_path = face_path
        self.model_path = model_path
        self.train_images, self.train_labels = load_dataset(self.file_path)  # 加载训练集
        self.model = tf.keras.Sequential()  # 建立顺序模型
        self.image_size = IMAGE_SIZE
        self.labels = {v: k for k, v in labels_dict.items()}  # 反转字典，方便后续使用

    def tf_model(self):
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=self.train_images.shape[1:],
                                              activation='relu', padding='same'))

        # self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        # self.model.add(tf.keras.layers.MaxPool2D())
        # self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        # self.model.add(tf.keras.layers.GlobalAveragePooling2D())
        # self.model.add(tf.keras.layers.Dense(5, activation='softmax'))

        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.MaxPool2D())
        self.model.add(tf.keras.layers.Dropout(0.25))  # 抑制过拟合
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.MaxPool2D())
        self.model.add(tf.keras.layers.Dropout(0.25))
        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        # self.model.add(tf.keras.layers.MaxPool2D())
        # self.model.add(tf.keras.layers.Dropout(0.5))
        # self.model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        # self.model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(tf.keras.layers.GlobalAveragePooling2D())
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(6, activation='softmax'))

    def show_summary(self):  # 展示模型概况
        self.model.summary()

    def resize_test_image(self, image_path, image=None):
        if image is None:
            image = cv2.imread(image_path)
            i = resize_image(image)  # 需要进行预测的数据也要进行等比缩放
            inp = i.reshape(1, 64, 64, 3)  # 重新确定image的形状 测试数据输入维度应当为4维 数量*IMAGESIZE*IMAGESIZE*RGB
            inp = inp.astype('float32')  # dtype需要一致 这里更该为 float32
            return inp
        else:
            i = resize_image(image)  # 需要进行预测的数据也要进行等比缩放
            inp = i.reshape(1, 64, 64, 3)  # 重新确定image的形状 测试数据输入维度应当为4维
            inp = inp.astype('float32')  # dtype需要一致 这里更该为 float32
            return inp

    def face_fit(self):
        self.model.compile(loss='sparse_categorical_crossentropy',
                           # optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6,
                           #                                   momentum=0.9, nesterov=True),
                           optimizer='adam',
                           metrics=['acc'])
        self.model.fit(self.train_images, self.train_labels, epochs=3)

    def image_test(self, file_path):
        test_image = self.resize_test_image(file_path)
        index = np.argmax(self.model.predict(test_image))  # 取出最大几率值对应索引
        print(self.labels)
        print('this is {}'.format(self.labels.get(index)))
        return self.labels.get(index)

    def video_test(self):
        color = (0, 255, 0)

        # 捕获指定摄像头的实时视频流
        origin = 0  # 改为视频路径则为识别视频中人脸
        cap = cv2.VideoCapture(origin)
        # 循环检测识别人脸
        while True:
            ret, frame = cap.read()  # 读取一帧视频
            if ret is True:
                # 图像灰化，降低计算复杂度
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue
            # 使用人脸识别分类器，读入分类器
            cascade = cv2.CascadeClassifier(cascade_path)
            # 利用分类器识别出哪个区域为人脸
            faceRects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, w, h = faceRect
                    # 截取脸部图像提交给模型识别这是谁
                    image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    image = self.resize_test_image(image_path='', image=image)
                    chenjia = self.model.predict(image)
                    index = np.argmax(chenjia)  # 取出最大几率值对应索引
                    print(chenjia, end=' ')
                    print(self.labels, index)
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                    # 文字提示是谁
                    cv2.putText(frame, self.labels.get(index),  # 获取labels中的标注
                                (x + 30, y + 30),  # 坐标
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                1,  # 字号
                                (255, 0, 255),  # 颜色
                                2)  # 字的线宽

            cv2.imshow("softmax", frame)

            # 等待按键响应 10毫秒
            k = cv2.waitKey(10)
            # 如果输入q则退出循环
            if k & 0xFF == ord('q'):
                break

        # 释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()


F = Faces()
F.tf_model()
F.face_fit()
F.image_test('test/259.jpg')
F.video_test()
