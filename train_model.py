# FaceAge 관련 모델

import numpy as np
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 무작위 샘플링
print("데이터 로딩 중...")
data = np.load("data/training_dataset_preproc.npz")
X_all = data['arr_0']
y_all = data['arr_1']

np.random.seed(42)
indices = np.random.choice(len(X_all), size=10000, replace=False)
X = X_all[indices].astype(np.float16) / 255.0
y = y_all[indices]

# 2. 훈련/검증 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. FaceAge 유사 모델 정의 (AU 제외)
input_tensor = Input(shape=(160, 160, 3))
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
age_output = Dense(1)(x)
model = Model(inputs=input_tensor, outputs=age_output)

# 4. 컴파일
model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])

# 5. 학습
print("🚀 학습 시작")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10, batch_size=16)

# 6. 저장
model.save("models/my_faceage_model.h5", save_format="h5")
print("✅ 모델 저장 완료: models/my_faceage_model.h5")

# 7. 시각화
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE Over Epochs (FaceAge-style)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid()
plt.show()

