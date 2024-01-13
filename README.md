### 간단한 프로젝트 
-------------------
사용자가 자신이 원하는 표정의 얼굴 사진이나 표정이 포함된 이미지 사진을 
텐서플로우를 활용하여 API를 적용시켜 사진에 대한 감정을 7가지
[화남,우울함,두려움,행복,슬픔,놀람,중림] 으로 분석하여 줌.

-예측 모델: 얼굴 감정 분석 알고리즘의 구현된 부분으로 얼굴의 7가지 감정으로 분류 된다.
-데이터 분석: 감정결과와 가지고 있는 movie.csv 를 바탕으로 같은 감정을 찾아 새로운 CSV 로 저장이 되는 부분

#####Feeling.ipynb 
Keras 와 TensorFlow 를 사용하여 딥러닝 모델 구축 및 훈련
##### test.py 
훈련돈 모델 적용시킨 후 파라미터 적용

