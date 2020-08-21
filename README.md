# Document Classification
## 관공서 서류와 같은 특징점이 있는 서류들을 분리(Custom Data)

### Dependencies
- python3
- Pytorch
- Cuda

### Data
1. 학습 데이터와 테스트 데이터를 비율을 나누어 진행
2. 라벨을 나눌 기준이 정해졌으면 파일의 이름에 라벨 이름을 지정.
(ex. label0_0, label0_1, label0_2..... label1_0, label1_1....)
3. 구현된 **CustomDataSet class**를 본인의 라벨에 맞게 변경.
- CustomData Example

![image](https://user-images.githubusercontent.com/37897891/90865944-257df000-e3ce-11ea-8258-59a5fdab46d9.png)

### Model
- Optimizer : Adam
- criterion : CrossEntropy
- 기본적인 CNN모델을 사용할 수도, 아래 reference에서 참고한 Spinal CNN을 사용할 수 있다.
- 변경방법은 _280_ line에 model을 설정할 수 있는 부분이 존재

### Accuracy

![image](https://user-images.githubusercontent.com/37897891/90865857-00897d00-e3ce-11ea-92ac-4f5ccb632e1a.png)

- Softmax 함수 추가하여 결과물 Score 표현 -> threshold 잡기 쉽게 하기 위함.

![image](https://user-images.githubusercontent.com/37897891/90866064-4cd4bd00-e3ce-11ea-8025-bd5bb535c157.png)

1. label Data.

![image](https://user-images.githubusercontent.com/37897891/90866239-92918580-e3ce-11ea-8493-507242b914b9.png)

2. unLabeled Data가 들어왔을 때

![image](https://user-images.githubusercontent.com/37897891/90866333-adfc9080-e3ce-11ea-911f-802733fb81e5.png)


## references
- https://github.com/dipuk0506/SpinalNet
(모델 코드 참조)
- (Custom data, trained data save, classify 및 score 값은 개인구현)
