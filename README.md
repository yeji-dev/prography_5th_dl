# prography_5th_dl
프로그라피 5기 지원자 홍예지

##Summary of assiginment
+ Project : 냉장고 속 물체 탐지(Object Detection in refrigerator)
+ Framework : `Keras`
+ Detection Network : `Mask RCNN`
+ Network Structure
    1. Multi Classification Model (CNN)
    2. Binary Classification Model (이미지에 객체있는지 분류)
    3. Regression Model (박스 그려주는 모델)

##Result
+ Input
   
   - `python test.py --model mask_rcnn_refri_cfg_0030.h5 --image apple_0068.jpg`
    
   - or `python test.py --image apple_0068.jpg`

+ Result
    - Object detection result
    ![result_predict](./readme/result_predict.png "Object detection")
    
    - Results when inserting unrelated pictures
    ![result_unrelated](./readme/result_unrelated.png "Results when inserting unrelated pictures")
    
* Note: command line options
![help](./readme/cmdline_help.png "Show help option in the command line")

##Implementations
+ `mask_rcnn_refri_cfg_0030.h5`
    - this model has learned `detection_dataset` in the existing `mask_rcnn_coco.h5` model
    - training description
        - **Accurcy** : 
        - data size
            - train size : 1341, validation size : 169
        - batch_size : 1341, epoch : 30
        - learning duration : 1d
+ `test.py`
    - training and testing in 한파일에서 한번에된다. 맨위의 MODE 변수에 True/False값 넣어서 제어
    - **Accurcy**
        - find object(binary classification) : 
        - bbox position(regression) : 
    - added custom func to `mrcnn/visualize.py`
    - handle command line args
+ `csv_to_xml.py`
    - make an absolute coordinate for use in the bbox
    - convert .csv to .xml and create new files at each `refri_dataset/.../annots/` folder
    - `test.py`의 main에서 csv_to_xml()로 사용가능
    
###References
- [Mask RCNN Project, GitHub.](https://github.com/matterport/Mask_RCNN)
- [Mask R-CNN, 2017, Paper.](https://arxiv.org/abs/1703.06870)
- [How to Train an Object Detection Model with Keras, Blog.](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras)