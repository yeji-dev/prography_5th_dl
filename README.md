# prography_5th_dl
프로그라피 5기 지원자 홍예지
###Summary of assiginment
<ul>
    <li>Mission : 냉장고 속 물체 탐지(Object Detection)</li>
    <li>Framework : Keras</li>
    <li>Detection Network : <a href="https://github.com/matterport/Mask_RCNN">Mask RCNN</a> (Two-Stage Method)</li>
    <li>Structure
        <ol>
        <li>Multi Classification Model (CNN)</li>
        <li>Binary Classification Model (이미지에 객체있는지 분류)</li>
        <li>Regression Model (박스 그려주는 모델)</li>
        </ol>
    </li>
</ul>

###Result
입력

    python test.py --model [model name] --image [input image]

수행 결과
    
    어쩌구저쩌구

<ul>
    <li>Accuracy : </li>
</ul>

<hr>

###Description of Code
<ul>model.h5
    <ul>d</ul>
    <ul>d</ul>
</ul>
<ul>csv_to_xml.py
    <ul>Transfer csv to xml file</ul>
    <ul>bbox에 사용할 절대좌표로 만들어 줌 (w*상대값, h*상대값)</ul>
</ul>