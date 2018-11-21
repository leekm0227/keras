import sys
import cv2 
import numpy as np

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
def test():
    for i in range(14):
        # 이미지 읽어 들이기
        im = cv2.imread('./img/tmp/img' + str(i) + '.png')

        # 그레이스케일로 변환하고 블러를 걸고 이진화하기
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

        # 윤곽 추출하기
        # 두번째 매개변수를 cv2.RETR_LIST로 지정하면 모든 구간의 외곽을 검출합니다.
        # 두번째 매개변수를 cv2.RETR_EXTERNAL로 지정하면 영역의 가장 외곽 부분만 검출합니다.
        contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

        # 추출한 윤곽을 반복 처리하기
        for j, cnt in enumerate(contours):
          x, y, w, h = cv2.boundingRect(cnt)

          # 너무 작으면 건너뛰기
          if h < 20 or w < 25: continue 

          #잘라내어 저장
          cv2.imwrite('./img/sample1/img_' + str(i) + '_' + str(j) + '.png', im[y:y + h, x:x + w])
    
    print("-----complete-----")
    quit()
    
    #cv2.imwrite('result2.png', im)
    #plt.imshow(im); plt.show()
    #image = mpimg.imread("result.png")
    #plt.imshow(image); plt.show()

'''
#빨간상자처리
red = (0, 0, 255)
cv2.rectangle(im, (x, y), (x+w, y+h), red, 2)
'''

'''
def im_trim(img):
    x = 845; y = 325; #자르고 싶은 지점의 x좌표와 y좌표 지정 
    w = 180; h = 235; #x로부터 width, y로부터 height를 지정 
    img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다 
    cv2.imwrite('org_trim.jpg',img_trim) #org_trim.jpg 라는 이름으로 저장 
    return img_trim #필요에 따라 결과물을 리턴 

org_image = cv2.imread('test.jpg') #test.jpg 라는 파일을 읽어온다 
trim_image = im_trim(org_image) #trim_image 변수에 결과물을 넣는다
'''