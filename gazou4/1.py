import cv2

#入力画像
image = cv2.imread('apple.jpeg')

#画像のサイズ縮小
height = image.shape[0]
width = image.shape[1]
image = cv2.resize(image,(round(width/4), round(height/4)))

image_copy1 = image.copy()
#グレースケール化
image_copy1 = cv2.cvtColor(image_copy1,cv2.COLOR_BGR2GRAY)

#閾値処理
ret,thresh = cv2.threshold(image_copy1,95,255,cv2.THRESH_BINARY)
#輪郭検出 （cv2.ChAIN_APPROX_SIMPLE）
contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#輪郭の描画
cv2.drawContours(image, contours1, -1, (0, 255, 0), 2, cv2.LINE_AA)

#実行結果
cv2.imshow('Drawn contours', image)
cv2.imshow('Original', image_copy1)
cv2.waitKey(0)
cv2.destroyAllWindows()