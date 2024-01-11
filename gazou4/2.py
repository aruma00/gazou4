import cv2
from matplotlib import pyplot as plt

path = 'lena.jpg'   # 画像のパス
i = cv2.imread(path, 0)  

th = 127     # 閾値
i_max = 255  # 最大輝度値
ret, i_binary = cv2.threshold(i, th, i_max, cv2.THRESH_BINARY)   # 二値化処理

path_o = 'lena_binary.jpg'       # 出力画像のパス
cv2.imwrite(path_o, i_binary)    # 画像保存

#ここからグラフ設定
fig = plt.figure()
ax1 = fig.add_subplot(111)
 
# 画像をプロット
ax1.imshow(i_binary, cmap = 'gray')
 
# 軸を消す設定
ax1.tick_params(labelbottom = False, bottom = False)
ax1.tick_params(labelleft = False, left = False)
 
fig.tight_layout()
plt.show()
plt.close()