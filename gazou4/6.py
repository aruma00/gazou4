import numpy as np
import cv2
from matplotlib import pyplot as plt

# 画像から輪郭を検出する関数
def contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = np.vstack(contours)  # 輪郭情報をndarrayに変換
    
    x = np.mean(contours[:, 0, 0])  # 輪郭のx方向平均値を算出
    y = np.mean(contours[:, 0, 1])  # 輪郭のy方向平均値を算出
    return x, y

# トラッカーの初期化
tracker = cv2.TrackerKCF_create()

# 動画ファイルの読み込み
movie = cv2.VideoCapture('testmovie1.avi')

# フレームを1枚ずつ取得して動画処理後に保存する
x_list = []
y_list = []

# 最初のフレームを読み込む
ret, frame = movie.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# ユーザーによって選択された初期のバウンディングボックス（静止している物体）を取得
bbox = cv2.selectROI(frame, False)

# トラッカーに初期フレームとバウンディングボックスを渡す
tracker.init(frame, bbox)

while True:
    ret, frame = movie.read()

    if not ret:
        break

    # トラッキングを更新
    ret, bbox = tracker.update(frame)

    # BoundingBoxの座標を整数に変換
    bbox = tuple(map(int, bbox))

    # トラッキング結果を描画
    if ret:
        cv2.rectangle(frame, bbox, (0, 255, 0), 3, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    x, y = contours(frame)
    x_list.append(x)
    y_list.append(y)

    cv2.imshow("Tracking", frame)

    # 'q'キーを押したらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 動画オブジェクト解放
movie.release()
cv2.destroyAllWindows()
