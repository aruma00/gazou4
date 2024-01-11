import cv2
 
# 動画読み込みの設定
movie = cv2.VideoCapture('testmovie1.avi')
 
# 動画ファイル保存用の設定
fps = int(movie.get(cv2.CAP_PROP_FPS))                         # 動画のFPSを取得
w = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))                   # 動画の横幅を取得
h = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))                  # 動画の縦幅を取得
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')            # 動画保存時のfourcc設定（mp4用）
video = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h), False)  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）
 
# 背景差分の設定
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()              # 背景オブジェクトを作成
 
# ファイルからフレームを1枚ずつ取得して動画処理後に保存する
while True:
    ret, frame = movie.read()                                  # フレームを取得
    fgmask = fgbg.apply(frame)                                 # 前景領域のマスクを取得する
    video.write(fgmask)                                        # 動画を保存する
 
    # フレームが取得できない場合はループを抜ける
    if not ret:
        break
 
# 撮影用オブジェクトとウィンドウの解放
movie.release()