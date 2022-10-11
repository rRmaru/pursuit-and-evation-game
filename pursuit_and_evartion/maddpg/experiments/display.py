#%%
### インポート
import tkinter
#%%
### 定数
SIZE = 20       # 円サイズ
 
### 変数
x_pos = 0       # X座標
y_pos = 190     # Y座標

### キャンバス作成
canvas = tkinter.Canvas(width=640, height=400)
#%%
 
### キャンバス表示
canvas.pack()
 
### 処理実行関数
def run():
 
    ### グローバル変数宣言
    global x_pos
    global y_pos
 
    ### キャンバスクリア
    canvas.delete("all")
 
    ### 円表示
    canvas.create_oval(x_pos, y_pos, x_pos+SIZE, y_pos+SIZE, outline="red", fill="red")
 
    ### 座標移動
    x_pos += 3
 
    ### 待機
    canvas.after(50, run)
 
### 処理実行
run()
 
### イベントループ
canvas.mainloop()
# %%
