import tkinter as tk
from tkinter import filedialog
import pandas as pd

def load_file():
    file_path = filedialog.askopenfilename(title="데이터 파일 선택")
    if file_path:
        dataset = pd.read_csv(file_path)
        text.insert('end', "파일을 성공적으로 불러왔습니다.\n")
        text.insert('end', str(dataset.head()) + "\n")  # dataset의 첫 몇 행을 출력
    else:
        text.insert('end', "파일 선택이 취소되었거나 잘못된 파일을 선택하셨습니다.\n")

root = tk.Tk()
root.geometry("600x400")

load_button = tk.Button(root, text="데이터 파일 불러오기", command=load_file)
load_button.pack()

text = tk.Text(root)
text.pack()

root.mainloop()
