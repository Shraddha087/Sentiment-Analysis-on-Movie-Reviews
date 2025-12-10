import tkinter as tk
from tkinter import messagebox, filedialog
import joblib
import os


class SentimentGUI:
    def __init__(self, master):
        self.master = master
        master.title("Sentiment Analysis — GUI")

        self.model_path = 'model/sentiment_model.pkl'
        self.model = None

        # Top frame: load model
        top = tk.Frame(master)
        top.pack(padx=10, pady=6, fill='x')

        self.model_label = tk.Label(top, text=f"Model: {self.model_path}")
        self.model_label.pack(side='left')

        load_btn = tk.Button(top, text="Load Model", command=self.load_model)
        load_btn.pack(side='right')

        browse_btn = tk.Button(top, text="Browse...", command=self.browse_model)
        browse_btn.pack(side='right', padx=(0,6))

        # Text input
        tk.Label(master, text="Enter review text:").pack(anchor='w', padx=10)
        self.text = tk.Text(master, height=8, width=80)
        self.text.pack(padx=10, pady=(0,8))

        # Predict button and result
        bottom = tk.Frame(master)
        bottom.pack(padx=10, pady=6, fill='x')

        predict_btn = tk.Button(bottom, text="Predict Sentiment", command=self.predict)
        predict_btn.pack(side='left')

        self.result_var = tk.StringVar(value="Result: —")
        result_label = tk.Label(bottom, textvariable=self.result_var, font=(None, 12, 'bold'))
        result_label.pack(side='left', padx=12)

        # Try to auto-load model if available
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.model_label.config(text=f"Model: {self.model_path} (loaded)")
            except Exception:
                pass

    def browse_model(self):
        path = filedialog.askopenfilename(title='Select model file', filetypes=[('PKL files','*.pkl'),('All files','*.*')])
        if path:
            self.model_path = path
            self.model_label.config(text=f"Model: {self.model_path}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror("Model not found", f"Model file not found:\n{self.model_path}")
            return
        try:
            self.model = joblib.load(self.model_path)
            self.model_label.config(text=f"Model: {self.model_path} (loaded)")
            messagebox.showinfo("Loaded", "Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load model:\n{e}")

    def predict(self):
        text = self.text.get('1.0', 'end').strip()
        if not text:
            messagebox.showwarning("Input needed", "Please enter a review to predict.")
            return
        if self.model is None:
            self.load_model()
            if self.model is None:
                return

        try:
            pred = self.model.predict([text])
            self.result_var.set(f"Result: {pred[0]}")
        except Exception as e:
            messagebox.showerror("Prediction error", f"Failed to predict:\n{e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = SentimentGUI(root)
    root.mainloop()
