import tkinter as tk
from tkinter import filedialog
import requests

class PlagiarismCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plagiarism Checker")

        # UI Elements
        self.upload_button = tk.Button(root, text="Upload Document", command=self.upload_file)
        self.check_button = tk.Button(root, text="Check Plagiarism", command=self.check_plagiarism)
        self.result_label = tk.Label(root, text="Plagiarism Result: ")
        self.reset_button = tk.Button(root, text="Reset", command=self.reset)

        # Layout
        self.upload_button.pack(pady=10)
        self.check_button.pack(pady=10)
        self.result_label.pack(pady=10)
        self.reset_button.pack(pady=10)

        # Initialize file_path attribute
        self.file_path = None

    def upload_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path = file_path
            # You can handle the uploaded file here (e.g., display the file path)

    def check_plagiarism(self):
        if not self.file_path:
            self.result_label.config(text="Upload a document first.")
            return

        # Send the file path to the FastAPI backend for plagiarism detection
        url = "http://127.0.0.1:8000/detect"
        files = {'file': open(self.file_path, 'rb')}
        data = {'file_path': self.file_path}
        response = requests.post(url, files=files, data=data)

        # Parse the response and update the result_label with the plagiarism result
        if response.status_code == 200:
            similarity_percentage = float(response.json()['similarity_percentage'])
            result_text = f"Plagiarism Result: {similarity_percentage:.2f}%"
            self.result_label.config(text=result_text)
        else:
            self.result_label.config(text="Plagiarism check failed")

    def reset(self):
        # Reset the state to allow uploading a new file
        self.result_label.config(text="Plagiarism Result: ")
        self.file_path = None  # Reset the file path attribute

if __name__ == "__main__":
    root = tk.Tk()
    app = PlagiarismCheckerApp(root)
    root.mainloop()
