import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from PlagiarismChecker import train_plagiarism_model

app = FastAPI()

dataset_path = 'C:\\Users\\user\\Downloads\\CMPS285-AI-Project\\fake.xlsx'  # Replace with your Excel dataset path

def load_dataset(dataset_path):
    try:
        # Load the Excel file into a DataFrame
        df = pd.read_excel(dataset_path, engine='openpyxl')
        print("Dataset loaded successfully.")

        # Check if 'text' and 'label' columns are present
        if 'text' not in df.columns or 'label' not in df.columns:
            print("Columns 'text' and 'label' are required in the dataset.")
            return [], []

        # Access columns using their names
        texts = df['text'].tolist()
        labels = df['label'].tolist()

    except FileNotFoundError:
        print(f"Error: File '{dataset_path}' not found.")
        return [], []

    except pd.errors.EmptyDataError:
        print("Error: The Excel file is empty.")
        return [], []

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], []

    return texts, labels

def train_new_model():
    # Load the dataset for each new file
    train_texts, train_labels = load_dataset(dataset_path)

    # Train a new plagiarism model for each new file
    return train_plagiarism_model(train_texts, train_labels)

@app.post("/detect")
async def detect_plagiarism(file: UploadFile = File(...), file_path: str = Form(...)):
    # Train a new model for each new file
    plagiarism_model = train_new_model()

    if plagiarism_model is None:
        return JSONResponse(content={"error": "Failed to load or train the plagiarism model."}, status_code=500)

    # Save the uploaded file temporarily
    with open("temp_file.txt", "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)

    try:
        # Predict plagiarism using the trained model
        similarity_percentage = plagiarism_model.predict(["text to compare with the uploaded document"])[0] * 100
    except Exception as e:
        # Handle any errors that might occur during prediction
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Remove the temporary file
        os.remove("temp_file.txt")

    # Return the similarity percentage in the response
    return JSONResponse(content={"similarity_percentage": similarity_percentage})
