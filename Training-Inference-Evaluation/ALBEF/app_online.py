# Author 
# Rahul Kumar (Northeastern University)

import gradio as gr
import requests
import json
import pandas as pd
from tqdm import trange
import torch
import os
import cv2
from ALBEF import CXR_ReDonE_pipeline1


# Function to get completion from local Ollama instance
def get_ollama_completion(prompt, model="mistral"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    return json.loads(response.text)['response']

# Function to generate a report using retrieved documents
def generate_report(retrieved_documents, model="mistral"):
    # Combine retrieved documents into a single context
    context = "\n\n".join(retrieved_documents)

    # Define system and user prompts
    system_prompt = "You are an assistant designed to generate radiology reports based on provided context and query."

    user_prompt = f'''Based on the following context and query, generate a detailed report:

    Context:
    {context}

    Query: 
    {"""Please generate a concise radiology report with only impression section using information only from retrieved context. 
        ##
        Follow these guidelines to generate the report:
        1) Do not provide report by bullet points or numbers.
        2) Do not write the word impression, provide only the report. 
        3) Do not mention any comparison with prior or earlier reports.
        4) Always give a single line report and dont introduce new line characters.
        5) Do not provide any special characters.
        ##

        ** 
        Stable appearance of lower cervical fusion Heart size normal. No pneumothorax or pleural effusion. No focal airspace disease. Calcified nodules consistent with chronic granulomatous disease. Bony structures appear intact.
        **
    """}
    '''

    # Combine prompts and generate the report
    full_prompt = f"System: {system_prompt}\n\nHuman: {user_prompt}\n\nAssistant:"
    
    report = get_ollama_completion(full_prompt, model)
    
    return report.strip()

def inference(ChestImg):
    file_path = 'inference_images'
    os.makedirs(file_path, exist_ok=True)
    cv2.imwrite(os.path.join(file_path, "Input_image.png"), ChestImg)
    CXR_ReDonE_pipeline1.main()
    prediction = pd.read_csv('ALBEF/top_k_pred.csv')
    reports = prediction['findings']
    reports = reports.to_string(index=False)
    print(reports)
    split_reports = reports.split('|')
    return generate_report(split_reports)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Chest X-Ray",height=400,width=600)
            generate_btn = gr.Button("Generate Report")
        with gr.Column():
            report = gr.Textbox(label="Report")
    generate_btn.click(fn=inference,inputs=input_img, outputs=report)
    examples = gr.Examples(examples=['../../images/9_IM-2407-1001.dcm.png','../../images/1000_IM-0003-1001.dcm.png','../../images/1001_IM-0004-1002.dcm.png'],
            inputs=[input_img],label="Sample X-Ray scan")

demo.launch(share=True)