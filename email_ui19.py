import streamlit as st
import os
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import (DataCollatorWithPadding,create_optimizer,AutoTokenizer,DataCollatorForSeq2Seq,
                          T5TokenizerFast,T5ForConditionalGeneration,TFAutoModelForSeq2SeqLM)

ver = 1
output_directory="/content/drive/MyDrive/projects/AI-based-Generative-QA-System/Model_files"
model_path = os.path.join(output_directory,f"tuned_model_{ver}" )

tokenizer4local = AutoTokenizer.from_pretrained(model_path)
model4local = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def generate_prompt(email_content, llm, tokenizer,device="cpu:0"):
    #print('generate_prompt new method called')
    # Move the model to the specified device
    llm = llm.to(device)

    #Send a clear and concise subject line for the following email.

    # Construct the input prompt
    input_prompt = f"""
                    Generate subject for below email:

                    {email_content}

                    Subject:
                    """

    # Tokenize the input prompt
    input_ids = tokenizer(input_prompt, return_tensors='pt', max_length=512)

    # Move the input tensor to the specified device
    input_ids = {key: value.to(device) for key, value in input_ids.items()}

    # Generate output on the same device
    tokenized_output = llm.generate(input_ids['input_ids'], min_length=10, max_length=128)

    # Move the output tensor back to CPU if needed
    tokenized_output = tokenized_output.to("cpu")

    # Decode the generated output
    output = tokenizer.decode(tokenized_output[0], skip_special_tokens=True)

    return output

def get_email_subject(email_content):
    email_subject = generate_prompt(email_content, model4local, tokenizer4local)
    return email_subject

# Function to be called on generate subject
def get_hello_world(user_input):
    return f"Email Subject, User Input: {user_input}"

# Streamlit app
def main():
    # Set the page title and a heading
    st.markdown("<h1 style='font-size:32px; color:#33FFF6; text-align:center'>Email Subject Line Generator - Team 19</h1>", unsafe_allow_html=True)

    # Display a text area for user input with increased height and width
    st.markdown("<label for='font-size:24px;font-weight:bold;'>Enter your email body:</label>", unsafe_allow_html=True)
    user_input = st.text_area("", key='user_input', height=180)

    # Submit button
    if st.button("Generate subject"):
        # Call the function and get the result
        result = get_email_subject(user_input)

        # Display the result in a separate text box with a heading and adjusted font size
        st.markdown("<label for='font-size:24px; font-weight:bold;'>Generated Subject:</label>", unsafe_allow_html=True)
        #st.text(result)
        st.text_area("", value=result, height=25, key='generated_subject', disabled=True)

if __name__ == "__main__":
    main()