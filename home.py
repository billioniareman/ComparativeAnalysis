import streamlit as st
import os
import base64
import pandas as pd

# Function to save the uploaded file
def save_uploaded_file(uploaded_file, folder_path):
    if uploaded_file is not None:
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File saved successfully at: {file_path}")
        return file_path
    else:
        st.warning("No file uploaded.")
        return None

# Function to download a file as a CSV
def download_csv_file(df, file_name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Function to display uploaded datasets on the Home page
def display_uploaded_datasets():
    st.subheader("Uploaded Datasets:")

    # Display uploaded datasets
    uploaded_files = os.listdir("uploaded_files")
    for file in uploaded_files:
        # Print dataset when clicked
        if st.button(file):
            df = pd.read_csv(os.path.join("uploaded_files", file))
            st.write(df)


def main():
    st.title("Rain Data Hub")

    # Uploading part
    upload_folder = "uploaded_files"
    os.makedirs(upload_folder, exist_ok=True)
    uploadfile = st.file_uploader('Upload your file here', type=['csv'])
    if st.button("Save Uploaded File"):
        saved_file_path = save_uploaded_file(uploadfile, upload_folder)
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        st.session_state.uploaded_files.append(saved_file_path)  # Save file path in session state

    # Display uploaded datasets
    display_uploaded_datasets()

    # Code to download files
    files_directory = r'uploaded_files'
    files_list = os.listdir(files_directory)
    st.title("Download Specific File")
    selected_file = st.selectbox("Select a file:", files_list)
    if st.button("Download Selected File"):
        file_path = os.path.join(files_directory, selected_file)
        with open(file_path, 'rb') as file:
            file_content = file.read()
            file_encoded = base64.b64encode(file_content).decode()
            st.markdown(
                f'<a href="data:file/{selected_file.split(".")[-1]};base64,{file_encoded}" download="{selected_file}">Click here to download</a>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
