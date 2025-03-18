import streamlit as st
import os
import json
import asyncio
import re
from typing import Optional, Union, Iterable
from pyzerox import zerox
from dotenv import load_dotenv

def load_vertex_credentials(uploaded_file):
    if uploaded_file is not None:
        credentials = json.load(uploaded_file)
        return json.dumps(credentials)
    return None

def load_env_config():
    load_dotenv()

def check_provider_requirements(provider):
    """Check if all required credentials for the selected provider are available"""
    if provider == "OpenAI":
        return os.getenv("OPENAI_API_KEY") is not None, "OpenAI API key is missing"
    
    elif provider == "Azure OpenAI":
        missing = []
        if not os.getenv("AZURE_API_KEY"): missing.append("API key")
        if not os.getenv("AZURE_API_BASE"): missing.append("API base")
        if not os.getenv("AZURE_API_VERSION"): missing.append("API version")
        return not missing, f"Azure OpenAI {', '.join(missing)} missing"
    
    elif provider == "Gemini":
        return os.getenv("GEMINI_API_KEY") is not None, "Gemini API key is missing"
    
    elif provider == "Anthropic":
        return os.getenv("ANTHROPIC_API_KEY") is not None, "Anthropic API key is missing"
    
    elif provider == "Vertex AI":
        missing = []
        if not os.getenv("VERTEX_CREDENTIALS"): missing.append("credentials")
        if not os.getenv("VERTEXAI_PROJECT"): missing.append("project")
        if not os.getenv("VERTEXAI_LOCATION"): missing.append("location")
        return not missing, f"Vertex AI {', '.join(missing)} missing"
    
    return False, "Unknown provider"

def extract_content(zerox_output):
    """Extract and format content from ZeroxOutput object"""
    try:
        # Extract content from pages
        all_content = []
        for page in zerox_output.pages:
            # Extract content between content='**xxx'
            content = page.content
            # Remove content='**' prefix if it exists
            content = re.sub(r"content='?\*\*", "", content)
            # Remove trailing quotes if they exist
            content = re.sub(r"'$", "", content)
            all_content.append(content)
        
        return "\n\n".join(all_content)
    except Exception as e:
        st.error(f"Error extracting content: {str(e)}")
        return None

def format_stats(zerox_output):
    """Format processing statistics"""
    stats = {
        "Processing Time (ms)": zerox_output.completion_time,
        "Input Tokens": zerox_output.input_tokens,
        "Output Tokens": zerox_output.output_tokens,
        "Pages Processed": len(zerox_output.pages)
    }
    return stats

async def process_pdf(file_path, model, output_path, system_prompt, selected_pages, **kwargs):
    result = await zerox(
        file_path=file_path,
        model=model,
        output_dir=output_path,
        custom_system_prompt=system_prompt,
        select_pages=selected_pages,
        **kwargs
    )
    return result

def main():
    st.title("PDF to Markdown Converter")
    
    # Load environment variables at startup
    load_env_config()
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Model provider selection
    provider = st.sidebar.selectbox(
        "Select Model Provider",
        ["OpenAI", "Azure OpenAI", "Gemini", "Anthropic", "Vertex AI"]
    )
    
    # Model configuration based on provider
    kwargs = {}
    model = None
    
    if provider == "OpenAI":
        model = st.sidebar.text_input("Model Name", value="gpt-4o")
        api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    elif provider == "Azure OpenAI":
        deployment_name = st.sidebar.text_input("Deployment Name")
        model = f"azure/{deployment_name}"
        api_key = st.sidebar.text_input("Azure API Key (optional)", type="password")
        api_base = st.sidebar.text_input("Azure API Base", value="")
        api_version = st.sidebar.text_input("Azure API Version", value="2023-05-15")
        
        if api_key: os.environ["AZURE_API_KEY"] = api_key
        if api_base: os.environ["AZURE_API_BASE"] = api_base
        if api_version: os.environ["AZURE_API_VERSION"] = api_version
    
    elif provider == "Gemini":
        model = st.sidebar.text_input("Model Name", value="gemini/gemini-1.5-pro-002")
        api_key = st.sidebar.text_input("Gemini API Key (optional)", type="password")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
    
    elif provider == "Anthropic":
        model = st.sidebar.text_input("Model Name", value="claude-3-5-sonnet-20240620")
        api_key = st.sidebar.text_input("Anthropic API Key (optional)", type="password")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
    
    elif provider == "Vertex AI":
        model = f"vertex_ai/{st.sidebar.text_input('Model Name', value='gemini-1.5-flash-001')}"
        credentials_file = st.sidebar.file_uploader("Upload Vertex AI Credentials JSON (optional)")
        project = st.sidebar.text_input("Project ID (optional)", value="")
        location = st.sidebar.text_input("Location (optional)", value="")
        
        if credentials_file:
            vertex_credentials = load_vertex_credentials(credentials_file)
            kwargs["vertex_credentials"] = vertex_credentials
            os.environ["VERTEX_CREDENTIALS"] = vertex_credentials
        if project: os.environ["VERTEXAI_PROJECT"] = project
        if location: os.environ["VERTEXAI_LOCATION"] = location
    
    # Main content area
    st.header("PDF Processing")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # System prompt
    system_prompt = st.text_area("Custom System Prompt (optional)")
    if not system_prompt:
        system_prompt = None
    
    # Page selection
    page_selection = st.text_input(
        "Select Pages (leave empty for all pages, enter single number or comma-separated list)"
    )
    selected_pages = None
    if page_selection:
        try:
            if ',' in page_selection:
                selected_pages = [int(p.strip()) for p in page_selection.split(',')]
            else:
                selected_pages = int(page_selection)
        except ValueError:
            st.error("Invalid page selection format")
    
    # Output directory
    output_dir = st.text_input("Output Directory", value="./output")
    
    # Process button
    if st.button("Process PDF"):
        # Check requirements only for the selected provider
        requirements_met, error_message = check_provider_requirements(provider)
        if not requirements_met:
            st.error(f"Configuration error: {error_message}")
            return
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Process the PDF
                with st.spinner("Processing PDF..."):
                    result = asyncio.run(process_pdf(
                        file_path="temp.pdf",
                        model=model,
                        output_path=output_dir,
                        system_prompt=system_prompt,
                        selected_pages=selected_pages,
                        **kwargs
                    ))
                
                # Extract and display content
                content = extract_content(result)
                if content:
                    st.header("Preview")
                    st.markdown(content)
                    
                    # Display processing stats
                    st.header("Processing Statistics")
                    stats = format_stats(result)
                    for key, value in stats.items():
                        st.metric(key, value)
                    
                    # Download button
                    st.download_button(
                        label="Download Markdown",
                        data=content,
                        file_name="output.md",
                        mime="text/markdown"
                    )
                
                # Cleanup
                os.remove("temp.pdf")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please upload a PDF file first")

if __name__ == "__main__":
    main()