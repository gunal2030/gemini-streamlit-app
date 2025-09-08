import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import time
import random

# --- Configuration ---
# Use Streamlit's secrets management for the API key
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to configure API key. Please ensure it's set in Streamlit secrets.")
    st.stop()

# Initialize the Gemini model with better configuration for rate limits
model = genai.GenerativeModel('gemini-1.5-flash')  # Use flash model for better quota efficiency

# Set up the Streamlit app page
st.set_page_config(
    page_title="Multimodal Q&A with Gemini",
    page_icon="🤖"
)
st.title("🤖 Ask a question about your image")

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image_name" not in st.session_state:
    st.session_state.uploaded_image_name = None
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

# --- Helper Functions ---
def wait_for_rate_limit():
    """Implement basic rate limiting to avoid quota issues"""
    current_time = time.time()
    time_since_last_request = current_time - st.session_state.last_request_time
    
    # Wait at least 4 seconds between requests (15 RPM = 4 seconds between requests)
    min_wait_time = 4
    if time_since_last_request < min_wait_time:
        wait_time = min_wait_time - time_since_last_request
        with st.spinner(f"Rate limiting... waiting {wait_time:.1f} seconds"):
            time.sleep(wait_time)
    
    st.session_state.last_request_time = time.time()

def get_gemini_response(prompt_parts, max_retries=3):
    """
    Sends a prompt and image to the Gemini model and returns the response.
    Includes retry logic for quota errors and rate limiting.
    """
    for attempt in range(max_retries):
        try:
            # Implement rate limiting
            wait_for_rate_limit()
            
            start_time = time.time()
            
            # Generate content with lower temperature and max tokens for efficiency
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,  # Lower temperature for more consistent responses
                max_output_tokens=1000,  # Limit output to save quota
            )
            
            response = model.generate_content(
                prompt_parts,
                generation_config=generation_config
            )
            
            end_time = time.time()
            st.sidebar.success(f"Response Time: {end_time - start_time:.2f}s")
            return response.text
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "quota" in error_str or "rate" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(1, 3)  # Exponential backoff
                    st.warning(f"Quota limit hit. Retrying in {wait_time:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    return "❌ **Quota Exceeded**: You've hit your API limits. Please try again later or consider upgrading your plan. The student plan has limited daily requests."
            else:
                return f"❌ **API Error**: {e}"
    
    return "❌ **Failed**: Maximum retries exceeded. Please try again later."

def resize_image_for_api(image, max_size=(800, 800)):
    """Resize image to reduce token usage"""
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        st.sidebar.info(f"Image resized to {image.size} to save quota")
    return image

# --- UI Elements ---
with st.sidebar:
    st.header("Settings & Info")
    
    # Quota information
    st.subheader("📊 Quota Info")
    st.info("""
    **Student Plan Limits:**
    - ~15 requests/minute
    - Limited daily requests
    - Shared across all projects using same key
    
    **Tips:**
    - Wait between requests
    - Use smaller images
    - Ask concise questions
    """)
    
    # Model selection
    model_choice = st.selectbox(
        "Model",
        ["gemini-1.5-flash", "gemini-1.5-pro"],
        help="Flash model uses less quota"
    )
    
    if model_choice != "gemini-1.5-flash":
        model = genai.GenerativeModel(model_choice)
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Check if a new image was uploaded to clear history
    if st.session_state.uploaded_image_name != uploaded_file.name:
        st.session_state.uploaded_image_name = uploaded_file.name
        st.session_state.messages = []  # Clear history for new image

    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Resize image to save quota
    image = resize_image_for_api(image)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Add warning about quota
    if len(st.session_state.messages) > 10:
        st.warning("⚠️ You've made many requests. Consider clearing chat history to avoid quota issues.")
    
    # Add "What's in the picture?" button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 What's in the picture?"):
            st.session_state.messages.append({"role": "user", "content": "What's in the picture?"})
            with st.chat_message("user"):
                st.markdown("What's in the picture?")
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing image... Please wait"):
                    response = get_gemini_response(["Describe this image briefly in 2-3 sentences.", image])
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        if st.button("📝 Extract any text"):
            st.session_state.messages.append({"role": "user", "content": "Extract any text from this image"})
            with st.chat_message("user"):
                st.markdown("Extract any text from this image")
            
            with st.chat_message("assistant"):
                with st.spinner("Extracting text... Please wait"):
                    response = get_gemini_response(["Extract and transcribe any text visible in this image. If no text, say 'No text found'.", image])
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Show conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input for the chat
    if user_prompt := st.chat_input("Ask a question about the image..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Get assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Processing your question... Please be patient"):
                # Create a more efficient prompt
                efficient_prompt = f"Answer this question about the image in 2-3 sentences: {user_prompt}"
                response = get_gemini_response([efficient_prompt, image])
            st.markdown(response)

        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please upload an image to begin.")
