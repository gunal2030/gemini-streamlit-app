Multi-modal Q&A with Gemini and Streamlit

🚀 About the Project
This project is a web application built using Streamlit that leverages the Google Gemini API to perform multi-modal question-answering. It allows users to upload an image and then ask questions about its content. The application demonstrates the power of the Gemini model to understand and reason across both visual and textual information in a single conversation. Key features include built-in rate limiting and image resizing to manage API quota usage efficiently.

🌟 Features
Multi-modal Input: Accepts both an image upload and a text prompt from the user.

Unified Analysis: The Gemini API processes the image and text together to provide a cohesive and relevant response.

Interactive UI: A clean, user-friendly interface built with Streamlit makes the application easy to use.

API Quota Management: Includes rate limiting with exponential backoff and image resizing to prevent hitting API usage limits, which is especially useful for users on a student plan.

Quick Actions: Provides one-click buttons to describe the image or extract text from it.

Conversation History: Maintains a chat history to provide context for follow-up questions.

🛠️ Technologies Used
Python: The core programming language for the application logic.

Streamlit: A powerful framework for creating interactive web applications quickly.

Google Generative AI SDK (google.generativeai): The official Python library for interacting with the Gemini API.

Pillow (PIL): Used for image processing, including resizing.

dotenv or Streamlit Secrets: For securely managing the API key.

💻 Getting Started
Prerequisites
Python 3.8 or higher

pip (Python package installer)

A Google Gemini API key
