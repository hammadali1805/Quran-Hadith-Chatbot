# AI-Powered Islamic Query Chatbot

## Introduction

The **AI-Powered Islamic Query Chatbot** is an AI-driven solution designed to help users obtain quick and accurate answers to their Islamic-related queries. The chatbot pulls relevant information from the Quran and Hadiths using a combination of advanced natural language processing techniques and machine learning models. This project was born out of the need to simplify the process of finding specific information in Islamic texts, which can often be time-consuming and challenging.

## Features

- **Query Expansion**: Uses Gemini AI to expand user queries, improving the accuracy and relevance of the retrieved answers.
- **Similarity Search**: Performs similarity search on a vector store of Quran and Hadiths to fetch the most relevant information.
- **Answer Generation**: Leverages advanced AI models to generate detailed and contextually relevant answers, referencing Quran and Hadiths.
- **User-Friendly Interface**: Built using Streamlit, providing a clean and intuitive user interface.
- **Customizable API Key Input**: Users can input their Google Gemini API key securely, which is used to run the model.

## Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/hammadali1805/Quran-Hadith-Chatbot.git
   ```

2. **Install the required Python packages**:
   ```
   pip install -r requirements.txt
   ```

3. **Obtain your Google Gemini API key**:
   Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to get a free API key.

4. **Run the chatbot application**:
   ```
   streamlit run app.py
   ```

## How It Works

1. **Vector Store**: A vector store of Quran and Hadiths (Sahih al-Bukhari and Sahih al-Muslim) is created using Chroma DB and embeddings from the Hugging Face model `all-MiniLM-L6-v2`.
2. **Query Expansion**: The chatbot uses Gemini AI to expand user queries, ensuring comprehensive retrieval of results from the vector store.
3. **Similarity Search**: The expanded query is passed through the vector store, and relevant content is retrieved based on similarity scores.
4. **Answer Generation**: The context retrieved from the vector store is processed by the model to generate accurate, detailed answers with references to Quran and Hadiths.

## Usage

1. **Input Your API Key**:
   - The first time you load the app, you will be prompted to input your Google Gemini API key. 
   - Your API key will be stored temporarily for the session and will be deleted once the session ends or the page is reloaded.

2. **Ask a Question**:
   - Type your question in the chatbot interface.
   - The chatbot will search the Quran and Hadiths, expand your query if needed, and return the most relevant answer.

3. **View Results**:
   - The chatbot will display the answer, including references to the source texts where applicable.

## Example Queries

- *What does the Quran say about patience?*
- *Tell me about the hadith on honesty.*
- *What are the conditions of fasting in Islam?*

## Skills and Technologies Used

- **Natural Language Processing (NLP)**: Applied techniques to expand and process queries.
- **Vector Store and Embeddings**: Leveraged embeddings from Hugging Face and Chroma DB for similarity search.
- **Gemini AI**: Used for query expansion and answer generation.
- **Streamlit**: Designed a clean and intuitive user interface for easy interaction.
- **API Integration**: Integrated Google Gemini API for accessing advanced AI models.

## Contributors

- [Hammad Ali](https://github.com/hammadali1805) - Project Lead and Developer

## Acknowledgments

- Thanks to Google AI and Hugging Face for the powerful tools and models that made this project possible.

## Support

If you encounter any issues or have questions about the chatbot, feel free to [open an issue](https://github.com/hammadali1805/Quran-Hadith-Chatbot/issues) on the GitHub repository.
