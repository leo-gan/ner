# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

RUN pip install --upgrade pip

# Install the dependencies:
## Copy requirements first to take advantage of Docker layer caching
COPY requirements.txt .

## Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Install NLTK data
RUN python -m nltk.downloader stopwords wordnet
RUN python -m nltk.downloader punkt maxent_ne_chunker words averaged_perceptron_tagger

# Download Hugging Face model and tokenizer
RUN python -c "from transformers import BertTokenizer, BertForTokenClassification; \
   BertTokenizer.from_pretrained('dslim/bert-base-NER'); \
   BertForTokenClassification.from_pretrained('dslim/bert-base-NER')"

# Download Spacy model
RUN python -m spacy download en_core_web_sm

# Expose port if needed (for future extension)
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
