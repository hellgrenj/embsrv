# embsrv  
A simple FastAPI-based server that provides various text embedding and tokenization services using the `intfloat/e5-large-v2` model from Hugging Face. This server allows you to load the model once and expose it over HTTP for multiple applications to use.

## Requirements
* Python 3.11.10 (use pyenv to install)  
* **GPU**: NVIDIA GPU with CUDA support (e.g., RTX series)  
**Note**: This server is designed to run on systems with an NVIDIA GPU. Attempting to run it on non-NVIDIA hardware (e.g., M1 Macs) will result in installation errors due to CUDA dependencies. 
(I may explore supporting other setups in the future, but for now, since this server is built to solve my own use case, an NVIDIA GPU is a requirement.)

## Development
1. Clone the repository.  
2. Create a virtual environment using `python -m venv env`.  
3. run `source env/bin/activate` to activate the virtual environment.  
4. Install the dependencies using `pip install -r requirements.txt`.  
5. Run the server using ```./run_with_reload.sh```  

## Features
- **Embeddings**: Encode text into embeddings using the `intfloat/e5-large-v2` model.
- **Tokenization**: Calculate the number of tokens in a given text.
- **Truncate**: Truncate text to a specified number of tokens.
- **Chunking**: Split text into chunks based on a maximum number of tokens per chunk.

## API Endpoints
(checkout ./demo-client.py for example usage)  
### 1. `/` or `/info` (GET)
Returns basic information about the model loaded on the server.
- **Response:**
  - `model`: The name of the model loaded (e.g., `intfloat/e5-large-v2`).
  - `embedding size`: The size of the embeddings generated (1024 dimensions).
  - `sequence limit`: The maximum number of tokens that can be processed in a single input (512 tokens).

### 2. `/embeddings/encode/` (POST)
Generates embeddings for the provided text.
- **Request Body:**
  - `text`: The input text to encode into embeddings.
- **Response:**
  - `embeddings`: A list representing the text's embedding as a 1024-dimensional vector.

### 3. `/text/truncate/` (POST)
Truncates the provided text to the specified maximum number of tokens.
- **Request Body:**
  - `text`: The input text to truncate.
  - `max_tokens`: The maximum number of tokens to retain in the truncated text.
- **Response:**
  - `truncated_text`: The truncated version of the input text.

### 4. `/text/number-of-tokens/` (POST)
Returns the number of tokens in the provided text.
- **Request Body:**
  - `text`: The input text to analyze.
- **Response:**
  - `number_of_tokens`: The total number of tokens in the input text.

### 5. `/text/chunks/` (POST)
Splits the provided text into chunks, each containing the specified maximum number of tokens.
- **Request Body:**
  - `text`: The input text to split.
  - `max_tokens`: The maximum number of tokens per chunk.
- **Response:**
  - `text_in_chunks`: A list of text chunks, each of the specified token size.

