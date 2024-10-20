import requests
import numpy as np

BASE_URL = "http://localhost:9000"


def get_model_info():
    url = f"{BASE_URL}/info"
    response = requests.get(url)
    if response.status_code == 200:
        resp = response.json()
        return resp
    else:
        print(f"Error: {response.status_code}")
        return None


def get_number_of_tokens(text):
    url = f"{BASE_URL}/text/number-of-tokens/"
    payload = {"text": text}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        resp = response.json()
        number_of_tokens = resp["number_of_tokens"]
        return number_of_tokens
    else:
        print(f"Error: {response.status_code}")
        return None


def get_embedding(text):
    url = f"{BASE_URL}/embeddings/encode/"
    payload = {"text": text}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        resp = response.json()
        embedding_list = resp["embeddings"]
        embedding_array = np.array(embedding_list)
        return embedding_array
    else:
        print(f"Error: {response.status_code}")
        return None


def truncate_text_to_max_tokens(text: str, max_tokens: int):
    url = f"{BASE_URL}/text/truncate/"
    payload = {"text": text, "max_tokens": max_tokens}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        resp = response.json()
        print(resp)
        truncated_text = resp["truncated_text"]
        return truncated_text
    else:
        print(f"Error: {response.status_code}")
        return None


def text_to_token_sized_chunks(text: str, number_of_tokens: int) -> list:
    url = f"{BASE_URL}/text/chunks/"
    payload = {"text": text, "max_tokens": number_of_tokens}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        resp = response.json()
        print(resp)
        list_of_chunks = resp["text_in_chunks"]
        return list_of_chunks
    else:
        print(f"Error: {response.status_code}")
        return None


if __name__ == "__main__":
    print(get_model_info())
    text = "Hello, how are you?"
    number_of_tokens = get_number_of_tokens(text)
    print("number of tokens for text:", number_of_tokens)
    embeddings = get_embedding(text)
    if embeddings is not None:
        print("Embedding array shape:", embeddings.shape)
        print("Embeddings:", embeddings)

    truncated_text = truncate_text_to_max_tokens(text, max_tokens=2)
    if truncated_text is not None:
        print("Truncated text to 2 tokens:", truncated_text)

    text_in_chunks = text_to_token_sized_chunks(text, 2)
    print(f"got {len(text_in_chunks)} back")
    for t in text_in_chunks:
        print(t)
