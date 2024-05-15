# Llama2-Medical-Chatbot

## Overview

Llama2-Medical-Chatbot is a project aimed at developing a medical chatbot using the Llama 2 language model.

## Getting Started

Follow these steps to set up the project:

1. **Clone the Repository**: Create a GitHub repository and clone it locally. Then, run the `template.py` script to initialize the project structure.

2. **Create Virtual Environment**: Set up a virtual environment and install the required dependencies using `requirements.txt` and `setup.py`.

    ```bash
    conda create -n medchatbot python=3.8 -y
    conda activate medchatbot
    pip install -r requirements.txt
    ```

3. **Download Llama 2 Model**: Download the Llama 2 model (`llama-2-7b-chat.ggmlv3.q4_0.bin`) from [here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main) and place it in a directory named `Llama2-model`. Add `Llama2-model/` to `.gitignore` to exclude it from version control.

4. **Prepare Experimentation**: Set up an `experiments` directory for notebook experimentation. Push the embeddings to the Pinecone index via a notebook.

5. **Add Code Files**: Add the necessary code files to the `src` directory, including `helper.py`, `prompt.py`, `store_index.py`, and `app.py`.

6. **Add Templates**: Include the `chat.html` template file in the `templates` directory and the `style.css` file in the `static` directory.

## Usage

After completing the setup steps, you can start the Flask application by running `app.py`. Access the chatbot interface through the provided URL and interact with the medical chatbot.

## Contributors

- [Your Name](https://github.com/yourusername)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
