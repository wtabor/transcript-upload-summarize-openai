import gradio as gr
import openai, config

openai.api_key = config.OPENAI_API_KEY

# Define available prompts for user to choose from
prompts = [
    "1. Create a detailed summary outline of the transcription in meeting notes organized by discussion topics and 2. List action items and key metrics or numbers discussed; use appropriate formatting for all notes such as bullet points, numbered lists, bold or italicized text to enhance the organization and readability of the notes.",
    "Can you summarize the meeting notes?",
]

MAX_TOKENS = 2048  # Set the max token limit here

def split_text(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []

    current_tokens = 0
    for word in words:
        word_tokens = len(word) + 1  # Include space before the word
        if current_tokens + word_tokens <= max_tokens:
            current_chunk.append(word)
            current_tokens += word_tokens
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def transcribe(prompt, text_file):
    # Read the content of the text_file directly
    transcript_text = text_file.read().decode("utf-8")

    transcript_chunks = split_text(transcript_text, MAX_TOKENS)
    summarized_chunks = []

    for chunk in transcript_chunks:
        prompt_text = f"{prompt}\n{chunk}\nAssistant:"

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt_text,
            max_tokens=3000,
            n=1,
            stop=None,
            temperature=0.7,
        )

        system_message = response.choices[0].text.strip()
        summarized_chunks.append(system_message)

    summarized_text = "\n\n".join(summarized_chunks)

    return summarized_text

# Set up Gradio interface
ui = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Dropdown(choices=prompts, label="Choose a prompt:"),
        gr.File(type="file", label="Upload your text file:")
    ],
    outputs="text",
    title="Investment Banker Personal Assistant",
    description="This app provides a personal assistant for an investment banker that can transcribe voice input and respond to prompts using OpenAI's powerful language model."
)

# Launch Gradio interface
ui.launch()
