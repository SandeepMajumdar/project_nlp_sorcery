from transformers import pipeline
import gradio as gr

# Sentiment analysis interface
sa_examples = [
    'the food delivered was stale',
    'i like your shirt',
    'this is not the way to work',
]

sa_app = gr.Interface.load(
    'huggingface/distilbert-base-uncased-finetuned-sst-2-english',
    title='Sentiment Analysis',
    examples=sa_examples,
    description='Type your sentence here and Submit',
)

# Text Generation
tg_examples = [
    'I want to feel',
    'It is possible to',
    'When the world seems'
]

tg_app = gr.Interface.load(
    'huggingface/distilgpt2',
    title='Text Generation',
    examples=tg_examples,
    description="Write an incomplete sentence and submit",
)

# Fill Mask
fm_examples = [
    'Do you know how much I <mask> you?',
    'When we went to the forest, it <mask> raining',
]

fm_app = gr.Interface.load(
    'huggingface/distilroberta-base',
    title='Fill In The Blank',
    examples=fm_examples,
    description="Write a sentence with a missing word using \<mask\>",
)

# Named Entity Recognition
ner_examples = [
    'My name is Doug and I live in Delhi',
    'Vishal works at Google',
]

ner_app = gr.Interface.load(
    'huggingface/dbmdz/bert-large-cased-finetuned-conll03-english',
    title='Named Entity Recognition',
    examples=ner_examples,
    description="Write a sentence with a name, place, organization, etc and I'll try to reveal them",
)

# Summarization
sum_examples = [
    '''Television is one of the many wonders of modern science and technology. It was invented in England by the Scottish scientist J.N. Baird 
in 1928 and the British Broadcasting Corporation was the first to broadcast television images in 1929. Previously the radio helped us 
hear things from far and near. spread information and knowledge from one corner of the globe to another. But all this was done through 
sound only. But television combined visual images with sound. Today we can watch games, shows, and song and dance programs from all 
corners of the world while sitting at our own homes. TV can be used for educating the masses, for bringing to us the latest pieces of 
information audio-visually and can provide us all kinds of entertainment even in color. But as in all things, too much televiewing may 
prove harmful. In many cases, the habit of watching TV has an adverse effect on the study habits of the young. When we read books, we 
have to use our intelligence and imagination. But in most cases, TV watching is a passive thing. It may dull our imagination and 
intelligence.''',
]

sum_app = gr.Interface.load(
    'huggingface/sshleifer/distilbart-cnn-12-6',
    title='Text Summarization',
    examples=sum_examples,
    description="Copy and dump a long paragraph here for summarization, or click the example below",
)

# Translation to Hindi
trans_examples = [
    'I want to go home',
    "i will go to the station tomorrow",
]

trans_app = gr.Interface.load(
    'huggingface/Helsinki-NLP/opus-mt-en-hi',
    title='Translate From English to Hindi',
    examples=trans_examples,
    description="Write a sentence to translate from English to Hindi",
)

# Text to Speech
tts_examples = [
    "How do you do?",
    'i thought we were supposed to go to the park'
]

tts_app = gr.Interface.load(
    "huggingface/facebook/fastspeech2-en-ljspeech",
    title='Text to Speech',
    examples=tts_examples,
    description="Give me something to say!",
)

# Speech to Text
stt_app = gr.Interface.load(
    "huggingface/facebook/wav2vec2-base-960h",
    title='Speech to Text',
    inputs="mic",
    description="Let me try to guess what you're saying! Stop the recording button before submitting.",
)

with gr.Blocks() as demo:
  gr.Markdown("# App For Various NLP Tasks (use landscape on phone)")
  gr.TabbedInterface([sa_app, tg_app, fm_app, ner_app, sum_app, trans_app, tts_app, stt_app], 
                          ["Sentiment Analysis", "Text Generation", "Fill Blank", "Named Entity", "Summary", 
                           "Translation", "Text to Speech", "Speech to Text"]
                          )

if __name__ == "__main__":
    demo.launch()