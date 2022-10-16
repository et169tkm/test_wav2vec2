import speech_recognition as sr
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io
from pydub import AudioSegment

from os import path

#MODEL='facebook/wav2vec2-base-960h'
MODEL='facebook/wav2vec2-large-960h'

tokenizer = Wav2Vec2Processor.from_pretrained(MODEL)
model = Wav2Vec2ForCTC.from_pretrained(MODEL)

r = sr.Recognizer()
for i in range(6):
  print("Going to process file %d" % i)
  audio_file = path.join(path.dirname(path.realpath(__file__)), "data/test%02d.wav" % (i+1))
  with sr.AudioFile(audio_file) as source:
    audio = r.record(source)

    data = io.BytesIO(audio.get_wav_data()) # list of bytes
    clip = AudioSegment.from_file(data) # numpy array
    x = torch.FloatTensor(clip.get_array_of_samples()) # tensor

    inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
    logits = model(inputs).logits
    tokens = torch.argmax(logits, axis=-1)
    text = tokenizer.batch_decode(tokens) # convert tokens to string

    print("text: %s" % text)
