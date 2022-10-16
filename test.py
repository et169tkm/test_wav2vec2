import speech_recognition as sr
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io
from pydub import AudioSegment
from pydub.silence import split_on_silence

from os import path

#MODEL='facebook/wav2vec2-base-960h'
MODEL='facebook/wav2vec2-large-960h'

tokenizer = Wav2Vec2Processor.from_pretrained(MODEL)
model = Wav2Vec2ForCTC.from_pretrained(MODEL)

r = sr.Recognizer()
print("Loading file")
audio_file = AudioSegment.from_wav("data/test_long.44100.wav")
chunks = split_on_silence(audio_file, min_silence_len=300, silence_thresh=-40, keep_silence=True) # keep the silence so that we can calculate the offset of a chunk base on the length of previosu chunks

offset=0.0
i = 0
for chunk in chunks:
  i += 1
  #chunk.export("temp/%04d.mp3" % i, format="mp3")

  clip = chunk.set_frame_rate(16000) # numpy array
  x = torch.FloatTensor(clip.get_array_of_samples()) # tensor

  inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
  logits = model(inputs).logits
  tokens = torch.argmax(logits, axis=-1)
  text = tokenizer.batch_decode(tokens) # convert tokens to string

  print("text[%04d, %.4f]: %s" % (i, offset/1000.0, text))
  offset += len(chunk)

#audio_file = path.join(path.dirname(path.realpath(__file__)), "data/136FinalFinal.wav")
#with sr.AudioFile(audio_file) as source:
#  while True:
#    #audio = r.record(source)
#    print("Going to listen")
#    audio = r.listen(source)
#
#    data = io.BytesIO(audio.get_wav_data()) # list of bytes
#    clip = AudioSegment.from_file(data) # numpy array
#    x = torch.FloatTensor(clip.get_array_of_samples()) # tensor
#
#    inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
#    logits = model(inputs).logits
#    tokens = torch.argmax(logits, axis=-1)
#    text = tokenizer.batch_decode(tokens) # convert tokens to string
#
#    print("text: %s" % text)
