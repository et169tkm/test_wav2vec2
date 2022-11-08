#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Your Name"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
import io
import speech_recognition as sr
import sys
import torch

from os import path
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Blog post explaining this: https://distill.pub/2017/ctc/
# https://huggingface.co/facebook/wav2vec2-base-960h
MODEL='facebook/wav2vec2-base-960h'
# https://huggingface.co/facebook/wav2vec2-large-960h
#MODEL='facebook/wav2vec2-large-960h'
# https://huggingface.co/speech-seq2seq/wav2vec2-2-bert-large
# MODEL='speech-seq2seq/wav2vec2-2-bert-large' # doesn't seem to work, tokens can't be translated back to text
SAMPLING_RATE=16000 # the model only support this exact sampling rate

MIN_CHUNK_LENGTH_MS=10000 # it's faster to infer on bigger chunks than a lot of smaller chunks (it might be more accurate too because the model has more context)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def join_short_chunks(in_chunks, min_length_ms):
  temp_chunks = list(in_chunks)
  i = 0
  while True:
    if i >= len(temp_chunks) - 1: # minus one because we don't need to check the last chunk
      break

    chunk = temp_chunks[i]
    if len(chunk) >= min_length_ms:
      # this chunk is long enough, go on to the next one
      i += 1
      continue
    
    next_chunk = temp_chunks.pop(i+1)
    #log("i=%d: Chunk too short (%dms), concating next chunk(%dms)" % (i, len(chunk), len(next_chunk)))
    temp_chunks[i] = chunk + next_chunk
  return temp_chunks

def transcribe(in_path, default_silence_threshold):
  tokenizer = Wav2Vec2Processor.from_pretrained(MODEL)
  model = Wav2Vec2ForCTC.from_pretrained(MODEL).half().to(DEVICE)
  log("model dtype: %s" % model.dtype)

  recognizer = sr.Recognizer()
  log("Loading file")
  audio_file = AudioSegment.from_mp3(in_path)
  log("Split file based on silence")
  chunks = split_on_silence(audio_file, min_silence_len=300, silence_thresh=-40, keep_silence=True) # keep the silence so that we can calculate the offset of a chunk base on the length of previosu chunks
  log("Chunk count: %d" % len(chunks))
  log("Join short segments")
  chunks = join_short_chunks(chunks, 10000)
  log("Chunk count: %d" % len(chunks))

  offset_ms=0.0
  chunk_index = 0
  output = []
  for chunk in chunks:
    log("processing chunk %d/%d" % (chunk_index, len(chunks)))
    chunk_index += 1
    #chunk.export("temp/%04d.mp3" % i, format="mp3")

    # resample
    log("resampling")
    clip = chunk.set_frame_rate(SAMPLING_RATE) # numpy array
    # convert to tensor
    log("conveting to FloatTensor")
    x = torch.HalfTensor(clip.get_array_of_samples())
    log("float tensor:")
    log(x.shape)

    log("tokenizing")
    inputs = tokenizer(x, sampling_rate=SAMPLING_RATE, return_tensors='pt', padding='longest').input_values
    inputs = inputs.half().to(DEVICE)
    log("tokens")
    log(inputs.shape)
    #log(inputs)
    log("getting logits")
    logits = model(inputs).logits
    log("logits:")
    log(logits.shape)
    #log(logits)
    log("argmax logits")
    tokens = torch.argmax(logits, axis=-1)
    log("tokens")
    log(tokens.shape)
    #log(tokens)
    log("decoding tokens")
    text = tokenizer.batch_decode(tokens)[0] # convert tokens to string

    print("%04d,%s,%s" % (chunk_index, format_time_from_sec(offset_ms/1000.0), text))
    offset_ms += len(chunk)

def format_time_from_sec(sec):
  h = "%02d" % int(sec // 3600)
  m = "%02d" % int((sec % 3600) // 60)
  s = "%02d" % int(sec % 60)
  ms = "%03d" % int((sec * 1000) % 1000)
  return "%s:%s:%s.%s" % (h, m, s, ms)

def log(message):
  print(message, file=sys.stderr)

def main(args):
  """ Main entry point of the app """
  log(args)
  transcribe(args.in_path, args.silence_threshold)

if __name__ == "__main__":
  """ This is executed when run from the command line """
  parser = argparse.ArgumentParser()

  # Required positional argument
  #parser.add_argument("arg", help="Required positional argument")

  # Optional argument flag which defaults to False
  #parser.add_argument("-f", "--flag", action="store_true", default=False)

  # Optional argument which requires a parameter (eg. -d test)
  #parser.add_argument("-n", "--name", action="store", dest="name")

  parser.add_argument("-i", "--in", action="store", dest="in_path")
  parser.add_argument("-s", "--silence-threshold", action="store", dest="silence_threshold", default=-40)

  # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
  parser.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=0,
    help="Verbosity (-v, -vv, etc)")

  # Specify output of "--version"
  parser.add_argument(
    "--version",
    action="version",
    version="%(prog)s (version {version})".format(version=__version__))

  args = parser.parse_args()
  main(args)

