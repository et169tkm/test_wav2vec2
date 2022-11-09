#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Your Name"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
import GPUtil
import io
import speech_recognition as sr
import sys
import time
import tqdm
import torch

from os import path
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Blog post explaining this: https://distill.pub/2017/ctc/
# https://huggingface.co/facebook/wav2vec2-base-960h
# MODEL='facebook/wav2vec2-base-960h'
# https://huggingface.co/facebook/wav2vec2-large-960h
# MODEL='facebook/wav2vec2-large-960h'
# https://huggingface.co/speech-seq2seq/wav2vec2-2-bert-large
# MODEL='speech-seq2seq/wav2vec2-2-bert-large' # doesn't seem to work, tokens can't be translated back to text
MODEL='jonatasgrosman/wav2vec2-large-xlsr-53-english'
SAMPLING_RATE=16000 # the model only support this exact sampling rate

MIN_CHUNK_LENGTH_MS=10000 # it's faster to infer on bigger chunks than a lot of smaller chunks (it might be more accurate too because the model has more context)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.HalfTensor
SHOULD_SHOW_GPU = False

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
  log("Loading tokenizer")
  tokenizer = Wav2Vec2Processor.from_pretrained(MODEL)
  log("Loading model")
  model = Wav2Vec2ForCTC.from_pretrained(MODEL).type(MODEL_DTYPE).to(DEVICE)
  log("model dtype: %s" % model.dtype)
  log_gpu()

  #recognizer = sr.Recognizer()
  log("Loading file")
  audio_file = AudioSegment.from_mp3(in_path)
  log("input file -       length: %ds, frame rate: %d, sample_width: %dbits, channels: %d" % (len(audio_file)//1000, audio_file.frame_rate, audio_file.sample_width * 8, audio_file.channels))
  audio_file = audio_file.set_channels(1).set_frame_rate(SAMPLING_RATE)
  log("after resampling - length: %ds, frame rate: %d, sample_width: %dbits, channels: %d" % (len(audio_file)//1000, audio_file.frame_rate, audio_file.sample_width * 8, audio_file.channels))
  log("Split file based on silence")
  chunks = split_on_silence(audio_file, min_silence_len=40, silence_thresh=-40, keep_silence=True) # keep the silence so that we can calculate the offset of a chunk base on the length of previosu chunks
  log("Chunk count: %d" % len(chunks))
  log("Join short segments")
  chunks = join_short_chunks(chunks, 15000)
  log("Chunk count: %d" % len(chunks))

  offset_ms=0.0
  chunk_index = 0
  output = []
  for chunk_index in tqdm.tqdm(range(len(chunks))):
    chunk = chunks[chunk_index]
    log("processing chunk %d/%d, offset(ms): %.02f" % (chunk_index, len(chunks), offset_ms))
    #chunk.export("temp/%04d.mp3" % i, format="mp3")

    # resample
    log("resampling")
    clip = chunk # numpy array
    # convert to tensor
    log("conveting to Tensor")
    x = MODEL_DTYPE(clip.get_array_of_samples())
    log("sample tensor (%s) :(%s)" % (x.dtype, list(x.shape)))

    log("tokenizing")
    inputs = tokenizer(x, sampling_rate=SAMPLING_RATE, return_tensors='pt', padding='longest').input_values
    inputs = inputs.type(MODEL_DTYPE).to(DEVICE)
    log("input (%s) length: %d, duration: %f" % (inputs.dtype, inputs.shape[1], len(chunk)))
    log_gpu()

    log("getting logits")
    logits = model(inputs).logits
    log_gpu()
    del inputs
    torch.cuda.empty_cache()
    log("logits (%s, %s): %s" % (logits.dtype, logits.get_device(), list(logits.shape)))
    log_gpu()
    log("argmax logits")
    tokens = torch.argmax(logits, axis=-1)
    log_gpu()
    del logits
    torch.cuda.empty_cache()
    log("tokens (%s, %s): %s" % (tokens.dtype, tokens.get_device(),  list(tokens.shape)))
    log_gpu()
    log("decoding tokens")
    text = tokenizer.batch_decode(tokens)[0] # convert tokens to string
    del tokens 
    torch.cuda.empty_cache()


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

def log_gpu():
  if SHOULD_SHOW_GPU:
    GPUtil.showUtilization()

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

