import ffmpeg
import io
import tempfile
import whisper
from TTS.utils.radam import RAdam
import torch
#torch.serialization.add_safe_globals({RAdam.__module__ + '.' + RAdam.__name__: RAdam})
from TTS.api import TTS
import os
from num2words import num2words
import re
from moviepy.editor import VideoFileClip, AudioFileClip
#from google.colab import drive


#drive.mount('/content/drive')
#filename = 'BE1_data.mov'
#path = '/content/drive/MyDrive/TranslatorAPP/engineering/data'
#filepath = os.path.join(path, filename)

class translate_batch_movie:
  def __init__(self, file_path, output_path = None, source_language='en', target_language='es'):
    self.file_path = file_path
    self.output_path = output_path
    self.source_language = source_language
    self.target_language = target_language

  def convert_numbers_to_words(self, text):
    return re.sub(r'\d+', lambda x: num2words(int(x.group()), lang=self.target_language), text)

  def transcribe_video(self,video_path: str, model_name: str = "base") -> str:
    # 1) Use ffmpeg-python to extract audio as raw WAV in-memory
    out, _ = (
        ffmpeg
        .input(video_path)
        # -ar 16000: resample to 16 kHz
        # -ac 1: mono
        # -f wav -acodec pcm_s16le: 16-bit PCM WAV
        .output('pipe:', format='wav', acodec='pcm_s16le', ac=1, ar='16000')
        .run(capture_stdout=True, capture_stderr=True)
    )

    # 2) Wrap bytes in a buffer
    audio_buffer = io.BytesIO(out)

    # 3) Whisper still wants a filename, so write to a NamedTemporaryFile
    #    (this file is deleted as soon as the context exits)
    model = whisper.load_model(model_name)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_buffer.read())
        tmp.flush()
        # 4) Transcribe, forcing Spanish
        result = model.transcribe(tmp.name, language=self.target_language)

    return self.convert_numbers_to_words(result["text"])

  def save_audio(self):
    transcription = transcribe_video_to_spanish(self.file_path, model_name="small")
    print(f"Transcription in target language {self.target_language}:")
    print(transcription)

    wav_file = "x_transcribed.wav"
    if self.output_path is None:
      self.output_audio_filepath = os.path.splitext(filepath)[0] + wav_file
    else:
      self.output_audio_filepath = self.output_path+wav_file
    tts = TTS(model_name="tts_models/es/css10/vits", progress_bar=False, gpu=False)#
    tts.tts_to_file(text=transcription, file_path=self.output_audio_filepath)

  def replace_movie_audio(self):
    # Paths
    video_path = self.file_path
    new_audio_path = self.output_audio_filepath
    if self.output_path is None:
      output_path = self.file_path.replace(".mov", "_translatedXyX.mp4")

    # Load video and new audio
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(new_audio_path)
    # Set new audio
    video_with_new_audio = video.set_audio(new_audio)
    # Export final video
    video_with_new_audio.write_videofile(output_path, codec='libx264', audio_codec='aac')

  def run(self):
    #transcription = transcribe_video_to_spanish(self.filepath, model_name="small")
    self.save_audio()
    self.replace_movie_audio()
