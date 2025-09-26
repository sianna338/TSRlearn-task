# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 10:42:01 2025

@author: Sianna.Groesser
"""
#conda install -c conda-forge ffmpeg
from pydub import AudioSegment
import os
import io
import pandas as pd
from tqdm import tqdm
import google.cloud
from scipy.io import wavfile
from pydub import AudioSegment
from functools import cache
from google.cloud import texttospeech


# file needs to be present. JSON file with google credentials.
# see https://developers.google.com/workspace/guides/create-credentials
# needs to have the Google-TTS API service active as well.
# then set the following env var with the file path
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/zi/home/simon.kern/Nextcloud/ZI/2020.1 Pilotstudie//google-cloud-creds.json'

def save_tts_google(word, filename, language_code="de-DE", target_length=1):
    tts_client = texttospeech.TextToSpeechClient()

    # First generate with default speaking rate to get baseline duration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        effects_profile_id=["headphone-class-device"],
        pitch=0,
        speaking_rate=1.0
    )

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender ='MALE',
        # name=f"{language_code}-Neural2-G"
    )

    synthesis_input = texttospeech.SynthesisInput(text=word)

    # Generate initial sample
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Measure initial duration
    with io.BytesIO(response.audio_content) as f:
        # if this fails, you need to have FFMPEG installed
        # conda install -c conda-forge ffmpeg
        audio = AudioSegment.from_mp3(f)
    initial_duration = len(audio) / 1000  # Convert milliseconds to seconds

    # if target duration is provided,
    speaking_rate = 1.0
    if target_length:
        # Calculate required speaking rate (limit adjustment to ±25%)
        speaking_rate = min(max(initial_duration / target_length, 0.75), 1.25)

        # Generate final audio with adjusted speaking rate
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            effects_profile_id=["headphone-class-device"],
            pitch=0,
            speaking_rate=speaking_rate
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

    # Save the file
    with open(filename, "wb") as out:
        out.write(response.audio_content)

    # Return actual duration
    with io.BytesIO(response.audio_content) as f:
        audio = AudioSegment.from_file(f)
    final_duration = len(audio) / 1000

    return final_duration, speaking_rate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:/sync_folder/TSRlearn-analysis/formal-province-473109-k6-62ae26505f12.json"