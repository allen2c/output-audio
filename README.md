# output-audio

A Python library for streaming audio output with playlist support, featuring real-time text-to-speech (TTS) capabilities using OpenAI's API.

## Features

- **Real-time streaming audio**: Stream audio directly to output devices with minimal latency
- **Playlist support**: Queue multiple audio items for seamless playback
- **Dynamic playlist management**: Add audio items to playlists during playback
- **OpenAI TTS integration**: Convert text to speech using OpenAI's TTS models
- **Seamless transitions**: Automatic padding between audio segments for smooth playback
- **Multi-language support**: Works with English, Mandarin, Japanese, and other languages
- **Low-latency buffering**: Pre-buffering system for smooth playback experience

## Installation

### Basic Installation

```bash
pip install output-audio
```

### With OpenAI TTS Support

```bash
pip install output-audio[all]
```

## Requirements

- Python 3.11+
- Audio output device (speakers/headphones)
- OpenAI API key (for TTS functionality)

## Quick Start

### Basic TTS Example

```python
from output_audio import OpenAITTSAudioItem, output_audio

# Create audio items
audio_items = [
    OpenAITTSAudioItem(content="Hello, this is the first segment."),
    OpenAITTSAudioItem(content="And this is the second segment."),
]

# Play audio
output_audio(audio_items)
```

### Dynamic Playlist Example

```python
import time
import threading
from output_audio import Playlist, OpenAITTSAudioItem, output_playlist_audio

# Create empty playlist
playlist = Playlist()
stop_event = threading.Event()

# Start playback in background
playback_thread = threading.Thread(
    target=output_playlist_audio,
    args=(playlist,),
    kwargs={"playback_stop_event": stop_event}
)
playback_thread.start()

# Add items dynamically
playlist.add_item(OpenAITTSAudioItem(content="First dynamic item"))
time.sleep(2)
playlist.add_item(OpenAITTSAudioItem(content="Second dynamic item"))

# Stop playback
time.sleep(5)
stop_event.set()
playback_thread.join()
```

## Configuration

### Audio Configuration

The library uses the following default audio settings:

- **Sample Rate**: 24,000 Hz (matches OpenAI PCM output)
- **Channels**: 1 (Mono)
- **Format**: 16-bit PCM
- **Buffer Size**: 1024 frames
- **Pre-buffer Duration**: 0.2 seconds

### TTS Configuration

Customize OpenAI TTS settings:

```python
from output_audio import TTSAudioConfig, OpenAITTSAudioItem
import openai

config = TTSAudioConfig(
    model="gpt-4o-mini-tts",  # or "tts-1"
    voice="nova",             # alloy, echo, fable, onyx, nova, shimmer
    speed=1.0,                # 0.25 to 4.0
    openai_client=openai.OpenAI(api_key="your-api-key")
)

audio_item = OpenAITTSAudioItem(
    content="Hello world!",
    audio_config=config
)
```

## API Reference

### Core Classes

#### `AudioItem`

Base class for audio items.

#### `OpenAITTSAudioItem`

Audio item that generates speech from text using OpenAI's TTS API.

**Parameters:**

- `content` (str): Text to convert to speech
- `audio_config` (TTSAudioConfig, optional): TTS configuration

#### `Playlist`

Container for managing multiple audio items with dynamic insertion support.

**Methods:**

- `add_item(audio_item)`: Add an audio item to the playlist
- `play(playback_queue)`: Start playlist playback

#### `TTSAudioConfig`

Configuration for OpenAI TTS settings.

**Parameters:**

- `model` (str): TTS model ("gpt-4o-mini-tts" or "tts-1")
- `voice` (str): Voice selection
- `speed` (float): Playback speed (0.25-4.0)
- `instructions` (str): Voice instructions
- `openai_client`: OpenAI client instance

### Functions

#### `output_audio(audio_items)`

Play a sequence of audio items.

**Parameters:**

- `audio_items`: List of AudioItem instances

#### `output_playlist_audio(playlist, playback_stop_event=None)`

Play a playlist with dynamic item insertion support.

**Parameters:**

- `playlist`: Playlist instance
- `playback_stop_event`: Threading event to stop playback

## Examples

See `scripts/demo.py` for comprehensive examples including:

- English TTS demo
- Mandarin TTS demo
- Dynamic playlist management

Run the demo:

```bash
python scripts/demo.py
```

## Environment Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```bash
OPENAI_API_KEY=your-api-key-here
```

## Dependencies

- `numpy`: Numerical operations for audio data
- `sounddevice`: Audio device interface
- `pydantic`: Data validation and settings
- `openai`: OpenAI API client (optional)

## License

MIT License - see LICENSE file for details.

## Author

Allen Chou (<f1470891079@gmail.com>)
