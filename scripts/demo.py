# /Users/allenchou/works/ac/output-audio/scripts/demo.py
import threading
import time

from output_audio import (
    OpenAITTSAudioItem,
    Playlist,
    output_audio,
    output_playlist_audio,
)


def demo_english():

    # Create demo playlist
    demo_items = [
        OpenAITTSAudioItem(
            content="Part 1: Hello, welcome to the OpenAI text-to-speech demonstration."
        ),
        OpenAITTSAudioItem(
            content="Part 2: Now we enter the second sentence, and we will seamlessly continue from the previous segment."  # noqa: E501
        ),
    ]

    # Play demo
    output_audio(demo_items)
    print("✅ Demo completed!")


def demo_mandarin():

    # Create demo playlist
    demo_items = [
        OpenAITTSAudioItem(content="第一段：你好，歡迎收聽 OpenAI 文字轉語音示範。"),
        OpenAITTSAudioItem(
            content="第二段：現在進入第二句，我們會無縫接在上一段後面。"
        ),
    ]

    # Play demo
    output_audio(demo_items)
    print("✅ Demo completed!")


def demo_playlist_dynamic():
    print("🎵 Starting dynamic playlist demo...")

    playlist = Playlist()
    playback_stop_event = threading.Event()

    thread = threading.Thread(
        target=output_playlist_audio,
        args=(playlist,),
        kwargs={"playback_stop_event": playback_stop_event},
    )
    thread.start()

    # Add items dynamically
    texts = [
        "パート1：こんにちは。OpenAIのテキスト読み上げデモへようこそ。",
        "パート2：では、2つ目の文に入り、前のセグメントからスムーズに続けていきます。",
    ]

    print(f"📝 Adding {len(texts)} items to playlist...")
    for i, text in enumerate(texts):
        print(f"  Adding item {i+1}: {text[:50]}...")
        try:
            playlist.add_item(OpenAITTSAudioItem(content=text))
            print(f"  ✅ Item {i+1} added successfully")
            time.sleep(1.0)
        except Exception as e:
            print(f"  ❌ Failed to add item {i+1}: {e}")

    print("⏰ Waiting 5 seconds for playback to complete...")
    time.sleep(5.0)

    print("🛑 Setting stop event...")
    playback_stop_event.set()

    # Wait for playback thread to finish
    print("⏳ Waiting for playback thread to finish...")
    thread.join(timeout=10.0)

    if thread.is_alive():
        print("⚠️  Playback thread didn't finish in time")
    else:
        print("✅ Dynamic playlist demo completed!")


if __name__ == "__main__":
    # demo_english()
    # time.sleep(1.0)

    # demo_mandarin()
    # time.sleep(1.0)

    demo_playlist_dynamic()
    time.sleep(1.0)
