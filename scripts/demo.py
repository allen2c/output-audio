import time

from output_audio import OpenAITTSAudioItem, output_audio


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


if __name__ == "__main__":
    demo_english()
    time.sleep(1.0)
    demo_mandarin()
