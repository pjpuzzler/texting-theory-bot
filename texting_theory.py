import enum
import json
import os
import random
from dataclasses import dataclass
from typing import List
from pilmoji import Pilmoji
from pilmoji.source import AppleEmojiSource
from PIL import Image, ImageDraw, ImageFont, ImageColor
from google import genai
from google.genai import types
from prompt import decrypt_prompt


# from dotenv import load_dotenv
# load_dotenv()


client = genai.Client(api_key=os.environ[["GEMINI_API_KEY", "GEMINI_API_KEY_ALT", "GEMINI_API_KEY_ALT_2"][random.randint(0, 2)]])


class Classification(enum.Enum):
  BEST = "best"
  BLUNDER = "blunder"
  BOOK = "book"
  BRILLIANT = "brilliant"
  CHECKMATE = "checkmate"
  CLOCK = "clock"
  DRAW = "draw"
  EXCELLENT = "excellent"
  FORCED = "forced"
  GOOD = "good"
  GREAT = "great"
  INACCURACY = "inaccuracy"
  MISS = "miss"
  MISTAKE = "mistake"
  RESIGN = "resign"
  WINNER = "winner"

  def png_path(self, color: str) -> str:
    match self:
      case Classification.CHECKMATE | Classification.DRAW | Classification.RESIGN | Classification.CLOCK:
        return f"1024x/{self.value}_{color}_1024x.png"
      case _:
        return f"1024x/{self.value}_1024x.png"


@dataclass
class TextMessage:
  side: str
  content: str
  classification: Classification
  unsent: bool = False

def load_system_prompt():
    key = os.environ.get("PROMPT_KEY")
    
    with open("system_prompt_e.txt", "r", encoding="utf-8") as f:
        encrypted_prompt = f.read()
    
    system_prompt = decrypt_prompt(encrypted_prompt, key)
    return system_prompt


SYSTEM_PROMPT = load_system_prompt()

def call_llm_on_image(image_path: str, title: str, body: str) -> dict:
  image = client.files.upload(file=image_path)
  response = client.models.generate_content(
      model="gemini-2.5-pro-exp-03-25",
    #   model="gemini-2.5-flash-preview-04-17",
      contents=[
          types.Part.from_text(
              text=
              f'Here is the possibly stitched-together image, along with the title and body text (if any) of the post, which may have additional context to help inform you. Do not let any opinion of the poster that may be included in this text influence yours, as this is your decision to make.\n\nTitle: {title}\nBody: {body}'
          ),
          types.Part.from_uri(file_uri=image.uri, mime_type="image/jpeg"),
      ],
      config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0, safety_settings=[types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.OFF,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            threshold=types.HarmBlockThreshold.OFF,
        )]))
#   print(response.__dict__)
  print(f'Result: {response.text}')
  data = json.loads(response.text.removeprefix('```json\n').removesuffix('\n```'))
  return data


def parse_llm_response(data) -> List[TextMessage]:
  msgs = []
  for item in data.get("messages", []):
    classification = Classification[item["classification"]]
    msgs.append(
        TextMessage(
            side=item["side"],
            content=item["content"],
            classification=classification,
            unsent=item.get("unsent", False)
        ))
  return msgs


def wrap_text(text, draw, font, max_width):
    def ellipsize(word):
        ellipsis = "..."
        ellipsis_width = draw.textbbox((0, 0), ellipsis, font=font)[2]
        if ellipsis_width > max_width:
            return ""
        truncated = ""
        for char in word:
            test_word = truncated + char + ellipsis
            test_width = draw.textbbox((0, 0), test_word, font=font)[2]
            if test_width <= max_width:
                truncated += char
            else:
                break
        return truncated + ellipsis

    lines = []
    for para in text.split("\n"):
        words = para.split(" ")
        line = ""
        for w in words:
            w_width = draw.textbbox((0, 0), w, font=font)[2]
            if w_width > max_width:
                w = ellipsize(w)
            test_line = (line + " " + w).strip()
            test_box = draw.textbbox((0, 0), test_line, font=font)
            if test_box[2] - test_box[0] <= max_width:
                line = test_line
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
    return "\n".join(lines)


def hex_to_rgba(hex_str, alpha_factor=1.0):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 6:
        r, g, b = (int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    elif len(hex_str) == 3:
        r, g, b = (int(hex_str[i]*2, 16) for i in range(3))
    else:
        raise ValueError("Invalid hex color")
    a = int(255 * alpha_factor)
    return (r, g, b, a)

def render_conversation(messages: list[TextMessage], color_data_left, color_data_right, background_hex, output_path="output.png"):
    base_w = 320
    scale = 4
    img_w = base_w * scale

    font = ImageFont.truetype("Arial.ttf", 14 * scale)
    pad = 12 * scale
    line_sp = 6 * scale
    radius = 16 * scale
    badge_sz = 36 * scale
    badge_margin = 42 * scale

    max_bubble_w = int(img_w * 0.75)

    dummy = Image.new("RGB", (1, 1))
    dd = ImageDraw.Draw(dummy)
    wrapped, dims = [], []
    with Pilmoji(dummy, source=AppleEmojiSource) as pilmoji:
        for m in messages:
            txt = wrap_text(m.content, dd, font, max_bubble_w - 2 * pad)
            wrapped.append(txt)
            w, h = pilmoji.getsize(txt, font=font, spacing=line_sp)
            dims.append((w, h))

    total_h = pad
    for i, (w, h) in enumerate(dims):
        bh = h + 2 * pad
        total_h += bh
        if i < len(dims) - 1:
            next_spacing = pad // 4 if messages[i + 1].side == messages[i].side else pad
            total_h += next_spacing
    total_h += pad

    bg_rgba = ImageColor.getcolor(background_hex, "RGBA")
    
    img_bg = Image.new("RGBA", (img_w, total_h), bg_rgba)
    
    bubble_layer = Image.new("RGBA", (img_w, total_h), (0, 0, 0, 0))

    
    text_drawings = []

    y = pad
    text_offset = int(0 * scale)
    for i, (m, txt, (w, h)) in enumerate(zip(messages, wrapped, dims)):
        bw = w + 2 * pad
        bh = h + 2 * pad
        if m.side == "left":
            x0 = pad
            badge_x = x0 + bw - badge_sz + badge_margin
            bubble_color = color_data_left["bubble_hex"]
            text_hex = color_data_left["text_hex"]
        else:
            x0 = img_w - bw - pad
            badge_x = x0 - badge_margin
            bubble_color = color_data_right["bubble_hex"]
            text_hex = color_data_right["text_hex"]

        x1, y1 = x0 + bw, y + bh

        
        opacity = 0.67 if m.unsent else 1.0
        bubble_rgba = hex_to_rgba(bubble_color, opacity)
        bubble_draw = ImageDraw.Draw(bubble_layer)
        if m.side == "left":
            tail = [(x0 + 2 * scale, y + bh - 16 * scale),
                    (x0 - 6 * scale, y + bh),
                    (x0 + 10 * scale, y + bh - 4 * scale)]
            bubble_draw.polygon(tail, fill=bubble_rgba)
        else:
            tail = [(x1 - 2 * scale, y + bh - 16 * scale),
                    (x1 + 6 * scale, y + bh),
                    (x1 - 10 * scale, y + bh - 4 * scale)]
            bubble_draw.polygon(tail, fill=bubble_rgba)
        bubble_draw.rounded_rectangle((x0, y, x1, y1), radius, fill=bubble_rgba)

        
        text_drawings.append(((x0 + pad, y + pad - text_offset), txt, font, text_hex, line_sp, -10 if m.side=="left" else 10))

        badge = Image.open(
            m.classification.png_path("white" if m.side == "right" else "black")
        ).resize((badge_sz, badge_sz))
        if badge.mode != 'RGBA':
            badge = badge.convert('RGBA')
        if m.unsent:
            badge.putalpha(badge.getchannel('A').point(lambda x: int(x * 0.67)))
        by = y + (bh - badge_sz) // 2
        img_bg.paste(badge, (badge_x, by), badge)

        spacing = pad // 4 if (i < len(messages) - 1 and messages[i + 1].side == m.side) else pad
        y += bh + spacing

    
    composite_img = Image.alpha_composite(img_bg, bubble_layer)
    
    with Pilmoji(composite_img, source=AppleEmojiSource) as pilmoji:
        for pos, t, f, col, sp, offs in text_drawings:
            pilmoji.text(pos, t, font=f, fill=col, spacing=sp, emoji_scale_factor=1.3, emoji_position_offset=(offs, 0))

    final_img = composite_img.convert("RGB")
    final_img.save(output_path)


if __name__ == "__main__":
    data = call_llm_on_image("convo.jpeg", "", "")
    elo_left, elo_right = data["elo"].get("left"), data["elo"].get("right")
    color_data_left, color_data_right = data["color"].get("left"), data["color"].get("right")
    msgs = parse_llm_response(data)
    render_conversation(msgs, color_data_left, color_data_right, data["color"]["background_hex"], "final_chat.png")
    print('rendered image')
