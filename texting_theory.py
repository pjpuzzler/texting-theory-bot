import enum
import json
import os
import random
import textwrap
import datetime
import requests
import io
from dataclasses import dataclass
from typing import List
from pilmoji import Pilmoji
from pilmoji.source import AppleEmojiSource
from PIL import Image, ImageDraw, ImageFont, ImageColor
from google import genai
from google.genai import types
from prompt import decrypt_prompt
from random_key import key_id


# from dotenv import load_dotenv
# load_dotenv()


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
  MEGABLUNDER = "megablunder"
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

def api_key():
    return os.environ["GEMINI_API_KEY" + key_id()]

@dataclass
class TextMessage:
  side: str
  content: str
  classification: Classification
  unsent: bool = False
  username: str = None
  avatar_url: str = None

def load_system_prompt():
    key = os.environ.get("PROMPT_KEY")
    
    with open("system_prompt_e.txt", "r", encoding="utf-8") as f:
        encrypted_prompt = f.read()
    
    system_prompt = decrypt_prompt(encrypted_prompt, key)
    return system_prompt


API_KEY = api_key()
client = genai.Client(api_key=API_KEY)
SYSTEM_PROMPT = load_system_prompt()



def call_llm_on_image(image_path: str, title: str, body: str) -> dict:
  if datetime.datetime.now().weekday() == 0:
      extra = "\n\nP.S. Today is Monday, which means you have the special ability to classify a message as a MEGABLUNDER! If a message is truly deserving of something even worse than a BLUNDER, you have the ability today to give it the rating it truly deserves. Use it sparingly, only for the worst-of-the-worst incomprehesibly bad BLUNDERs; the absolute worst move you could have played there. "
  else:
      extra = ''
  image = client.files.upload(file=image_path)
  response = client.models.generate_content(
    #   model="gemini-2.5-pro-exp-03-25",
    #   model="gemini-2.5-flash-preview-04-17",
      model="gemini-2.5-flash-preview-04-17-thinking",
      contents=[
          types.Part.from_text(
              text=f'Post Title: "{title}"\n\nPost Body: "{body}"'
          ),
          types.Part.from_uri(file_uri=image.uri, mime_type="image/jpeg"),
      ],
      config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT + extra, temperature=0.0, safety_settings=[types.SafetySetting(
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
  no_book = False
  for item in data.get("messages", []):
    classification = Classification[item["classification"]]
    if classification is not Classification.BOOK:
        no_book = True
    elif no_book:
        classification = Classification.GOOD
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
            next_spacing = pad // 5 if messages[i + 1].side == messages[i].side else int(pad * 0.67)
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

        
        bubble_draw = ImageDraw.Draw(bubble_layer)
        if m.unsent:
            if m.side == "left":
                center_big = (x0 + 5 * scale, y1 - 5 * scale)
                big_rad = 7 * scale
                bbox_big = (center_big[0] - big_rad, center_big[1] - big_rad,
                            center_big[0] + big_rad, center_big[1] + big_rad)
                bubble_draw.ellipse(bbox_big, fill=bubble_color)
                
                center_small = (x0 - 3 * scale, y1 + 3 * scale)
                small_rad = 3 * scale
                bbox_small = (center_small[0] - small_rad, center_small[1] - small_rad,
                            center_small[0] + small_rad, center_small[1] + small_rad)
                bubble_draw.ellipse(bbox_small, fill=bubble_color)
            else:
                center_big = (x1 - 5 * scale, y1 - 5 * scale)
                big_rad = 7 * scale
                bbox_big = (center_big[0] - big_rad, center_big[1] - big_rad,
                            center_big[0] + big_rad, center_big[1] + big_rad)
                bubble_draw.ellipse(bbox_big, fill=bubble_color)
                
                center_small = (x1 + 3 * scale, y1 + 3 * scale)
                small_rad = 3 * scale
                bbox_small = (center_small[0] - small_rad, center_small[1] - small_rad,
                            center_small[0] + small_rad, center_small[1] + small_rad)
                bubble_draw.ellipse(bbox_small, fill=bubble_color)
        else:
            if m.side == "left":
                tail = [(x0 + 2 * scale, y + bh - 16 * scale),
                        (x0 - 6 * scale, y + bh),
                        (x0 + 10 * scale, y + bh - 4 * scale)]
                bubble_draw.polygon(tail, fill=bubble_color)
            else:
                tail = [(x1 - 2 * scale, y + bh - 16 * scale),
                        (x1 + 6 * scale, y + bh),
                        (x1 - 10 * scale, y + bh - 4 * scale)]
                bubble_draw.polygon(tail, fill=bubble_color)
        bubble_draw.rounded_rectangle((x0, y, x1, y1), radius, fill=bubble_color)

        
        text_drawings.append(((x0 + pad, y + pad - text_offset), txt, font, text_hex, line_sp, -10 if m.side=="left" else 10))

        badge = Image.open(
            m.classification.png_path("white" if m.side == "right" else "black")
        ).resize((badge_sz, badge_sz))
        if badge.mode != 'RGBA':
            badge = badge.convert('RGBA')
        by = y + (bh - badge_sz) // 2
        img_bg.paste(badge, (badge_x, by), badge)

        spacing = pad // 5 if (i < len(messages) - 1 and messages[i + 1].side == m.side) else int(pad * 0.67)
        y += bh + spacing

    
    composite_img = Image.alpha_composite(img_bg, bubble_layer)
    
    with Pilmoji(composite_img, source=AppleEmojiSource) as pilmoji:
        for pos, t, f, col, sp, offs in text_drawings:
            pilmoji.text(pos, t, font=f, fill=col, spacing=sp, emoji_scale_factor=1.3, emoji_position_offset=(offs, 0))

    final_img = composite_img.convert("RGB")
    final_img.save(output_path)


def wrap_text_by_width(text: str, font, max_width: int, measure_fn) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        line = ""
        for word in words:
            test = f"{line} {word}".strip()
            w, _ = measure_fn(test, font)
            if w <= max_width:
                line = test
            else:
                if line:
                    lines.append(line)
                # If a single word is too long, break it character-by-character
                sub = ""
                for ch in word:
                    wch, _ = measure_fn(sub + ch, font)
                    if wch <= max_width:
                        sub += ch
                    else:
                        lines.append(sub)
                        sub = ch
                line = sub
        if line:
            lines.append(line)
    return lines


def render_reddit_chain(
    messages: list[TextMessage],
    output_path: str,
    *,
    max_width: int = 1280,
    bg_color: str = "#101214",
    username_color: str = "#8FA1AB",
    text_color: str = "#FFFFFF",
    connector_color: str = "#484848",
):
    # Constants
    avatar_sz = 114
    pad = 45  # Standard padding value for consistency
    badge_sz = 144
    
    # Fonts
    font_username = ImageFont.truetype("Arial Bold.ttf", 40)
    font_text = ImageFont.truetype("Arial.ttf", 46)

    # Setup for text measurement
    dummy = Image.new("RGB", (1, 1))
    measurer = ImageDraw.Draw(dummy)
    def measure(txt, fnt):
        bb = measurer.textbbox((0, 0), txt, font=fnt)
        return bb[2] - bb[0], bb[3] - bb[1]

    # Get line height once for consistency
    line_height = measure("Test", font_text)[1]
    line_spacing = 18
    
    # Pre-process all messages to calculate proper dimensions
    wrapped = []
    message_positions = []
    current_y = 0
    
    for msg in messages:
        # Calculate available width for text
        max_bubble_w = max_width - (pad + avatar_sz + pad) - (pad + badge_sz)
        
        # Wrap text
        lines = wrap_text_by_width(msg.content, font_text, max_bubble_w, measure)
        wrapped.append(lines)
        
        # Calculate positions
        message_top = current_y
        avatar_pos = (pad, message_top + pad)  # Avatar is padded from top of message area
        
        # Username position (relative to avatar)
        username_y = message_top + pad + avatar_sz//2 - 20
        username_pos = (pad + avatar_sz + pad - 10, username_y)
        
        # Text position (relative to username)
        text_start_y = username_y + 96
        
        # Calculate total text height precisely
        text_height = 0
        if lines:
            text_height = (len(lines) * line_height) + ((len(lines) - 1) * line_spacing)
        
        # Calculate text bottom position
        text_bottom = text_start_y + text_height
        
        # Find center of text area
        text_center_y = text_start_y + (text_height // 2) if lines else username_y + 48
        
        # Badge positions
        badge_y = text_center_y - (badge_sz // 2)
        badge_pos = (max_width - (pad + badge_sz), badge_y)
        badge_bottom = badge_y + badge_sz
        
        # Find the bottom-most element
        message_bottom = max(text_bottom, badge_bottom)
        
        # Always ensure there's proper padding at the bottom from the lowest element
        block_bottom = message_bottom + pad
        
        # Store all positions for this message
        message_positions.append({
            'avatar_pos': avatar_pos,
            'username_pos': username_pos,
            'text_start_y': text_start_y,
            'text_height': text_height,
            'badge_pos': badge_pos,
            'block_bottom': block_bottom,
            'message_top': message_top,
        })
        
        # Next message starts at the bottom of this block
        current_y = block_bottom
    
    # Create canvas with calculated total height
    canvas = Image.new("RGB", (max_width, current_y), bg_color)
    draw = ImageDraw.Draw(canvas)
    
    # Draw each message
    for idx, msg in enumerate(messages):
        positions = message_positions[idx]
        
        # Extract positions
        avatar_pos = positions['avatar_pos']
        username_pos = positions['username_pos']
        text_start_y = positions['text_start_y']
        badge_pos = positions['badge_pos']
        
        # Avatar processing
        try:
            resp = requests.get(msg.avatar_url, timeout=3)
            av_src = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        except:
            av_src = Image.new("RGBA", (avatar_sz, avatar_sz), "#888888")
            
        # Avatar resize and mask preparation
        big = avatar_sz * 4
        av4 = av_src.resize((big, big), Image.LANCZOS)
        
        # Avatar background gradient
        top_col = (40, 40, 40, 255)
        bot_col = (40, 40, 40, 255)
        grad_bg = Image.new("RGBA", (big, big))
        draw_bg = ImageDraw.Draw(grad_bg)
        for gy in range(big):
            t = gy / (big - 1)
            r = int(top_col[0] * (1 - t) + bot_col[0] * t)
            g = int(top_col[1] * (1 - t) + bot_col[1] * t)
            b = int(top_col[2] * (1 - t) + bot_col[2] * t)
            draw_bg.line([(0, gy), (big, gy)], fill=(r, g, b, 255))
            
        grad_bg.paste(av4, (0, 0), av4)
        
        # Create circular mask
        mask4 = Image.new("L", (big, big), 0)
        md = ImageDraw.Draw(mask4)
        md.ellipse((0, 0, big, big), fill=255)
        
        # Resize avatar and mask
        avatar = grad_bg.resize((avatar_sz, avatar_sz), Image.LANCZOS)
        mask = mask4.resize((avatar_sz, avatar_sz), Image.LANCZOS)
        
        # Place avatar
        canvas.paste(avatar, avatar_pos, mask)
        
        # Username
        draw.text(username_pos, msg.username, font=font_username, fill=username_color)
        
        # Text content
        text_x = pad + avatar_sz + pad - 10
        text_y = text_start_y
        
        # Draw each line of text
        for line in wrapped[idx]:
            draw.text((text_x, text_y), line, font=font_text, fill=text_color)
            text_y += line_height + line_spacing
            
        # Add classification badge if available
        bp = msg.classification.png_path("white")
        if os.path.exists(bp):
            bd = Image.open(bp).convert("RGBA").resize((badge_sz, badge_sz), Image.LANCZOS)
            canvas.paste(bd, badge_pos, bd)
    
    # Save the final image
    canvas.save(output_path)


if __name__ == "__main__":
    # data = call_llm_on_image("convo.png", "", "")
    # elo_left, elo_right = data["elo"].get("left"), data["elo"].get("right")
    # color_data_left, color_data_right = data["color"].get("left"), data["color"].get("right")
    # msgs = parse_llm_response(data)
    # print(msgs)
    # render_conversation(msgs, color_data_left, color_data_right, data["color"]["background_hex"], "final_chat.png")
    # print('rendered image')
    render_reddit_chain([
        # TextMessage("", "I want to date aant to dfdsfda fsaeaaaaa", Classification.MEGABLUNDER, username="asfdsafs", avatar_url="https://styles.redditmedia.com/t5_58b4ep/styles/profileIcon_snooc8df9e5b-e1ee-451d-ba16-265db020b93e-headshot.png?width=256&height=256&crop=256:256,smart&s=0a62fc9c851c8ef3571abfa282d08ad5df4cc0ac"),
        # TextMessage("", "I want to date and fuck fdsjkfldjslfsthe fdsafdsafdkslafjdklsjflasldfsa", Classification.MEGABLUNDER, username="Equal-Bowl-377", avatar_url="https://styles.redditmedia.com/t5_58b4ep/styles/profileIcon_snooc8df9e5b-e1ee-451d-ba16-265db020b93e-headshot.png?width=256&height=256&crop=256:256,smart&s=0a62fc9c851c8ef3571abfa282d08ad5df4cc0ac"),
        # TextMessage("", "u right date aI want to date a", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        # TextMessage("", "u right date aI want to date a", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        # TextMessage("", "u right date aI want to date a", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        # TextMessage("", "u right date aI want to date a", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        # TextMessage("", "u right date aI want to date a", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        # TextMessage("", "u right date aI want to date a", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        # TextMessage("", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        # TextMessage("", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        TextMessage("", "u right coulda been more iiiifdsafdsafdsa fdsajilfdjs;a fdjskalfjd s;lakjf ;dsaiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiie a I want to date aI want to date a", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        # TextMessage("", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        TextMessage("", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        TextMessage("", "Where's the goddamn bot at???", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
        TextMessage("", "FdsjlaUf fLJ:fdsoiUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods ", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
                         ], "out2.jpg")
