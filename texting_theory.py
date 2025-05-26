import enum
import json
import os
import random
import textwrap
import requests
import io
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from datetime import datetime
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
    ABANDON = "abandon"
    BEST = "best"
    BLUNDER = "blunder"
    BOOK = "book"
    BRILLIANT = "brilliant"
    CHECKMATED = "checkmated"
    DRAW = "draw"
    EXCELLENT = "excellent"
    FORCED = "forced"
    GOOD = "good"
    GREAT = "great"
    INACCURACY = "inaccuracy"
    INTERESTING = "interesting"
    MEGABLUNDER = "megablunder"
    MISS = "miss"
    MISTAKE = "mistake"
    RESIGN = "resign"
    TIMEOUT = "timeout"
    WINNER = "winner"

    def png_path(self, color: str) -> str:
        match self:
            case Classification.CHECKMATED:
                return f"badges/{self.value}_{color}.png"
            case _:
                return f"badges/{self.value}.png"


def api_key():
    return os.environ["GEMINI_API_KEY"]


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


def call_llm_on_image(image_paths: list[str], title: str, body: str) -> dict:
    if datetime.now(ZoneInfo("America/New_York")).weekday() == 0:
        extra = "\n\nP.S. Today is Monday, which means you have the special ability to classify a message as a MEGABLUNDER! If a message is truly deserving of something even worse than a BLUNDER, you have the ability today to give it the rating it truly deserves. Use it sparingly, only for the worst-of-the-worst incomprehesibly bad BLUNDERs; the absolute worst move you could have played there. "
    else:
        extra = ""

    main_images = [client.files.upload(file=img_path) for img_path in image_paths]
    example_r = client.files.upload(file="examples/r.png")
    example_l = client.files.upload(file="examples/l.png")

    contents = [
        types.Part.from_text(text=f'Post Title: "{title}"\n\nPost Body: "{body}"')
    ]
    for main_image in main_images:
        contents.append(
            types.Part.from_uri(file_uri=main_image.uri, mime_type="image/jpeg")
        )
    contents.extend(
        [
            types.Part.from_text(
                text="Here is a blank example of a Hinge prompt from left being replied to by right (pink bubble with tail pointing to right):"
            ),
            types.Part.from_uri(file_uri=example_r.uri, mime_type="image/png"),
            types.Part.from_text(
                text="Here is a blank example of a Hinge prompt from right being replied to by left (pink bubble with tail pointing to left):"
            ),
            types.Part.from_uri(file_uri=example_l.uri, mime_type="image/png"),
        ]
    )

    response = client.models.generate_content(
        # model="gemini-2.5-pro-exp-03-25",
        model="gemini-2.5-flash-preview-05-20",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT + extra,
            temperature=0.0,
            top_k=1.0,
            seed=63,
            thinking_config=types.ThinkingConfig(thinking_budget=24576),
            safety_settings=[
                types.SafetySetting(
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
                ),
            ],
        ),
    )
    #   print(response.__dict__)
    print(f"Result: {response.text}")
    text = response.text[response.text.find("```json") :]
    data = json.loads(text.removeprefix("```json\n").removesuffix("\n```"))
    return data


def parse_llm_response(data, ignore_classifications=False) -> List[TextMessage]:
    msgs = []
    no_book = False
    messages = data.get("messages", [])
    for i, item in enumerate(messages):
        if ignore_classifications:
            classification = None
        else:
            classification = Classification[item["classification"]]
            if classification is not Classification.BOOK:
                no_book = True
            elif no_book:
                classification = Classification.GOOD

            if classification in [
                Classification.CHECKMATED,
                Classification.RESIGN,
                Classification.ABANDON,
                Classification.TIMEOUT,
                Classification.DRAW,
                Classification.WINNER,
            ] and not (
                (i == len(messages) - 1)
                or (
                    (i == len(messages) - 2)
                    and (messages[-1]["classification"] == "WINNER")
                )
            ):
                classification = Classification.GOOD
        msgs.append(
            TextMessage(
                side=item["side"],
                content=item["content"],
                classification=classification,
                unsent=item.get("unsent", False),
            )
        )
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


def render_conversation(
    messages: list[TextMessage],
    color_data_left,
    color_data_right,
    background_hex,
    output_path="output.png",
):
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
            next_spacing = (
                pad // 5
                if messages[i + 1].side == messages[i].side
                else int(pad * 0.67)
            )
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
                bbox_big = (
                    center_big[0] - big_rad,
                    center_big[1] - big_rad,
                    center_big[0] + big_rad,
                    center_big[1] + big_rad,
                )
                bubble_draw.ellipse(bbox_big, fill=bubble_color)

                center_small = (x0 - 3 * scale, y1 + 3 * scale)
                small_rad = 3 * scale
                bbox_small = (
                    center_small[0] - small_rad,
                    center_small[1] - small_rad,
                    center_small[0] + small_rad,
                    center_small[1] + small_rad,
                )
                bubble_draw.ellipse(bbox_small, fill=bubble_color)
            else:
                center_big = (x1 - 5 * scale, y1 - 5 * scale)
                big_rad = 7 * scale
                bbox_big = (
                    center_big[0] - big_rad,
                    center_big[1] - big_rad,
                    center_big[0] + big_rad,
                    center_big[1] + big_rad,
                )
                bubble_draw.ellipse(bbox_big, fill=bubble_color)

                center_small = (x1 + 3 * scale, y1 + 3 * scale)
                small_rad = 3 * scale
                bbox_small = (
                    center_small[0] - small_rad,
                    center_small[1] - small_rad,
                    center_small[0] + small_rad,
                    center_small[1] + small_rad,
                )
                bubble_draw.ellipse(bbox_small, fill=bubble_color)
        else:
            if m.side == "left":
                tail = [
                    (x0 + 2 * scale, y + bh - 16 * scale),
                    (x0 - 6 * scale, y + bh),
                    (x0 + 10 * scale, y + bh - 4 * scale),
                ]
                bubble_draw.polygon(tail, fill=bubble_color)
            else:
                tail = [
                    (x1 - 2 * scale, y + bh - 16 * scale),
                    (x1 + 6 * scale, y + bh),
                    (x1 - 10 * scale, y + bh - 4 * scale),
                ]
                bubble_draw.polygon(tail, fill=bubble_color)
        bubble_draw.rounded_rectangle((x0, y, x1, y1), radius, fill=bubble_color)

        text_drawings.append(
            (
                (x0 + pad, y + pad - text_offset),
                txt,
                font,
                text_hex,
                line_sp,
                -10 if m.side == "left" else 10,
            )
        )

        badge = Image.open(
            m.classification.png_path("white" if m.side == "right" else "black")
        ).resize((badge_sz, badge_sz))
        if badge.mode != "RGBA":
            badge = badge.convert("RGBA")
        by = y + (bh - badge_sz) // 2
        img_bg.paste(badge, (badge_x, by), badge)

        spacing = (
            pad // 5
            if (i < len(messages) - 1 and messages[i + 1].side == m.side)
            else int(pad * 0.67)
        )
        y += bh + spacing

    composite_img = Image.alpha_composite(img_bg, bubble_layer)

    with Pilmoji(composite_img, source=AppleEmojiSource) as pilmoji:
        for pos, t, f, col, sp, offs in text_drawings:
            pilmoji.text(
                pos,
                t,
                font=f,
                fill=col,
                spacing=sp,
                emoji_scale_factor=1.3,
                emoji_position_offset=(offs, 0),
            )

    final_img = composite_img.convert("RGB")
    final_img.save(output_path)


def wrap_text_by_width(text: str, font, max_width: int, measure_fn) -> list[str]:
    lines = []
    if not text.strip():
        return []

    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        current_line_being_built = ""
        for word_idx, word in enumerate(words):
            if (
                not word
                and word_idx > 0
                and (not words[word_idx - 1] or current_line_being_built.endswith(" "))
            ):
                continue
            if not word and not current_line_being_built:
                continue

            test_line = (
                f"{current_line_being_built} {word}".strip()
                if current_line_being_built
                else word
            )

            w, _ = measure_fn(test_line, font)
            if w <= max_width:
                current_line_being_built = test_line
            else:
                if current_line_being_built:
                    lines.append(current_line_being_built)

                word_w_itself, _ = measure_fn(word, font)
                if word_w_itself <= max_width:
                    current_line_being_built = word
                else:
                    sub_word_segment = ""
                    for char_in_word in word:
                        test_char_segment = sub_word_segment + char_in_word
                        char_seg_w, _ = measure_fn(test_char_segment, font)
                        if char_seg_w <= max_width:
                            sub_word_segment = test_char_segment
                        else:
                            if sub_word_segment:
                                lines.append(sub_word_segment)
                            sub_word_segment = char_in_word
                    current_line_being_built = sub_word_segment

        if current_line_being_built:
            lines.append(current_line_being_built)
    return lines


def render_reddit_chain(
    messages: list[TextMessage],
    output_path: str,
    *,
    max_image_width: int = 1280,
    bg_color: str = "#101214",
    username_color: str = "#8FA1AB",
    text_color: str = "#D4D7D9",
):
    SIDE_MARGIN = 45
    TOP_MARGIN = 45
    BETWEEN_MESSAGES_VERTICAL_SPACING = 40
    BOTTOM_IMAGE_PADDING = BETWEEN_MESSAGES_VERTICAL_SPACING

    AVATAR_SIZE = 136

    USERNAME_AVATAR_HORIZONTAL_GAP = 30
    AVATAR_TEXT_BLOCK_VERTICAL_SPACING = 50

    TEXT_LINE_LEADING = 18

    BADGE_SIZE = 144
    TEXT_BADGE_HORIZONTAL_GAP = 30

    try:
        font_username = ImageFont.truetype("Arial Bold.ttf", 56)
        font_text = ImageFont.truetype("Arial.ttf", 64)
    except IOError:
        print("Warning: Arial fonts not found. Using default.")
        font_username = ImageFont.load_default()
        font_text = ImageFont.load_default()

    dummy_image = Image.new("RGB", (1, 1))
    measurer = ImageDraw.Draw(dummy_image)

    def measure(text_to_measure, font_to_use):
        if not text_to_measure:
            return (0, 0)
        bbox = measurer.textbbox((0, 0), text_to_measure, font=font_to_use, anchor="lt")
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    TEXT_LINE_BBOX_HEIGHT = measure("Tg", font_text)[1]

    if not messages:
        final_height = TOP_MARGIN + BOTTOM_IMAGE_PADDING
        canvas = Image.new("RGB", (max_image_width, final_height), bg_color)
        canvas.save(output_path)
        print("No messages to render. Saved empty image.")
        return

    message_layouts = []
    for msg in messages:
        max_text_width = (
            max_image_width
            - SIDE_MARGIN
            - (BADGE_SIZE + TEXT_BADGE_HORIZONTAL_GAP + SIDE_MARGIN)
        )
        wrapped_lines = wrap_text_by_width(
            msg.content, font_text, max_text_width, measure
        )

        text_block_height = 0
        if wrapped_lines:
            text_block_height = (len(wrapped_lines) * TEXT_LINE_BBOX_HEIGHT) + (
                (len(wrapped_lines) - 1) * TEXT_LINE_LEADING
                if len(wrapped_lines) > 1
                else 0
            )

        badge_path_check = msg.classification.png_path("white")
        badge_exists_check = os.path.exists(badge_path_check)

        message_layouts.append(
            {
                "lines": wrapped_lines,
                "text_block_height": text_block_height,
                "username_width": measure(msg.username, font_username)[0],
                "username_height": measure(msg.username, font_username)[1],
                "badge_exists": badge_exists_check,
                "badge_path": badge_path_check if badge_exists_check else None,
            }
        )

    message_draw_details = []
    current_y_cursor = TOP_MARGIN

    for idx, msg_layout_info in enumerate(message_layouts):

        avatar_draw_x = SIDE_MARGIN
        avatar_draw_y = current_y_cursor
        avatar_center_y = avatar_draw_y + AVATAR_SIZE / 2
        avatar_bottom_y = avatar_draw_y + AVATAR_SIZE

        username_draw_x = avatar_draw_x + AVATAR_SIZE + USERNAME_AVATAR_HORIZONTAL_GAP
        username_draw_y = avatar_center_y - (msg_layout_info["username_height"] / 2)
        username_bottom_y = username_draw_y + msg_layout_info["username_height"]

        current_text_block_height = msg_layout_info["text_block_height"]
        text_block_actual_start_x = avatar_draw_x
        text_block_actual_start_y = avatar_bottom_y + AVATAR_TEXT_BLOCK_VERTICAL_SPACING
        text_block_actual_bottom_y = (
            text_block_actual_start_y + current_text_block_height
        )

        badge_draw_x = max_image_width - SIDE_MARGIN - BADGE_SIZE
        badge_is_present = msg_layout_info["badge_exists"]
        badge_draw_y = 0
        badge_actual_bottom_y = text_block_actual_start_y

        if badge_is_present:

            effective_text_height_for_badge_centering = current_text_block_height
            if current_text_block_height == 0:
                effective_text_height_for_badge_centering = TEXT_LINE_BBOX_HEIGHT

            badge_draw_y = (
                text_block_actual_start_y
                + (effective_text_height_for_badge_centering - BADGE_SIZE) / 2
            )
            badge_actual_bottom_y = badge_draw_y + BADGE_SIZE

        lowest_of_avatar_username_row = max(avatar_bottom_y, username_bottom_y)

        elements_below_avatar_bottoms = [text_block_actual_bottom_y]
        if badge_is_present:
            elements_below_avatar_bottoms.append(badge_actual_bottom_y)
        else:
            if current_text_block_height == 0:
                elements_below_avatar_bottoms.append(text_block_actual_start_y)

        lowest_of_elements_below_avatar = max(elements_below_avatar_bottoms)

        current_message_content_bottom_y = max(
            lowest_of_avatar_username_row, lowest_of_elements_below_avatar
        )

        message_draw_details.append(
            {
                "avatar_pos": (avatar_draw_x, avatar_draw_y),
                "username_pos": (username_draw_x, username_draw_y),
                "text_lines": msg_layout_info["lines"],
                "text_block_start_pos": (
                    text_block_actual_start_x,
                    text_block_actual_start_y,
                ),
                "badge_pos": (badge_draw_x, badge_draw_y),
                "badge_exists": badge_is_present,
                "badge_path": msg_layout_info["badge_path"],
                "content_bottom_y": current_message_content_bottom_y,
            }
        )

        current_y_cursor = (
            current_message_content_bottom_y + BETWEEN_MESSAGES_VERTICAL_SPACING
        )

    if not message_draw_details:
        final_image_height = TOP_MARGIN + BOTTOM_IMAGE_PADDING
    else:
        last_message_content_bottom = message_draw_details[-1]["content_bottom_y"]
        final_image_height = int(last_message_content_bottom + BOTTOM_IMAGE_PADDING)

    min_height_calc = (
        TOP_MARGIN
        + AVATAR_SIZE
        + AVATAR_TEXT_BLOCK_VERTICAL_SPACING
        + BOTTOM_IMAGE_PADDING
    )
    final_image_height = max(final_image_height, min_height_calc)

    canvas = Image.new("RGB", (max_image_width, int(final_image_height)), bg_color)
    draw = ImageDraw.Draw(canvas)

    for idx, details in enumerate(message_draw_details):
        msg_obj = messages[idx]

        try:
            resp = requests.get(msg_obj.avatar_url, timeout=5)
            resp.raise_for_status()
            avatar_source_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
        except requests.exceptions.RequestException as e:

            avatar_source_img = Image.new("RGBA", (AVATAR_SIZE, AVATAR_SIZE), "#888")
        except IOError:

            avatar_source_img = Image.new("RGBA", (AVATAR_SIZE, AVATAR_SIZE), "#888")

        hires_avatar_size = AVATAR_SIZE * 4
        hires_avatar = avatar_source_img.resize(
            (hires_avatar_size, hires_avatar_size), Image.LANCZOS
        )
        avatar_bg_hires = Image.new(
            "RGBA", (hires_avatar_size, hires_avatar_size), bg_color
        )
        avatar_bg_hires.paste(hires_avatar, (0, 0), hires_avatar)
        mask_hires = Image.new("L", (hires_avatar_size, hires_avatar_size), 0)
        ImageDraw.Draw(mask_hires).ellipse(
            (0, 0, hires_avatar_size, hires_avatar_size), fill=255
        )
        final_avatar = avatar_bg_hires.resize((AVATAR_SIZE, AVATAR_SIZE), Image.LANCZOS)
        final_mask = mask_hires.resize((AVATAR_SIZE, AVATAR_SIZE), Image.LANCZOS)
        canvas.paste(
            final_avatar,
            (int(details["avatar_pos"][0]), int(details["avatar_pos"][1])),
            final_mask,
        )

        draw.text(
            (int(details["username_pos"][0]), int(details["username_pos"][1])),
            msg_obj.username,
            font=font_username,
            fill=username_color,
            anchor="lt",
        )

        current_text_y = details["text_block_start_pos"][1]
        for line_text in details["text_lines"]:
            draw.text(
                (int(details["text_block_start_pos"][0]), int(current_text_y)),
                line_text,
                font=font_text,
                fill=text_color,
                anchor="lt",
            )
            current_text_y += TEXT_LINE_BBOX_HEIGHT + TEXT_LINE_LEADING

        if details["badge_exists"] and details["badge_path"]:
            try:
                badge_img = Image.open(details["badge_path"]).convert("RGBA")
                badge_img_resized = badge_img.resize(
                    (BADGE_SIZE, BADGE_SIZE), Image.LANCZOS
                )
                canvas.paste(
                    badge_img_resized,
                    (int(details["badge_pos"][0]), int(details["badge_pos"][1])),
                    badge_img_resized,
                )
            except FileNotFoundError:
                print(f"Badge file not found: {details['badge_path']}")
            except IOError:
                print(f"Could not open badge: {details['badge_path']}")

    canvas.save(output_path)
    print(f"Reddit chain image saved to {output_path}")


if __name__ == "__main__":
    # data = call_llm_on_image("convo.png", "", "")
    # elo_left, elo_right = data["elo"].get("left"), data["elo"].get("right")
    # color_data_left, color_data_right = data["color"].get("left"), data["color"].get("right")
    # msgs = parse_llm_response(data)
    # print(msgs)
    # render_conversation(msgs, color_data_left, color_data_right, data["color"]["background_hex"], "final_chat.png")
    # print('rendered image')
    render_reddit_chain(
        [
            TextMessage(
                "",
                "u right date aI want to date a",
                Classification.EXCELLENT,
                username="Equal-Bowl-377",
                avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png",
            ),
            TextMessage(
                "",
                "u right date aI want to date a",
                Classification.EXCELLENT,
                username="Equal-Bowl-377",
                avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png",
            ),
            TextMessage(
                "",
                "u right date aI want to date a",
                Classification.EXCELLENT,
                username="Equal-Bowl-377",
                avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png",
            ),
            TextMessage(
                "",
                "u right date aI want to date a",
                Classification.EXCELLENT,
                username="Equal-Bowl-377",
                avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png",
            ),
            # TextMessage("", "u right date aI want to date a", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
            # TextMessage("", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
            # TextMessage("", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
            TextMessage(
                "",
                "u right coulda been more iiiifdsafdsafdsa fdsajilfdjs;a fdjskalfjd s;lakjf ;dsaiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiie a I want to date aI want to date a",
                Classification.EXCELLENT,
                username="Equal-Bowl-377",
                avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png",
            ),
            # TextMessage("", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Classification.EXCELLENT, username="Equal-Bowl-377", avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png"),
            TextMessage(
                "",
                "FdsjlaUf fLJ:fdsoiUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods FdsjlaUf fLJ:fdsoi fjdsifods ",
                Classification.EXCELLENT,
                username="Equal-Bowl-377",
                avatar_url="https://www.redditstatic.com/avatars/defaults/v2/avatar_default_0.png",
            ),
        ],
        "out2.jpg",
    )
