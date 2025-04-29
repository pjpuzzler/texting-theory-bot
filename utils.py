import os
import praw
import requests
import tempfile
import time
import json
from datetime import datetime, timezone, timedelta
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright
from texting_theory import call_llm_on_image, parse_llm_response, render_conversation, Classification

reddit = praw.Reddit(client_id=os.environ["REDDIT_CLIENT_ID"],
                     client_secret=os.environ["REDDIT_SECRET"],
                     username=os.environ["REDDIT_USERNAME"],
                     password=os.environ["REDDIT_PASSWORD"],
                     user_agent="texting-theory-replit/0.2")

STORAGE_FILE = "reddit_storage.json"

HUMANIZED_ORDER = [
    Classification.BRILLIANT,
    Classification.GREAT,
    Classification.BEST,
    Classification.EXCELLENT,
    Classification.GOOD,
    Classification.BOOK,
    Classification.INACCURACY,
    Classification.MISTAKE,
    Classification.MISS,
    Classification.BLUNDER,
]


def get_recent_posts():
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=360)
    return [
        post for post in reddit.subreddit("TextingTheory").new(limit=5)
        # if datetime.fromtimestamp(post.created_utc, tz=timezone.utc) > cutoff
    ]

def get_top_posts():
    return [
        post for post in reddit.subreddit("TextingTheory").top(time_filter="week", limit=10)
    ]

def get_post_by_id(post_id):
    return reddit.submission(id=post_id)


def stitch_images_vertically(image_paths, output_path):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    widths, heights = zip(*(i.size for i in images))
    stitched_image = Image.new("RGB", (max(widths), sum(heights)), "white")
    y_offset = 0
    for im in images:
        stitched_image.paste(im, (0, y_offset))
        y_offset += im.height
    stitched_image.save(output_path)


def upload_image_to_imgur(image_path):
    headers = {"Authorization": f"Client-ID {os.environ['IMGUR_CLIENT_ID']}"}
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}
        r = requests.post("https://api.imgur.com/3/image",
                          headers=headers,
                          files=files)
        r.raise_for_status()
        return r.json()["data"]["link"]


def format_counts(messages, color_left, color_right, elo_left, elo_right):
    counts = {c: [0, 0] for c in HUMANIZED_ORDER}
    has_message = [False, False]
    for m in messages:
        idx = 0 if m.side == "left" else 1
        has_message[idx] = True
        if m.classification in counts:
            counts[m.classification][idx] += 1
        elif m.classification is Classification.FORCED:
            counts[Classification.GOOD][idx] += 1

    lines = []
    lines.append(
        f"{color_left if color_left is not None and has_message[0] else ''}||{color_right if color_right is not None and has_message[1] else ''}\n:--:|:--:|:--:\n{elo_left if color_left is not None and has_message[0] else ''}|Elo (est.)|{elo_right if color_right is not None and has_message[1] else ''}\n||"
    )
    for c in HUMANIZED_ORDER:
        l, r = counts[c]
        label = c.name.replace('_', ' ').title()
        lines.append(f"{l if color_left is not None and has_message[0] else ''}|{label}|{r if color_right is not None and has_message[1] else ''}")

    return "\n".join(lines)


def post_comment_image(post_id, file_path, messages, color_left, color_right, elo_left, elo_right, opening):
    counts = {c: [0, 0] for c in HUMANIZED_ORDER}
    has_message = [False, False]
    for m in messages:
        idx = 0 if m.side == "left" else 1
        has_message[idx] = True
        if m.classification in counts:
            counts[m.classification][idx] += 1
        elif m.classification is Classification.FORCED:
            counts[Classification.GOOD][idx] += 1
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Headful mode
        context = None

        if Path(STORAGE_FILE).exists():
            print('Loading existing session...')
            context = browser.new_context(viewport={"width": 1600, "height": 900}, storage_state=STORAGE_FILE)
        else:
            assert False

        page = context.new_page()
        
        page.goto(f'https://www.reddit.com/r/TextingTheory/comments/{post_id}/')

        comments_button = page.locator('button[name="comments-action-button"]')
        comments_button.wait_for(state="visible", timeout=20000)
        comments_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        comments_button.click()

        formatting_button = page.locator('button:has(svg[icon-name="format-outline"])')
        formatting_button.wait_for(state="visible", timeout=5000)
        formatting_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        formatting_button.click()

        bold_button = page.locator('button:has(svg[icon-name="bold-outline"])')
        bold_button.wait_for(state="visible", timeout=5000)
        bold_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        bold_button.click()

        page.wait_for_timeout(100)

        page.keyboard.type("Game Review", delay=10)
        page.keyboard.press("Enter")

        page.wait_for_timeout(50)

        image_button = page.locator('button:has(svg[icon-name="image-post-outline"])')
        image_button.wait_for(state="visible", timeout=5000)

        with page.expect_file_chooser() as fc_info:
            image_button.scroll_into_view_if_needed()
            page.wait_for_timeout(100)
            image_button.click()
        
        page.wait_for_timeout(100)
        
        file_chooser = fc_info.value
        file_chooser.set_files(file_path)

        page.wait_for_timeout(100)

        page.keyboard.type(f"{opening}", delay=10)
        page.keyboard.press("Enter")

        page.wait_for_timeout(50)

        table_button = page.locator('button:has(svg[icon-name="table-outline"])')
        table_button.wait_for(state="visible", timeout=5000)
        table_button.scroll_into_view_if_needed()
        page.wait_for_timeout(250)
        table_button.click()

        for _ in range(8):
            table_actions_button = page.locator('button:has(svg[icon-name="overflow-horizontal-outline"]) >> text=Table actions menu')
            table_actions_button.wait_for(state="visible", timeout=5000)
            table_actions_button.scroll_into_view_if_needed()
            page.wait_for_timeout(200)
            table_actions_button.click()

            insert_row_button = page.get_by_text("Insert row below", exact=True)
            insert_row_button.wait_for(state="visible", timeout=5000)
            insert_row_button.scroll_into_view_if_needed()
            page.wait_for_timeout(200)
            insert_row_button.click()

        # table_actions_button = page.locator('button:has(svg[icon-name="overflow-horizontal-outline"]) >> text=Table actions menu')
        # table_actions_button.wait_for(state="visible", timeout=5000)
        # table_actions_button.scroll_into_view_if_needed()
        # page.wait_for_timeout(200)
        # table_actions_button.click()

        # align_center_button = page.get_by_text("Align center", exact=True)
        # align_center_button.wait_for(state="visible", timeout=5000)
        # align_center_button.scroll_into_view_if_needed()
        # page.wait_for_timeout(200)
        # align_center_button.click()

        page.wait_for_timeout(200)

        page.keyboard.type("a")
        page.wait_for_timeout(200)
        page.keyboard.press("Backspace")
        page.wait_for_timeout(100)

        if color_left is not None and has_message[0]:
            page.keyboard.type(f'{color_left} ({elo_left})', delay=10)
        page.keyboard.press("Tab")
        page.wait_for_timeout(50)

        # table_actions_button = page.locator('button:has(svg[icon-name="overflow-horizontal-outline"]) >> text=Table actions menu')
        # table_actions_button.wait_for(state="visible", timeout=5000)
        # table_actions_button.scroll_into_view_if_needed()
        # page.wait_for_timeout(200)
        # table_actions_button.click()

        # align_center_button = page.get_by_text("Align center", exact=True)
        # align_center_button.wait_for(state="visible", timeout=5000)
        # align_center_button.scroll_into_view_if_needed()
        # page.wait_for_timeout(200)
        # align_center_button.click()

        # page.wait_for_timeout(200)

        # page.keyboard.type("a")
        # page.wait_for_timeout(200)
        # page.keyboard.press("Backspace")
        # page.wait_for_timeout(100)

        page.keyboard.press("Tab")
        page.wait_for_timeout(50)

        # table_actions_button = page.locator('button:has(svg[icon-name="overflow-horizontal-outline"]) >> text=Table actions menu')
        # table_actions_button.wait_for(state="visible", timeout=5000)
        # table_actions_button.scroll_into_view_if_needed()
        # page.wait_for_timeout(200)
        # table_actions_button.click()

        # align_center_button = page.get_by_text("Align center", exact=True)
        # align_center_button.wait_for(state="visible", timeout=5000)
        # align_center_button.scroll_into_view_if_needed()
        # page.wait_for_timeout(200)
        # align_center_button.click()

        # page.wait_for_timeout(200)

        # page.keyboard.type("a")
        # page.wait_for_timeout(200)
        # page.keyboard.press("Backspace")
        # page.wait_for_timeout(100)

        if color_right is not None and has_message[1]:
            page.keyboard.type(f'{color_right} ({elo_right})', delay=10)
        page.keyboard.press("Tab")
        page.wait_for_timeout(50)

        for c in HUMANIZED_ORDER:
            l, r = counts[c]
            label = c.name.replace('_', ' ').title()

            if color_left is not None and has_message[0]:
                page.keyboard.type(str(l), delay=10)
            page.keyboard.press("Tab")
            page.wait_for_timeout(50)
            page.keyboard.type(label, delay=10)
            page.keyboard.press("Tab")
            page.wait_for_timeout(50)
            if color_right is not None and has_message[1]:
                page.keyboard.type(str(r), delay=10)
            page.keyboard.press("Tab")
            page.wait_for_timeout(50)

        page.keyboard.press("Enter")
        page.keyboard.press("Enter")

        link_button = page.locator('button:has(svg[icon-name="link-outline"])')
        link_button.wait_for(state="visible", timeout=5000)
        link_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        link_button.click()

        page.wait_for_timeout(100)

        page.keyboard.press("Tab")
        page.wait_for_timeout(100)
        page.keyboard.type("About the bot", delay=5)
        page.wait_for_timeout(100)
        page.keyboard.press("Tab")
        page.wait_for_timeout(100)
        # page.keyboard.type("https://support.chess.com/en/articles/8584089-how-does-game-review-work#h_49f5656333", delay=5)
        page.keyboard.type("https://www.reddit.com/r/TextingTheory/comments/1k8fed9/utextingtheorybot/", delay=5)

        save_link_button = page.get_by_test_id("btn-save-link")
        save_link_button.wait_for(state="visible", timeout=5000)
        save_link_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        save_link_button.click()

        comment_submit = page.locator('button[slot="submit-button"][type="submit"]')
        comment_submit.wait_for(state="visible", timeout=10000)
        comment_submit.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        comment_submit.click()

        print('Analysis Posted')

        page.wait_for_timeout(4000)

        # Clean up
        browser.close()


def handle_new_posts(post_id = None):
    # for post in get_recent_posts():
    # for post in get_top_posts():
    if post_id is None:
        posts = get_recent_posts()
    else:
        posts = [get_post_by_id(post_id)]
    for post in posts:
        print(f"Looking at post {post.id}")
        # if post.id != "1k40vss":
        #     continue

        if any(c.author
               and c.author.name.lower() == reddit.user.me().name.lower()
               for c in post.comments):
            print("Already analyzed")
            continue

        image_urls = []
        if hasattr(post, "post_hint") and post.post_hint == "image":
            image_urls = [post.url]
        elif hasattr(post, "gallery_data") and hasattr(post, 'media_metadata'):
            for item in post.gallery_data["items"]:
                u = post.media_metadata[item["media_id"]]["s"]["u"].replace(
                    "&amp;", "&")
                image_urls.append(u)
        elif hasattr(post, "preview") and "images" in post.preview:
            image_urls = [
                post.preview["images"][0]["source"]["url"].replace(
                    "&amp;", "&")
            ]

        if not image_urls:
            print("No images found")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            input_paths = []
            for idx, url in enumerate(image_urls):
                path = os.path.join(tmpdir, f"img{idx}.jpg")
                r = requests.get(url, headers={"User-Agent": "Mozilla"})
                with open(path, "wb") as f:
                    f.write(r.content)
                input_paths.append(path)

            stitched = os.path.join(tmpdir, "stitched.jpg")
            out_path = os.path.join(tmpdir, "out.jpg")
            stitch_images_vertically(input_paths, stitched)
            print(f'Analyzing post with title: {post.title}')
            data = call_llm_on_image(stitched, post.title, post.selftext)
            
            if data.get("is_convo") is False:
                print("Not a conversation, skipping")
            else:
                elo_left, elo_right = data["elo"].get("left"), data["elo"].get("right")
                color_data_left, color_data_right = data["color"].get("left"), data["color"].get("right")
                msgs = parse_llm_response(data)
                print("Parsed LLM response")
                render_conversation(msgs, color_data_left, color_data_right, data["color"]["background_hex"], out_path)
                print("Rendered analysis image")

                if any(c.author and c.author.name.lower() == reddit.user.me().name.lower() for c in post.comments):
                    print("Already analyzed")
                    continue

                post_comment_image(post.id, out_path, msgs, None if color_data_left is None else color_data_left["label"], None if color_data_right is None else color_data_right["label"], elo_left, elo_right, data.get("opening"))

                # img_url = upload_image_to_imgur(out_path)
                # print("Successfully uploaded to imgur")

                # breakdown = format_counts(msgs, None if color_data_left is None else color_data_left["label"], None if color_data_right is None else color_data_right["label"], elo_left, elo_right)
                # reply = f"**Game Review**\n\n{breakdown}\n\n[**Annotated Analysis**]({img_url})\n\n&nbsp;\n\n[*What do the classifications mean?*](https://support.chess.com/en/articles/8584089-how-does-game-review-work#h_49f5656333)"
                # post.reply(reply)
                # print(f"Commented on post {post.id}")
    print('Ran successfully')
    return "Done", 200
