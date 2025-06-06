import math
import os
import praw
import requests
import tempfile
import time
import json
from google import genai
from google.genai.types import EmbedContentConfig
from datetime import datetime, timezone, timedelta
from PIL import Image
from pathlib import Path
from pinecone import Pinecone
from playwright.sync_api import sync_playwright
from texting_theory import (
    call_llm_on_image,
    parse_llm_response,
    render_conversation,
    render_reddit_chain,
    Classification,
    TextMessage,
)

reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_SECRET"],
    username=os.environ["REDDIT_USERNAME"],
    password=os.environ["REDDIT_PASSWORD"],
    user_agent="texting-theory-replit/0.2",
)


from google import genai
from google.genai.types import EmbedContentConfig

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def get_convo_str(msgs):
    return "\n\n".join([f"{m.content}" for m in msgs])


def get_embedding(convo_str):
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=convo_str,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
        ),
    )
    return result.embeddings[0].values


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("texting-theory")


def pinecone_insert(post_id, embedding, convo_text):
    vector = {
        "id": post_id,
        "values": embedding,
        "metadata": {"convo_text": convo_text},
    }

    index.upsert(vectors=[vector])
    print(f"Uploaded vector for {post_id}")


def find_similar_conversations(embedding, cur_post_id, top_k=10, min_score=0.87, max=3):
    query_result = index.query(vector=embedding, top_k=top_k, include_metadata=True)

    return [
        (
            match.id,
            match.score,
            match.metadata["convo_text"],
        )
        for match in query_result.matches
        if match.id != cur_post_id
        and match.score >= min_score
        and not post_is_deleted(match.id)
    ][:max]


STORAGE_FILE = "reddit_storage.json"

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
KV_NAMESPACE_ID = os.getenv("KV_NAMESPACE_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

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
    Classification.MEGABLUNDER,
]

DIGIT_TO_CLASS = {
    "1": Classification.BRILLIANT,
    "2": Classification.GREAT,
    "3": Classification.BEST,
    "4": Classification.EXCELLENT,
    "5": Classification.GOOD,
    "6": Classification.INACCURACY,
    "7": Classification.MISTAKE,
    "8": Classification.MISS,
    "9": Classification.BLUNDER,
    "0": Classification.MEGABLUNDER,
    "b": Classification.BOOK,
    "f": Classification.FORCED,
    "i": Classification.INTERESTING,
    "a": Classification.ABANDON,
    "c": Classification.CHECKMATED,
    "d": Classification.DRAW,
    "r": Classification.RESIGN,
    "t": Classification.TIMEOUT,
    "w": Classification.WINNER,
}


def store_post_analysis_json(post_id: str, data: dict):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/storage/kv/namespaces/{KV_NAMESPACE_ID}/values/post:{post_id}"
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json",
    }
    response = requests.put(url, headers=headers, data=json.dumps(data))
    if not response.ok:
        raise Exception(
            f"KV store failed for post:{post_id} — {response.status_code}: {response.text}"
        )
    print(f"Stored post:{post_id} to KV")


def get_post_json_from_kv(post_id):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/storage/kv/namespaces/{KV_NAMESPACE_ID}/values/post:{post_id}"
    headers = {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        print(f"[!] Post {post_id} not found in KV.")
        return None
    response.raise_for_status()
    return response.json()


def get_recent_posts():
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=360)
    return [
        post
        for post in reddit.subreddit("TextingTheory").new(limit=10)
        # if datetime.fromtimestamp(post.created_utc, tz=timezone.utc) > cutoff
    ]


def get_top_posts():
    return [
        post
        for post in reddit.subreddit("TextingTheory").top(time_filter="week", limit=10)
    ]


def get_post_by_id(post_id):
    return reddit.submission(id=post_id)


def post_is_deleted(post_id):
    submission = get_post_by_id(post_id)
    return submission.author is None and not submission.is_robot_indexable


def apply_annotation_code(
    messages: list[TextMessage], code: str, reply: bool = False
) -> tuple[list[TextMessage] | None, str]:
    updated_msgs = []
    i = 0
    for j, ch in enumerate(code):
        if ch == "-":
            if j == len(code) - 1 or code[j + 1] == "-":
                return None, "char"
            continue  # skip, handled as prefix

        classification = DIGIT_TO_CLASS.get(ch.lower())
        if not classification:
            return None, "char"  # Invalid input

        negated = not reply and j > 0 and code[j - 1] == "-"

        try:
            msg = messages[i]
        except IndexError:
            return None, "len"

        # Make a copy with updated classification and (optionally) side
        new_msg = TextMessage(
            side=("right" if msg.side == "left" else "left") if negated else msg.side,
            content=msg.content,
            classification=classification,
            unsent=msg.unsent,
            username=msg.username,
            avatar_url=msg.avatar_url,
        )
        updated_msgs.append(new_msg)
        i += 1

    if i != len(messages):
        return None, "len"
    return updated_msgs, ""


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
        r = requests.post("https://api.imgur.com/3/image", headers=headers, files=files)
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
        label = c.name.replace("_", " ").title()
        lines.append(
            f"{l if color_left is not None and has_message[0] else ''}|{label}|{r if color_right is not None and has_message[1] else ''}"
        )

    return "\n".join(lines)


BLACK_SQUARE = "⬛"
WHITE_SQUARE = "⬜"


def eval_to_emoji_squares(eval_abs, total_squares: int = 16, max_pawn: float = 4.0):
    # 1) Saturate at max_pawn
    if eval_abs >= max_pawn:
        w_squares = total_squares - 1
    else:
        # 2) Arctan mapping for diminishing returns
        scale = max_pawn / 2.0
        frac = 0.5 + math.atan(eval_abs / scale) / math.pi
        # clamp to [0,1]
        frac = min(max(frac, 0.0), 1.0)
        # convert to square count
        w_squares = int(round(frac * total_squares))
        # ensure at least one white and one black remain
        w_squares = min(max(w_squares, 1), total_squares - 1)

    b_squares = total_squares - w_squares
    return b_squares, w_squares


def post_comment_image(
    post_id,
    file_path,
    messages,
    color_left,
    color_right,
    elo_left,
    elo_right,
    opening,
    similar_conversations,
    evaluation,
    best_continuation,
    summary,
):
    counts = {c: [0, 0] for c in HUMANIZED_ORDER}
    has_message = [False, False]
    for m in messages:
        idx = 0 if m.side == "left" else 1
        has_message[idx] = True
        if m.classification in counts:
            counts[m.classification][idx] += 1
        elif m.classification is Classification.FORCED:
            counts[Classification.GOOD][idx] += 1
    if counts[Classification.MEGABLUNDER] == [0, 0]:
        del counts[Classification.MEGABLUNDER]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Headful mode
        context = None

        if Path(STORAGE_FILE).exists():
            print("Loading existing session...")
            context = browser.new_context(
                viewport={"width": 1600, "height": 900}, storage_state=STORAGE_FILE
            )
        else:
            assert False

        page = context.new_page()

        page.goto(f"https://www.reddit.com/r/TextingTheory/comments/{post_id}/")

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

        page.keyboard.type("✪ Game Review", delay=10)
        page.keyboard.press("Enter")

        page.wait_for_timeout(50)

        bold_button = page.locator('button:has(svg[icon-name="bold-outline"])')
        bold_button.wait_for(state="visible", timeout=5000)
        bold_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        bold_button.click()

        page.wait_for_timeout(100)

        page.keyboard.type(summary, delay=10)
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

        page.wait_for_timeout(200)

        if evaluation is not None:
            LOSS_RESULTS = [
                Classification.CHECKMATED,
                Classification.ABANDON,
                Classification.RESIGN,
                Classification.TIMEOUT,
            ]
            left_classifications, right_classifications = [
                m.classification for m in messages if m.side == "left"
            ], [m.classification for m in messages if m.side == "right"]
            TOTAL_SQUARES = 16
            if any(loss_result in left_classifications for loss_result in LOSS_RESULTS):
                b_squares, w_squares = 0, TOTAL_SQUARES
                eval_str = "1-0"
                eval_str_right = True
            elif any(
                loss_result in right_classifications for loss_result in LOSS_RESULTS
            ):
                b_squares, w_squares = TOTAL_SQUARES, 0
                eval_str = "0-1"
                eval_str_right = False
            elif (
                Classification.DRAW in left_classifications
                or Classification.DRAW in right_classifications
            ):
                b_squares, w_squares = TOTAL_SQUARES // 2, TOTAL_SQUARES // 2
                eval_str = "½-½"
                eval_str_right = True
            elif evaluation[0] == "M":
                b_squares, w_squares = 0, TOTAL_SQUARES
                eval_str = evaluation
                eval_str_right = True
            elif evaluation[0] == "m":
                b_squares, w_squares = TOTAL_SQUARES, 0
                eval_str = evaluation.upper()
                eval_str_right = False
            else:
                evaluation = float(evaluation)
                eval_str_right = evaluation >= 0
                evaluation = abs(evaluation)
                eval_str = (
                    f"{evaluation:.1f}" if evaluation < 10 else f"{evaluation:.0f}"
                )

                b_squares, w_squares = eval_to_emoji_squares(evaluation)
                if not eval_str_right:
                    b_squares, w_squares = w_squares, b_squares

            eval_bar = BLACK_SQUARE * b_squares + WHITE_SQUARE * w_squares
            if eval_str_right:
                eval_bar = eval_bar + eval_str
            else:
                eval_bar = eval_str + eval_bar

            page.keyboard.type(eval_bar, delay=10)

            page.keyboard.press("Enter")

            page.wait_for_timeout(50)

        if best_continuation is not None:
            if best_continuation != "Resign":
                best_continuation = f'"{best_continuation}"'
            if messages and messages[-1].unsent:
                page.keyboard.type(
                    f"Suggested alternative: {best_continuation}", delay=10
                )
            else:
                page.keyboard.type(f"Best continuation: {best_continuation}", delay=10)
            page.keyboard.press("Enter")

            page.wait_for_timeout(50)

        page.keyboard.press("Control+I")

        page.wait_for_timeout(50)

        page.keyboard.type(f"{opening}", delay=10)
        page.keyboard.press("Enter")

        page.wait_for_timeout(50)

        # page.keyboard.press("Control+I")

        # page.wait_for_timeout(50)

        # page.keyboard.type(f"New Elo scale: ~600 median, ~450 average", delay=10)
        # page.keyboard.press("Enter")

        # page.wait_for_timeout(50)

        table_button = page.locator('button:has(svg[icon-name="table-outline"])')
        table_button.wait_for(state="visible", timeout=5000)
        table_button.scroll_into_view_if_needed()
        page.wait_for_timeout(250)
        table_button.click()

        for _ in range(len(counts) - 2):
            table_actions_button = page.locator(
                'button:has(svg[icon-name="overflow-horizontal-outline"]) >> text=Table actions menu'
            )
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
            page.keyboard.type(f"{color_left} ({elo_left})", delay=10)
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
            page.keyboard.type(f"{color_right} ({elo_right})", delay=10)
        page.keyboard.press("Tab")
        page.wait_for_timeout(50)

        for c in HUMANIZED_ORDER:
            if c not in counts:
                continue
            l, r = counts[c]
            label = c.name.replace("_", " ").title()

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
        # page.keyboard.press("Enter")

        if Classification.MEGABLUNDER in counts:
            page.keyboard.type("Megablunder Monday!", delay=5)
            page.wait_for_timeout(50)
            page.keyboard.press("Enter")

        if similar_conversations:
            page.keyboard.type("Similar Games:", delay=5)
            page.wait_for_timeout(50)
            page.keyboard.press("Enter")
            for i, (post_id, score, _) in enumerate(similar_conversations, start=1):
                page.keyboard.type(f"{i}. ", delay=5)

                link_button = page.locator('button:has(svg[icon-name="link-outline"])')
                link_button.wait_for(state="visible", timeout=5000)
                link_button.scroll_into_view_if_needed()
                page.wait_for_timeout(100)
                link_button.click()

                page.wait_for_timeout(100)

                page.keyboard.press("Tab")
                page.wait_for_timeout(100)
                page.keyboard.type(f"{get_post_by_id(post_id).title}", delay=5)
                page.wait_for_timeout(100)
                page.keyboard.press("Tab")
                page.wait_for_timeout(100)
                page.keyboard.type(
                    f"https://www.reddit.com/r/TextingTheory/comments/{post_id}/",
                    delay=5,
                )

                save_link_button = page.get_by_test_id("btn-save-link")
                save_link_button.wait_for(state="visible", timeout=5000)
                save_link_button.scroll_into_view_if_needed()
                page.wait_for_timeout(100)
                save_link_button.click()

                page.wait_for_timeout(50)

                page.keyboard.type(f" ({score:.1%})", delay=5)

                page.wait_for_timeout(50)

                page.keyboard.press("Enter")

        # page.keyboard.press("Enter")

        link_button = page.locator('button:has(svg[icon-name="link-outline"])')
        link_button.wait_for(state="visible", timeout=5000)
        link_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        link_button.click()

        page.wait_for_timeout(100)

        page.keyboard.press("Tab")
        page.wait_for_timeout(100)
        page.keyboard.type("about the bot", delay=5)
        page.wait_for_timeout(100)
        page.keyboard.press("Tab")
        page.wait_for_timeout(100)
        # page.keyboard.type("https://support.chess.com/en/articles/8584089-how-does-game-review-work#h_49f5656333", delay=5) https://www.reddit.com/r/TextingTheory/comments/1kdxh6x/comment/mqefbfm/
        page.keyboard.type(
            "https://www.reddit.com/r/TextingTheory/comments/1k8fed9/utextingtheorybot/",
            delay=5,
        )

        save_link_button = page.get_by_test_id("btn-save-link")
        save_link_button.wait_for(state="visible", timeout=5000)
        save_link_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        save_link_button.click()

        page.wait_for_timeout(50)

        page.keyboard.type(" | ", delay=5)

        page.wait_for_timeout(50)

        link_button = page.locator('button:has(svg[icon-name="link-outline"])')
        link_button.wait_for(state="visible", timeout=5000)
        link_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        link_button.click()

        page.wait_for_timeout(100)

        page.keyboard.press("Tab")
        page.wait_for_timeout(100)
        page.keyboard.type("what do the symbols mean?", delay=5)
        page.wait_for_timeout(100)
        page.keyboard.press("Tab")
        page.wait_for_timeout(100)
        page.keyboard.type(
            "https://www.reddit.com/r/TextingTheory/comments/1ksad21/classifications_badges_explained/",
            delay=5,
        )

        save_link_button = page.get_by_test_id("btn-save-link")
        save_link_button.wait_for(state="visible", timeout=5000)
        save_link_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        save_link_button.click()

        page.wait_for_timeout(50)

        page.keyboard.type(" | ", delay=5)

        page.wait_for_timeout(50)

        link_button = page.locator('button:has(svg[icon-name="link-outline"])')
        link_button.wait_for(state="visible", timeout=5000)
        link_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        link_button.click()

        page.wait_for_timeout(100)

        page.keyboard.press("Tab")
        page.wait_for_timeout(100)
        page.keyboard.type("!annotate", delay=5)
        page.wait_for_timeout(100)
        page.keyboard.press("Tab")
        page.wait_for_timeout(100)
        page.keyboard.type(
            "https://www.reddit.com/r/TextingTheory/comments/1kdxh6x/comment/mqk2jzn/",
            delay=5,
        )

        save_link_button = page.get_by_test_id("btn-save-link")
        save_link_button.wait_for(state="visible", timeout=5000)
        save_link_button.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        save_link_button.click()

        page.wait_for_timeout(100)

        page.keyboard.press("Shift+Home")

        table_button = page.locator('button:has(svg[icon-name="superscript-outline"])')
        table_button.wait_for(state="visible", timeout=5000)
        table_button.scroll_into_view_if_needed()
        page.wait_for_timeout(250)
        table_button.click()

        comment_submit = page.locator('button[slot="submit-button"][type="submit"]')
        comment_submit.wait_for(state="visible", timeout=10000)
        comment_submit.scroll_into_view_if_needed()
        page.wait_for_timeout(100)
        comment_submit.click()

        print("Analysis Posted")

        page.wait_for_timeout(4000)

        # Clean up
        browser.close()


def post_comment_replies(render_queue):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Headful mode
        context = None

        if Path(STORAGE_FILE).exists():
            print("Loading existing session...")
            context = browser.new_context(
                viewport={"width": 1600, "height": 900}, storage_state=STORAGE_FILE
            )
        else:
            assert False

        page = context.new_page()
        for post_id, comment_id, out_path in render_queue:
            try:
                page.goto(
                    f"https://www.reddit.com/r/TextingTheory/comments/{post_id}/comment/{comment_id}/"
                )

                reply_button = page.locator(
                    "button.button-plain-weak:has(svg[icon-name='comment-outline']):has-text('Reply')"
                ).nth(0)
                reply_button.wait_for(state="visible", timeout=5000)
                reply_button.scroll_into_view_if_needed()
                page.wait_for_timeout(100)
                reply_button.click()

                image_button = page.locator(
                    'button:has(svg[icon-name="image-post-outline"])'
                )
                image_button.wait_for(state="visible", timeout=5000)

                with page.expect_file_chooser() as fc_info:
                    image_button.scroll_into_view_if_needed()
                    page.wait_for_timeout(100)
                    image_button.click()

                page.wait_for_timeout(100)

                file_chooser = fc_info.value
                file_chooser.set_files(out_path)

                page.wait_for_timeout(200)

                comment_submit = page.locator(
                    'button[slot="submit-button"][type="submit"]'
                )
                comment_submit.wait_for(state="visible", timeout=5000)
                comment_submit.scroll_into_view_if_needed()
                page.wait_for_timeout(100)
                comment_submit.click()

                print(f"comment replied: {comment_id}")

                page.wait_for_timeout(2000)
            except Exception as e:
                print(f"[!] Failed to post comment reply for {comment_id}: {e}")
                continue

        # Clean up
        browser.close()


def reply_to_comment(comment_id: str, message: str):
    try:
        comment = reddit.comment(id=comment_id)
        comment.reply(message)
        print(f"Replied to comment {comment_id}")
    except Exception as e:
        print(f"[!] Failed to reply to comment {comment_id}: {e}")


def old_handle_top_level(
    cid: str, pid: str, code: str, tmpdir: str, render_queue: list
):

    # 1) fetch analysis JSON
    post_data = get_post_json_from_kv(pid)
    if not post_data:
        reply_to_comment(
            cid,
            "⚠️ Sorry, your `!annotate` request couldn't be processed:\n\n"
            "- No analysis found for current post.\n\n"
            "Please try again after the bot has left an analysis.\n\n"
            "[about !annotate](https://www.reddit.com/r/TextingTheory/comments/1kdxh6x/comment/mqk2jzn/)",
        )
        return

    # 2) re-parse the LLM messages
    msgs = parse_llm_response(post_data, ignore_classifications=True)
    if len(msgs) > 20:
        reply_to_comment(
            cid,
            f"⚠️ Sorry, your `!annotate` request couldn't be processed:\n\n"
            f"- This post has **{len(msgs)} messages**, which exceeds the 20-message limit.\n\n"
            "[about !annotate](https://www.reddit.com/r/TextingTheory/comments/1kdxh6x/comment/mqk2jzn/)",
        )
        return

    # 3) age check
    post = get_post_by_id(pid)
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(
        post.created_utc, tz=timezone.utc
    )
    if age > timedelta(days=7):
        reply_to_comment(
            cid,
            "⚠️ Sorry, your `!annotate` request couldn't be processed:\n\n"
            "- This post is **over 7 days old**.\n\n"
            "[about !annotate](https://www.reddit.com/r/TextingTheory/comments/1kdxh6x/comment/mqk2jzn/)",
        )
        return

    # 4) apply the user’s code
    updated_msgs, err = apply_annotation_code(msgs, code)
    if updated_msgs is None:
        if err == "len":
            err_msg = (
                f"⚠️ Sorry, your `!annotate` request couldn't be processed:\n\n"
                f"- The annotation code doesn't match the number of messages ({len(msgs)}).\n\n"
                "[about !annotate](https://www.reddit.com/r/TextingTheory/comments/1kdxh6x/comment/mqk2jzn/)"
            )
        else:
            err_msg = (
                f"⚠️ Sorry, your `!annotate` request couldn't be processed:\n\n"
                "- The annotation code contains an unexpected character.\n\n"
                "[about !annotate](https://www.reddit.com/r/TextingTheory/comments/1kdxh6x/comment/mqk2jzn/)"
            )
        reply_to_comment(cid, err_msg)
        return

    # 5) render into tmpdir and queue
    out_path = os.path.join(tmpdir, f"{cid}_annotated.png")
    color_left = post_data["color"].get("left")
    color_right = post_data["color"].get("right")
    background = post_data["color"].get("background_hex")

    render_conversation(
        updated_msgs,
        color_data_left=color_left,
        color_data_right=color_right,
        background_hex=background,
        output_path=out_path,
    )
    render_queue.append((pid, cid, out_path))


import re


def extract_display_text(md_text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md_text)
    text = re.sub(r"https?://preview\.redd\.it/[^\s\)]*", "[image]", text)
    # text = re.sub(r'(\*\*|__|\*|_|~~|`|>!|!<|\^)', '', text)
    return text


def handle_annotate(comments_json):
    render_queue = []

    # We open one tempdir for this whole run, so files live until after we reply:
    with tempfile.TemporaryDirectory() as tmpdir:
        for cmd in comments_json:
            cid = cmd["comment_id"]
            pid = cmd["post_id"]
            p_id = cmd["parent_id"]
            body = cmd["text"]
            print(f"Handling !annotate for comment {cid} (parent={p_id}) on post {pid}")

            # parse code
            parts = body.strip().split(maxsplit=2)
            if len(parts) < 2:
                reply_to_comment(
                    cid, "⚠️ Invalid `!annotate` syntax—no code found. Try again."
                )
                continue
            code = parts[1]
            depth = len([ch for ch in code if ch != "-"])
            if depth == 0:
                reply_to_comment(
                    cid, "⚠️ You must supply at least one classification digit."
                )
                continue

            # top‐level case: fall back to your existing flow
            if p_id.startswith("t3_"):
                # … just call your old top‐level logic here, e.g.
                old_handle_top_level(cid, pid, code, tmpdir, render_queue)
                # and continue
                continue

            # if "-" in code:
            #     reply_to_comment(
            #         cid,
            #         "⚠️ Hyphens (`-`) are only allowed in top-level annotations (to flip sides). "
            #         "When annotating a reply-chain, just supply your classification digits."
            #     )
            #     continue

            # otherwise, walk up the reply chain
            chain = []
            try:
                cur = reddit.comment(id=cid)
                while len(chain) < depth:
                    parent = cur.parent()
                    if isinstance(parent, praw.models.Comment):
                        chain.append(parent)
                        cur = parent
                    else:
                        break
            except Exception as e:
                print(cid, f"⚠️ Could not fetch comment chain: {e}")
                continue

            if len(chain) < depth:
                reply_to_comment(
                    cid,
                    f"⚠️ You asked for {depth} annotations but this reply is only {len(chain)} levels deep.",
                )
                continue

            # reverse so the oldest (top‐level) is first, then slice
            chain = list(reversed(chain))[:depth]

            # build messages with username + avatar
            msgs = []
            for i, c in enumerate(chain):
                author = c.author
                text_message = TextMessage(
                    side="right",
                    content=extract_display_text(c.body),
                    classification=None,  # placeholder
                    unsent=False,
                    username=author.name if author else "[deleted]",
                    avatar_url=getattr(author, "icon_img", None),
                )
                if text_message.username == "texting-theory-bot":
                    if text_message.content == "[image]":
                        text_message.content = "[!annotate Result]"
                    elif text_message.content.startswith("**Game Analysis**"):
                        text_message.content = "[Game Analysis]"
                msgs.append(text_message)
                print(f"{msgs[-1].username}: {msgs[-1].content}")

            # apply the code
            updated, err = apply_annotation_code(msgs, code, reply=True)
            if updated is None:
                msg = {
                    "len": "⚠️ Your code's length doesn't match the number of messages.",
                    "char": "⚠️ Your code contains invalid characters.",
                }.get(err, "⚠️ Unknown error.")
                reply_to_comment(cid, msg)
                continue

            # render into tmpdir
            out_path = f"{tmpdir}/{cid}_annotated.png"
            render_reddit_chain(updated, out_path)
            render_queue.append((pid, cid, out_path))

        # now that all files still exist, post your replies
        if render_queue:
            post_comment_replies(render_queue)

    print("All annotate commands handled.")


def extract_image_urls(post):
    image_urls = []
    # Handle direct image posts
    if hasattr(post, "post_hint") and post.post_hint == "image":
        image_urls = [post.url]
    # Handle gallery/slideshow posts
    elif hasattr(post, "gallery_data") and hasattr(post, "media_metadata"):
        for item in post.gallery_data["items"]:
            u = post.media_metadata[item["media_id"]]["s"]["u"].replace("&amp;", "&")
            image_urls.append(u)
    # Handle preview images
    elif hasattr(post, "preview") and "images" in post.preview:
        image_urls = [post.preview["images"][0]["source"]["url"].replace("&amp;", "&")]
    # Handle crossposts
    elif hasattr(post, "crosspost_parent_list") and post.crosspost_parent_list:
        # Use the first crosspost parent
        parent = post.crosspost_parent_list[0]

        # Create a dummy object with the parent's attributes for compatibility
        class Dummy:
            pass

        parent_post = Dummy()
        for k, v in parent.items():
            setattr(parent_post, k, v)
        image_urls = extract_image_urls(parent_post)
    return image_urls


def handle_new_posts(post_id=None):
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

        if any(
            c.author and c.author.name.lower() == reddit.user.me().name.lower()
            for c in post.comments
        ):
            print("Already analyzed")
            continue

        image_urls = extract_image_urls(post)

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

            # stitched = os.path.join(tmpdir, "stitched.jpg")
            out_path = os.path.join(tmpdir, "out.jpg")
            # stitch_images_vertically(input_paths, stitched)
            print(f"Analyzing post with title: {post.title}")
            for attempt in range(2):
                try:
                    data = call_llm_on_image(input_paths, post.title, post.selftext)
                    break
                except Exception as e:
                    print(
                        f"Call llm attempt {attempt + 1} failed due to unexpected error: {e}"
                    )
                    time.sleep(15)

            if data.get("is_convo") is False:
                print("Not a conversation, skipping")
            else:
                elo_left, elo_right = data["elo"].get("left"), data["elo"].get("right")
                color_data_left, color_data_right = data["color"].get("left"), data[
                    "color"
                ].get("right")
                msgs = parse_llm_response(data)
                print("Parsed LLM response")
                render_conversation(
                    msgs,
                    color_data_left,
                    color_data_right,
                    data["color"]["background_hex"],
                    out_path,
                )
                print("Rendered analysis image")

                if any(
                    c.author and c.author.name.lower() == reddit.user.me().name.lower()
                    for c in post.comments
                ):
                    print("Already analyzed")
                    continue

                convo_text = get_convo_str(msgs)
                embedding = get_embedding(convo_text)

                similar_conversations = find_similar_conversations(embedding, post.id)
                if similar_conversations:
                    print("Similar conversations found:")
                    for i, (post_id, score, convo_text) in enumerate(
                        similar_conversations, start=1
                    ):
                        print(f"#{i}: post:{post_id} (score {score:.2f})")
                        print(convo_text[:100])

                for attempt in range(2):
                    try:
                        post_comment_image(
                            post.id,
                            out_path,
                            msgs,
                            (
                                None
                                if color_data_left is None
                                else color_data_left["label"]
                            ),
                            (
                                None
                                if color_data_right is None
                                else color_data_right["label"]
                            ),
                            elo_left,
                            elo_right,
                            data["opening"],
                            similar_conversations,
                            data.get("evaluation"),
                            None,
                            data["coach_insight"],
                        )
                        break
                    except Exception as e:
                        print(
                            f"Post comment image attempt {attempt + 1} failed due to unexpected error: {e}"
                        )
                        time.sleep(15)

                store_post_analysis_json(post.id, data)
                pinecone_insert(post.id, embedding, convo_text)

                # img_url = upload_image_to_imgur(out_path)
                # print("Successfully uploaded to imgur")

                # breakdown = format_counts(msgs, None if color_data_left is None else color_data_left["label"], None if color_data_right is None else color_data_right["label"], elo_left, elo_right)
                # reply = f"**Game Review**\n\n{breakdown}\n\n[**Annotated Analysis**]({img_url})\n\n&nbsp;\n\n[*What do the classifications mean?*](https://support.chess.com/en/articles/8584089-how-does-game-review-work#h_49f5656333)"
                # post.reply(reply)
                # print(f"Commented on post {post.id}")
    print("Ran successfully")
    return "Done", 200
