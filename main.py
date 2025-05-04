import os
import json
from utils import handle_new_posts, handle_annotate

if __name__ == "__main__":
    post_id = os.environ.get("POST_ID")
    comments_json = os.environ.get("ANNOTATE_COMMENTS")

    if post_id:
        print(f'Got request to analyze post {post_id}')
        handle_new_posts(post_id)
    elif comments_json and comments_json.strip().startswith('['):
        print(f'Got request to annotate comments')
        handle_annotate(json.loads(comments_json))
    else:
        print('Got request to analyze recent posts')
        handle_new_posts()
