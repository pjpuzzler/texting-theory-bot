import os
from utils import handle_new_posts

if __name__ == "__main__":
    post_id = os.environ.get("POST_ID")
    if post_id:
        print(f'Got request to analyze post {post_id}')
        handle_new_posts(post_id)
    else:
        print('Got request to analyze recent posts')
        handle_new_posts()
