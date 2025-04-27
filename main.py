import os
from utils import handle_new_posts

if __name__ == "__main__":
    handle_new_posts(os.environ.get("POST_ID"))
