from apism import YouTubeAPI
import datetime
from tqdm import tqdm

import time
import random

from keys import YOUTUBE_KEY

from utils.gcs import upload_and_delete_local_file

# Set query and parameters
video_params = {"part": "id,statistics,topicDetails,contentDetails"}
comments_params = {"part": "id,replies,snippet", "order": "time"}
bucket_name = "youtube-us-tariffs2"


async def collect_data(query, yyyy=2025, mm=1, dd=1):
    # Define the start and end dates
    start_date = datetime.date(yyyy, mm, dd)
    end_date = datetime.date(2025, 5, 31)

    # Calculate the total number of days for the progress bar
    total_days = (end_date - start_date).days

    # Initialize tqdm progress bar
    with tqdm(total=total_days, desc=f"Processing {start_date}") as pbar:
        # Loop through the dates
        current_date = start_date

        while current_date <= end_date:
            # Format the date as strings for 'publishedAfter' and 'publishedBefore'
            published_after = current_date.strftime("%Y-%m-%d") + "T00:00:00Z"
            published_before = current_date.strftime("%Y-%m-%d") + "T23:59:59Z"

            # Update the search parameters with the current date
            search_params = {
                "part": "snippet",
                "type": "video",
                "maxResults": 30,
                "relevanceLanguage": "en",
                "publishedAfter": published_after,
                "publishedBefore": published_before,
                "order": "viewCount",
            }

            custom_params = {
                "search": search_params,
                "videos": video_params,
                "commentThreads": comments_params,
            }

            # Search API -> Videos API -> CommentThreads API -> Transcript API
            yt = YouTubeAPI(api_key=YOUTUBE_KEY, params=custom_params, retry_delay=600)
            await yt.search(query)
            await yt.videos()
            await yt.comment_threads()
            time.sleep(random.uniform(30, 60))
            await yt.transcript(batch_delay=random.uniform(10, 30), batch_size=5)

            # Process or store the results as needed
            yt.to_csv(default_cols=True, shorten_cols=True, force_output=True)
            foldername = "-".join(query.split()).lower()
            for filename in [
                "search.csv",
                "videos.csv",
                "commentThreads.csv",
                "commentThreadsreplies.csv",
                "transcripts.csv",
            ]:
                upload_and_delete_local_file(
                    bucket_name,
                    filename,
                    f"{foldername}/{current_date.strftime('%Y-%m-%d')}/{filename}",
                )

            # Move to the next day
            current_date += datetime.timedelta(days=1)

            # Update progress bar description with the current date
            pbar.set_description(f"Processing {current_date.strftime('%Y-%m-%d')}")

            # Update the progress bar
            pbar.update(1)
