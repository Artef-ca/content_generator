import os
from google import genai
from google.genai import types
import time


PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "mobily-genai")
# Veo 3.1 is typically restricted to "us-central1" or "europe-west4" 
# "global" often fails for video generation models.
LOCATION = "us-central1" 
BUCKET_NAME = os.getenv("GCS_BUCKET", "content_creation_data")

client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location=LOCATION
)


def generate_veo_video(prompt: str):
    print(f"🎬 Starting video generation for: {prompt[:50]}...")
    
    try:
        operation = client.models.generate_videos(
            model="veo-3.1-generate-001",
            prompt=prompt,
            config=types.GenerateVideosConfig(
                output_gcs_uri=f"gs://{BUCKET_NAME}/videos/",
                aspect_ratio="16:9"
            )
        )

        # Poll the operation until it completes
        print(f"⏳ Operation started: {operation.name}")
        while not operation.done:
            print("   ...still generating, checking again in 20s...")
            time.sleep(20)
            operation = client.operations.get(operation)

        # Check for errors
        if operation.error:
            print(f"❌ Generation failed: {operation.error}")
            return None

        # Now the result should be populated
        if operation.response and operation.response.generated_videos:
            video_uri = operation.response.generated_videos[0].video.uri
            print(f"✅ Success! Video saved to GCS: {video_uri}")
            return video_uri
        else:
            print("⚠️ Operation completed but no video was returned.")
            print(f"Full response: {operation.response}")
            return None

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return None

if __name__ == "__main__":
    my_prompt = "A futuristic desert city with solar panels reflecting a neon sunset, cinematic drone shot."
    generate_veo_video(my_prompt)