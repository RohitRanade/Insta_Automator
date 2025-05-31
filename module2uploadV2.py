import requests
import time
import cloudinary
import cloudinary.uploader
import cloudinary.api

# === Cloudinary Configuration ===
cloudinary.config(
   cloud_name='dsf9ilqvu',        # ← replace with your Cloudinary cloud name
    api_key='921713938293983',              # ← replace with your API key
    api_secret='Xgv82dnt9kw4gKoW1VUEOporzPA'        # ← replace with your API secret
)

# === General Configuration ===
VIDEO_FILE_PATH = "C:\\Users\\91871\\Downloads\\updatedbandana-9.mp4"
IG_USER_ID = "17841471913351721"
ACCESS_TOKEN = "EAAJ0yJ4E0koBO0DWwi9dRyGJYW8gVKYr2pWrh6HbXsxTMZAe3DRFryGasZBYeTZBSbbGvQN8bPHwZBtD1rFQRs0uUH88JIv4Cki4ozadZCFyAkEnQf74g69H024iokfPXEK3ASeBGkKWLkAaMdg2ZCPaRRltZCkb3Ec4mCb7ylBSassuO7tbYRIX49TjIaL"

# === 1. Upload to Cloudinary ===
def upload_to_cloudinary(file_path):
    try:
        response = cloudinary.uploader.upload_large(
            file_path,
            resource_type="video"
        )
        print("✅ Uploaded to Cloudinary.")
        return response["secure_url"], response["public_id"]
    except Exception as e:
        print("❌ Cloudinary upload failed:", e)
        return None, None

# === 2. Wait for Instagram media to be ready ===
def wait_until_ready(container_id, max_wait=3000):
    check_url = f"https://graph.facebook.com/v22.0/{container_id}?fields=status_code&access_token={ACCESS_TOKEN}"
    waited = 0

    while waited < max_wait:
        res = requests.get(check_url)
        if res.ok:
            data = res.json()
            status = data.get("status_code")
            print(f"⏳ Status check [{waited}s]: {status}")
            if status == "FINISHED":
                return True
            elif status == "ERROR":
                print("❌ Error from Instagram:", data)
                return False
        else:
            print("⚠️ Failed to check status:", res.text)
            return False
        time.sleep(5)
        waited += 5

    print("⏱ Timeout: Media still IN_PROGRESS after waiting.")
    return False

# === 3. Upload video to Instagram ===
def upload_reel_to_instagram(video_url, cloudinary_public_id):
    container_url = f"https://graph.facebook.com/v22.0/{IG_USER_ID}/media"
    container_payload = {
        'media_type': 'REELS',
        'video_url': video_url,
        'caption': 'This is an auto-posted reel via Cloudinary 🚀',
        'access_token': ACCESS_TOKEN
    }
    container_response = requests.post(container_url, data=container_payload)
    if not container_response.ok:
        print("❌ Failed to create container:", container_response.text)
        return

    container_id = container_response.json().get("id")
    print("✅ Container Created:", container_id)

    # Wait until media is ready
    print("⏳ Waiting for media to finish processing...")
    if not wait_until_ready(container_id):
        print("❌ Media processing timeout. Try again later.")
        return

    publish_url = f"https://graph.facebook.com/v22.0/{IG_USER_ID}/media_publish"
    publish_payload = {
        'creation_id': container_id,
        'access_token': ACCESS_TOKEN
    }

    publish_response = requests.post(publish_url, data=publish_payload)
    if publish_response.ok:
        print("✅ Reel published successfully!")
        print("Response:", publish_response.json())

        # === Delete media from Cloudinary after publishing ===
        try:
            cloudinary.api.delete_resources([cloudinary_public_id], resource_type="video")
            print("❌ Deleted video from Cloudinary after publishing.")
        except Exception as e:
            print("⚠️ Failed to delete Cloudinary video:", e)
    else:
        print("❌ Failed to publish:", publish_response.text)

# === MAIN ===
if __name__ == "__main__":
    print("📤 Uploading video to Cloudinary...")
    direct_video_url, public_id = upload_to_cloudinary(VIDEO_FILE_PATH)

    if direct_video_url:
        print("🔗 Cloudinary URL:", direct_video_url)
        print("📸 Uploading reel to Instagram...")
        upload_reel_to_instagram(direct_video_url, public_id)
    else:
        print("⚠️ Upload failed, cannot continue.")