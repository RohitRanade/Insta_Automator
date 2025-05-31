Insta_Automator
===============

Overview
--------
Insta_Automator is a Python-based tool designed to automate the creation of specific type of Instagram Reels. It combines images, audio clips, and text overlays to generate engaging videos, streamlining your content creation process.

Table of Contents
-----------------
- Features
- Prerequisites
- Installation
- Configuration
  - Cloudinary Setup
  - Facebook Access Token
- Usage
- File Structure
- Troubleshooting
- License
- Acknowledgments

Features
--------
- Automated video creation combining images, audio, and text.
- Text overlays using specific fonts.
- Incorporation of multiple audio clips.
- Logs tasks for scheduled uploads.

Prerequisites
-------------
Before you begin, ensure you have the following installed:
- Python 3.6 or higher: https://www.python.org/downloads/
- pip (Python package installer)
- Git: https://git-scm.com/downloads

Installation
------------
1. Clone the Repository:
   Open your terminal or command prompt and run:
   git clone https://github.com/RohitRanade/Insta_Automator.git

   Navigate to the project directory:
   cd Insta_Automator

2. Create a Virtual Environment (Optional but Recommended):
   python -m venv venv

   Activate the virtual environment:
   - Windows:
     venv\Scripts\activate
   - macOS/Linux:
     source venv/bin/activate

3. Install Required Packages:
   pip install -r requirements.txt

   Note: If 'requirements.txt' is not present, you may need to manually install dependencies as you encounter errors.

Configuration
-------------
Cloudinary Setup:
1. Create a Cloudinary Account:
   - Visit https://cloudinary.com/ and sign up for a free account.

2. Retrieve API Credentials:
   - After logging in, navigate to the Dashboard.
   - Locate your Cloud Name, API Key, and API Secret.

3. Set Environment Variables:
   - Windows:
     set CLOUDINARY_CLOUD_NAME=your_cloud_name
     set CLOUDINARY_API_KEY=your_api_key
     set CLOUDINARY_API_SECRET=your_api_secret

   - macOS/Linux:
     export CLOUDINARY_CLOUD_NAME=your_cloud_name
     export CLOUDINARY_API_KEY=your_api_key
     export CLOUDINARY_API_SECRET=your_api_secret

   Replace 'your_cloud_name', 'your_api_key', and 'your_api_secret' with the values from your Cloudinary dashboard.

   Alternatively, create a '.env' file in the project directory and add:
   CLOUDINARY_CLOUD_NAME=your_cloud_name
   CLOUDINARY_API_KEY=your_api_key
   CLOUDINARY_API_SECRET=your_api_secret

   Ensure to load these variables in your script using a library like 'python-dotenv'.

Facebook Access Token:
1. Create a Facebook Developer Account:
   - Visit https://developers.facebook.com/ and log in with your Facebook account.
   - Click on 'My Apps' > 'Create App'.
   - Choose an app type (e.g., "Business") and provide the necessary details.

2. Set Up Facebook Login:
   - In your app dashboard, navigate to 'Add Product' > 'Facebook Login' > 'Set Up'.
   - Configure the settings as required.

3. Generate Access Token:
   - Go to the Graph API Explorer: https://developers.facebook.com/tools/explorer/
   - Select your app from the top-right dropdown.
   - Click on 'Get Token' > 'Get User Access Token'.
   - In the permissions window, select the necessary permissions (e.g., 'pages_manage_posts', 'pages_read_engagement').
   - Click 'Generate Access Token'.
   - Copy the generated token.

   Note: For long-term use, consider generating a long-lived access token. Refer to Facebook's documentation for more details.

4. Set Environment Variable:
   - Windows:
     set FACEBOOK_ACCESS_TOKEN=your_access_token

   - macOS/Linux:
     export FACEBOOK_ACCESS_TOKEN=your_access_token

   Replace 'your_access_token' with the token you obtained.

   Alternatively, add it to your '.env' file:
   FACEBOOK_ACCESS_TOKEN=your_access_token

Usage
-----
1. Prepare Your Media:
   - Images: Place your images in a designated folder.
   - Audio: Ensure audio files (e.g., 'audio1cor.mp3', 'audio2cor.mp3', etc.) are present in the project directory.
   - Fonts: Make sure the font files ('NotoSansDevanagari_ExtraCondensed-Medium.ttf', 'NotoSerifDevanagari_ExtraCondensed-Bold.ttf') are in the project directory.

2. Run the Script:
   Execute the main script:
   python UltimateUploader.py

   This will generate a video ('final_reel_video.mp4') combining your images, audio, and text overlays.

3. Upload to Instagram:
   Currently, the script prepares the video. Uploading to Instagram must be done manually. Future updates may include automated uploading features.

File Structure
--------------
- 'UltimateUploader.py': Main script that orchestrates video creation.
- 'audio1cor.mp3' to 'audio6cor.mp3': Audio clips used in the video.
- 'NotoSansDevanagari_ExtraCondensed-Medium.ttf', 'NotoSerifDevanagari_ExtraCondensed-Bold.ttf': Font files for text overlays.
- 'final_reel_video.mp4': Output video file.
- 'last_image_index.txt': Tracks the last used image to avoid repetition.
- 'task_scheduler_log.txt': Logs scheduled tasks (if any).
- 'used_shlokas.txt': Records used text overlays to prevent duplicates.

Troubleshooting
---------------
- Missing Modules:
  If you encounter 'ModuleNotFoundError', install the missing module using pip:
  pip install module_name

- Permission Issues:
  Ensure you have the necessary permissions to read/write files in the project directory.

- Unsupported Media Formats:
  Verify that your images and audio files are in supported formats (e.g., '.jpg', '.png', '.mp3').

- Invalid Access Token:
  If you receive authentication errors, ensure your Facebook access token is valid and has not expired.

- Cloudinary Upload Issues:
  Double-check your Cloudinary credentials and ensure they are correctly set as environment variables.

License
-------
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
---------------
Thanks to RohitRanade for creating this project.

Feel free to contribute to this project by submitting issues or pull requests.
