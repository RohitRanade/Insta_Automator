import os
import subprocess

import shlex
import uharfbuzz as hb
import freetype
from PIL import Image, ImageDraw
import re
from io import BytesIO
import random
import time
import requests # For uploader part
import cloudinary # For uploader part
import cloudinary.uploader # For uploader part
import cloudinary.api # For uploader part

# --- Module 1: Video Creator Configurations & Functions ---

# SDKs for Gemini (Text Generation)
import google.generativeai as gemini_text_sdk


API_KEY = "AIzaSyDEKsW0GhR5pBHBcDVDz3xZSfZD1lkAvVM"
if not API_KEY:
    print("Error: GEMINI_API_KEY not found in .env or environment. Please set it for text generation.")
    exit()
gemini_text_sdk.configure(api_key=API_KEY)

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_DURATION_SECONDS = 10
VIDEO_FPS = 25

FONT_FILE_PATH_SANSKRIT = "NotoSerifDevanagari_ExtraCondensed-Bold.ttf"
FONT_FILE_PATH_HINDI = "NotoSerifDevanagari_ExtraCondensed-Bold.ttf"
FONT_FILE_PATH_ENGLISH = "NotoSerifDevanagari_ExtraCondensed-Bold.ttf" # Update if needed

TEXT_COLOR_RGBA = (255, 255, 255, 255)
TARGET_TEXT_WIDTH_PERCENT = 0.88
LINE_SPACING_FACTOR = 1.2
SPACE_BETWEEN_BLOCKS_PT = 15

BASE_FONT_SIZE_SANSKRIT = 60
MIN_FONT_SIZE_SANSKRIT = 38
IDEAL_LENGTH_FOR_BASE_SANSKRIT = 60
MAX_LENGTH_FOR_MIN_SANSKRIT = 150
BASE_FONT_SIZE_HINDI = 52
MIN_FONT_SIZE_HINDI = 32
IDEAL_LENGTH_FOR_BASE_HINDI = 90
MAX_LENGTH_FOR_MIN_HINDI = 220
BASE_FONT_SIZE_ENGLISH = 48
MIN_FONT_SIZE_ENGLISH = 28
IDEAL_LENGTH_FOR_BASE_ENGLISH = 120
MAX_LENGTH_FOR_MIN_ENGLISH = 280

PREEXISTING_BACKGROUND_IMAGE_FOLDER = "Images" # Make sure this folder exists and has images
ALLOWED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
AI_BACKGROUND_IMAGE_FILTER_OPACITY = 0.5
FILTERED_BACKGROUND_FOR_VIDEO_FILENAME = "filtered_background_for_video.png" # Renamed for clarity

USED_SHLOKAS_FILE = "used_shlokas.txt"
MAX_SHLOKA_RETRIES = 5

RENDERED_TEXT_PNG = "rendered_gemini_text_overlay.png"
FINAL_OUTPUT_VIDEO_FILENAME = "final_reel_video.mp4" # Filename for the created video

BACKGROUND_MUSIC_FILES_LIST = [
    "audio1cor.mp3", "audio2cor.mp3", "audio3cor.mp3",
    "audio4cor.mp3", "audio5cor.mp3", "audio6cor.mp3", "audio7cor.mp3"
]

MAX_OVERALL_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 60

# --- Module 2: Uploader Configurations ---
# Cloudinary Configuration (Best to use environment variables for these)
CLOUDINARY_CLOUD_NAME = 'dsf9ilqvu'
CLOUDINARY_API_KEY = '921713938293983'
CLOUDINARY_API_SECRET = 'Xgv82dnt9kw4gKoW1VUEOporzPA'

cloudinary.config(
   cloud_name=CLOUDINARY_CLOUD_NAME,
   api_key=CLOUDINARY_API_KEY,
   api_secret=CLOUDINARY_API_SECRET
)

# Instagram Graph API Configuration (Best to use environment variables)
IG_USER_ID = "17841471913351721"
ACCESS_TOKEN = "EAAJ0yJ4E0koBO0DWwi9dRyGJYW8gVKYr2pWrh6HbXsxTMZAe3DRFryGasZBYeTZBSbbGvQN8bPHwZBtD1rFQRs0uUH88JIv4Cki4ozadZCFyAkEnQf74g69H024iokfPXEK3ASeBGkKWLkAaMdg2ZCPaRRltZCkb3Ec4mCb7ylBSassuO7tbYRIX49TjIaL" # Ensure this is a long-lived token with necessary permissions

# Hashtags
HASHTAGS_LIST = [
    "#bhagavadgita", "#spiritualgrowth", "#mindfulness", "#karmayoga",
    "#lifelessons", "#instagramwisdom", "#innerpeace", "#purposedriven",
    "#motivationdaily", "#selfimprovement", "#motherlove", "#birthplacevibes",
    "#emotionalconnection", "#rootsandwings", "#heavenonearth", "#instaquotes",
    "#viralreels", "#indianroots", "#hearttouching", "#reelitfeelit"
]


# --- Helper Functions from Module 1 (Video Creator) ---
def _apply_black_filter_to_pil_image(pil_image: Image.Image, opacity: float) -> Image.Image:
    if not (0 <= opacity <= 1): opacity = max(0, min(1, opacity))
    if opacity == 0: return pil_image.convert('RGB') if pil_image.mode != 'RGB' else pil_image
    base_image_rgba = pil_image.convert('RGBA') if pil_image.mode != 'RGBA' else pil_image.copy()
    alpha_value = int(255 * opacity)
    black_layer = Image.new('RGBA', base_image_rgba.size, (0, 0, 0, alpha_value))
    return Image.alpha_composite(base_image_rgba, black_layer).convert('RGB')

def get_gemini_text_response(prompt: str) -> str:
    print(f"Fetching text response from Gemini for prompt: '{prompt}'")
    try:
        model = gemini_text_sdk.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        if response.parts: return "".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
        if hasattr(response, 'text') and response.text: return response.text.strip()
        print("Gemini text response did not contain text."); return "Error: No text in Gemini response."
    except Exception as e: print(f"Error fetching text from Gemini: {e}"); return f"Error: Could not fetch text. {e}"

def normalize_shloka_for_comparison(shloka_text: str) -> str:
    if not shloka_text: return ""
    text = shloka_text.strip()
    if text.startswith('"') and text.endswith('"'): text = text[1:-1]
    text = text.replace("\n", " ")
    return " ".join(text.split()).strip().lower()

def load_used_shlokas(filepath: str) -> set:
    used = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: used.update(line.strip() for line in f)
        except IOError as e: print(f"Warning: Could not read used shlokas file '{filepath}': {e}")
    return used

def add_shloka_to_used(shloka_key: str, used_shlokas_set: set, filepath: str):
    if shloka_key not in used_shlokas_set:
        used_shlokas_set.add(shloka_key)
        try:
            with open(filepath, 'a', encoding='utf-8') as f: f.write(shloka_key + "\n")
            print(f"Added shloka key '{shloka_key[:50]}...' to used list.")
        except IOError as e: print(f"Warning: Could not write to used shlokas file '{filepath}': {e}")

def split_shloka_hindi_english(full_text: str) -> tuple[str, str, str]:
    sanskrit, hindi, english = "", "", ""
    text = full_text.strip()
    eng_patterns = [r"(.*?)English Meaning:(.*)", r"(.*?)Eng Meaning:(.*)", r"(.*?)English Translation:(.*)", r"(.*?)English:(.*)"]
    text_before_eng = text
    for p in eng_patterns:
        m = re.search(p, text, re.DOTALL | re.IGNORECASE)
        if m: text_before_eng = m.group(1).strip(); english = re.sub(r"^(English Meaning:|Eng Meaning:|English Translation:|English:)\s*", "", m.group(2).strip(), flags=re.IGNORECASE).strip(); break
    hindi_patterns = [r"(.*?)हिंदी अर्थ:(.*)", r"(.*?)Hindi Meaning:(.*)", r"(.*?)अर्थ:(.*)", r"(.*?)Meaning:(.*)"]
    text_before_hindi = text_before_eng
    for p in hindi_patterns:
        m = re.search(p, text_before_eng, re.DOTALL | re.IGNORECASE)
        if m: text_before_hindi = m.group(1).strip(); hindi = re.sub(r"^(हिंदी अर्थ:|Hindi Meaning:|अर्थ:|Meaning:)\s*", "", m.group(2).strip(), flags=re.IGNORECASE).strip(); break
    sanskrit = text_before_hindi.strip()
    sanskrit_prefix_patterns = [r"^(श्लोकः|Shloka:|Shlokah:|Shlok:|Sloka:|Verse:)\s*", r"^\*\*\s*(श्लोकः|Shloka:|Shlokah:|Shlok:|Sloka:|Verse:)\s*\*\*\s*"]
    for prefix_pattern in sanskrit_prefix_patterns: sanskrit = re.sub(prefix_pattern, "", sanskrit, flags=re.IGNORECASE | re.UNICODE).strip()
    
    processed_texts = {}
    for text_var_str in ['sanskrit', 'hindi', 'english']:
        current_text = locals()[text_var_str]
        if current_text:
            current_text = re.sub(r"^\s*[\*#-_]+\s*|\s*[\*#-_]+\s*$", "", current_text).strip()
            current_text = re.sub(r"^\s*[\*]{2,}\s*(.*?)\s*[\*]{2,}\s*$", r"\1", current_text).strip()
            processed_texts[text_var_str] = current_text
        else:
            processed_texts[text_var_str] = ""
            
    sanskrit, hindi, english = processed_texts['sanskrit'], processed_texts['hindi'], processed_texts['english']
    if not sanskrit.strip() and (hindi.strip() or english.strip()): print("Warning: Parsed Sanskrit shloka is empty after keyword removal.")
    print(f"--- Cleaned Sanskrit Part ---\n{sanskrit}\n--------------------")
    print(f"--- Cleaned Hindi Meaning Part ---\n{hindi}\n-------------------")
    print(f"--- Cleaned English Meaning Part ---\n{english}\n-----------------")
    return sanskrit, hindi, english

def calculate_dynamic_font_size(text_length: int, base, min_size, ideal_len, max_len) -> int:
    if text_length <= 0: return base
    if text_length <= ideal_len: return base
    if text_length >= max_len: return min_size
    if max_len <= ideal_len: return min_size
    prog = (text_length - ideal_len) / (max_len - ideal_len)
    reduct = (base - min_size) * prog
    return max(min_size, int(base - reduct))

def shape_text_line_hb(hb_font, text_segment):
    buf = hb.Buffer(); buf.add_str(text_segment); buf.guess_segment_properties()
    features = {"kern": True, "liga": True, "clig": True, "calt": True}
    try: deva_tag = hb.ot_tag_to_script('deva')
    except TypeError: deva_tag = hb.ot_tag_to_script(b'deva')
    if buf.script == deva_tag:
        features.update({"akhn": True, "rkrf": True, "blwf": True, "half": True, "pstf": True, "vatu": True, "pres": True, "abvs": True, "blws": True, "psts": True, "haln": True, "locl": True, "nukt": True, "pref": True, "cjct": True})
    hb.shape(hb_font, buf, features)
    return buf.glyph_infos, buf.glyph_positions

def get_line_width_px(glyph_positions): return sum(p.x_advance for p in glyph_positions) // 64

def wrap_and_shape_text_block(text_content: str, ft_face, hb_font, font_size, max_width, line_factor):
    if not text_content.strip(): return [], 0, 0
    ft_face.set_char_size(font_size * 64); hb_font.scale = (font_size * 64, font_size * 64)
    lines = text_content.split('\n'); all_data = []; max_block_w = 0
    for segment in lines:
        words = segment.split(); cur_text = ""; cur_infos, cur_pos = None, None
        for word in words:
            if not word: continue
            test_word = (" " if cur_text else "") + word; pot_text = cur_text + test_word
            pot_infos, pot_pos = shape_text_line_hb(hb_font, pot_text)
            pot_w = get_line_width_px(pot_pos)
            if pot_w <= max_width or not cur_text:
                cur_text, cur_infos, cur_pos = pot_text, pot_infos, pot_pos
            else:
                if cur_infos: data = (cur_infos, cur_pos, get_line_width_px(cur_pos)); all_data.append(data); max_block_w = max(max_block_w, data[2])
                cur_text = word; cur_infos, cur_pos = shape_text_line_hb(hb_font, cur_text)
        if cur_text and cur_infos: data = (cur_infos, cur_pos, get_line_width_px(cur_pos)); all_data.append(data); max_block_w = max(max_block_w, data[2])
    line_h_px = int(ft_face.size.height // 64 * line_factor)
    total_h = len(all_data) * line_h_px
    if all_data and len(all_data) > 1: total_h -= int(ft_face.size.height // 64 * (line_factor - 1.0))
    return all_data, max_block_w, total_h if all_data else 0

def render_text_to_png_manual_v3(skt_txt, hin_txt, eng_txt, out_path, skt_fs, hin_fs, eng_fs):
    print(f"Rendering Skt({skt_fs}pt), Hin({hin_fs}pt), Eng({eng_fs}pt) to {out_path}...")
    fonts = {"sanskrit": FONT_FILE_PATH_SANSKRIT, "hindi": FONT_FILE_PATH_HINDI, "english": FONT_FILE_PATH_ENGLISH}
    for lang, p in fonts.items():
        if not os.path.exists(p): print(f"ERROR: Font for {lang} not found: '{p}'"); return False
    try:
        ft_f = {lang: freetype.Face(p) for lang, p in fonts.items()}
        hb_f = {lang: hb.Font(hb.Face(open(p, 'rb').read())) for lang, p in fonts.items()}
        max_w_px = int(VIDEO_WIDTH * TARGET_TEXT_WIDTH_PERCENT)
        skt_d, skt_w, skt_h = wrap_and_shape_text_block(skt_txt, ft_f["sanskrit"], hb_f["sanskrit"], skt_fs, max_w_px, LINE_SPACING_FACTOR)
        hin_d, hin_w, hin_h = wrap_and_shape_text_block(hin_txt, ft_f["hindi"], hb_f["hindi"], hin_fs, max_w_px, LINE_SPACING_FACTOR)
        eng_d, eng_w, eng_h = wrap_and_shape_text_block(eng_txt, ft_f["english"], hb_f["english"], eng_fs, max_w_px, LINE_SPACING_FACTOR)
        canv_w = max(skt_w, hin_w, eng_w, 1) + 40
        space_px = int((SPACE_BETWEEN_BLOCKS_PT / 72) * 96 * (max(skt_fs, hin_fs, eng_fs, 12) / 12.0))
        blocks = sum(1 for h in [skt_h, hin_h, eng_h] if h > 0)
        total_space = space_px * (blocks - 1) if blocks > 1 else 0
        canv_h = skt_h + hin_h + eng_h + total_space + 40
        if canv_h <= 40: print("No text content with height to render."); return False
        img = Image.new('RGBA', (int(canv_w), int(canv_h)), (0,0,0,0)); cur_y = 20
        to_render = [(skt_txt, skt_d, ft_f["sanskrit"], skt_fs, skt_h), (hin_txt, hin_d, ft_f["hindi"], hin_fs, hin_h), (eng_txt, eng_d, ft_f["english"], eng_fs, eng_h)]
        first_done = False
        for txt, data, ft, fs, bh in to_render:
            if bh > 0:
                if first_done: cur_y += space_px
                ft.set_char_size(fs * 64)
                line_h = int(ft.size.height // 64 * LINE_SPACING_FACTOR)
                asc = (ft.ascender / ft.units_per_EM) * fs if ft.units_per_EM else (ft.size.ascender // 64)
                y_base = cur_y + asc
                for infos, pos_list, line_w in data:
                    x_pen = (canv_w - line_w) // 2
                    for info, pos_item in zip(infos, pos_list):
                        ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
                        bmp = ft.glyph.bitmap
                        xr, yr = x_pen + (pos_item.x_offset//64) + ft.glyph.bitmap_left, y_base - ft.glyph.bitmap_top + (pos_item.y_offset//64)
                        if bmp.width > 0 and bmp.rows > 0:
                            g_data = [(TEXT_COLOR_RGBA[0], TEXT_COLOR_RGBA[1], TEXT_COLOR_RGBA[2], val) for val in bmp.buffer]
                            g_img = Image.new("RGBA", (bmp.width, bmp.rows)); g_img.putdata(g_data)
                            img.paste(g_img, (int(xr), int(yr)), mask=g_img)
                        x_pen += pos_item.x_advance // 64
                    y_base += line_h
                cur_y += bh; first_done = True
        img.save(out_path); print(f"Successfully rendered text to {out_path}"); return True
    except Exception as e: print(f"Error during text rendering: {e}"); import traceback; traceback.print_exc(); return False

def create_video_ffmpeg(bg_path, overlay_path, music_path, out_path):
    if not os.path.exists(bg_path): print(f"ERROR: BG not found: {bg_path}"); return False
    if not os.path.exists(overlay_path): print(f"ERROR: Overlay not found: {overlay_path}"); return False
    inputs = ['-loop', '1', '-framerate', str(VIDEO_FPS), '-i', bg_path.replace('\\', '/'), '-i', overlay_path.replace('\\', '/')]
    bg_filter = f"[0:v]scale='iw*max({VIDEO_WIDTH}/iw,{VIDEO_HEIGHT}/ih)':'ih*max({VIDEO_WIDTH}/iw,{VIDEO_HEIGHT}/ih)',crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},setsar=1[bg_scaled]"
    filters = [bg_filter, f"[bg_scaled][1:v]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2[outv]"]
    maps = ['-map', '[outv]']; opts = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', str(VIDEO_FPS), '-t', str(VIDEO_DURATION_SECONDS)]; final_opts = []
    if music_path and os.path.exists(str(music_path).replace('\\', '/')):
        inputs.extend(['-i', str(music_path).replace('\\', '/')]); maps.extend(['-map', '2:a?']); opts.extend(['-c:a', 'aac', '-b:a', '192k']); final_opts.append('-shortest')
    else: print(f"Warning: Music file '{music_path}' not found or not specified.")
    cmd = ['ffmpeg'] + inputs + ['-filter_complex', ";".join(filters)] + maps + opts + final_opts + ['-y', out_path]
    print(f"FFmpeg: {' '.join(shlex.quote(str(c)) for c in cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Video created: {out_path}")
        try: # Verify duration
            probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', out_path]
            actual_dur = float(subprocess.check_output(probe_cmd, text=True).strip())
            print(f"Verified duration: {actual_dur:.2f}s.")
            if abs(actual_dur - float(VIDEO_DURATION_SECONDS)) > 0.5 and not (music_path and actual_dur < float(VIDEO_DURATION_SECONDS)):
                 print(f"WARNING: Duration ({actual_dur:.2f}s) differs from target ({VIDEO_DURATION_SECONDS}s).")
        except Exception as e_probe: print(f"Could not verify duration: {e_probe}")
        return True
    except subprocess.CalledProcessError as e: print(f"FFmpeg error: {e}\n{e.stdout}\n{e.stderr}"); return False
    except FileNotFoundError: print("Error: ffmpeg/ffprobe not found."); return False


# --- Functions from Module 2 (Uploader) - Unchanged as per request, just integrated ---
def upload_to_cloudinary(file_path):
    if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
        print("❌ Cloudinary credentials not fully configured. Skipping upload.")
        return None, None
    try:
        response = cloudinary.uploader.upload_large(file_path, resource_type="video")
        print("✅ Uploaded to Cloudinary.")
        return response["secure_url"], response["public_id"]
    except Exception as e: print(f"❌ Cloudinary upload failed: {e}"); return None, None

def wait_until_ready(container_id, max_wait=300): # Reduced max_wait for faster testing if needed
    check_url = f"https://graph.facebook.com/v19.0/{container_id}?fields=status_code&access_token={ACCESS_TOKEN}" # Updated API version
    waited = 0; poll_interval = 10 # Increased poll interval slightly
    while waited < max_wait:
        print(f"⏳ Checking IG container status... attempt {waited//poll_interval + 1}")
        res = requests.get(check_url)
        if res.ok:
            data = res.json(); status = data.get("status_code")
            print(f"⏳ Status: {status}")
            if status == "FINISHED": return True
            if status == "ERROR": print(f"❌ Error from Instagram: {data}"); return False
        else: print(f"⚠️ Failed to check status: {res.status_code} {res.text}"); return False # More info on fail
        time.sleep(poll_interval); waited += poll_interval
    print("⏱ Timeout: Media still IN_PROGRESS."); return False

def upload_reel_to_instagram(video_url_for_ig, cloudinary_public_id_to_delete, caption_for_ig): # Added caption
    if not all([IG_USER_ID, ACCESS_TOKEN]):
        print("❌ Instagram User ID or Access Token not configured. Skipping Instagram upload.")
        return False
        
    container_url = f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media" # Updated API version
    container_payload = {
        'media_type': 'REELS',
        'video_url': video_url_for_ig,
        'caption': caption_for_ig, # Use the passed caption
        'share_to_feed': True, # Reels specific, usually true
        'access_token': ACCESS_TOKEN
    }
    print("📤 Creating Instagram media container...")
    container_response = requests.post(container_url, data=container_payload)
    if not container_response.ok: print(f"❌ Failed to create container: {container_response.text}"); return False
    container_id = container_response.json().get("id")
    print(f"✅ Container Created: {container_id}")
    if not wait_until_ready(container_id): print("❌ Media processing failed or timed out."); return False
    
    publish_url = f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media_publish" # Updated API version
    publish_payload = {'creation_id': container_id, 'access_token': ACCESS_TOKEN}
    print("🚀 Publishing Reel to Instagram...")
    publish_response = requests.post(publish_url, data=publish_payload)
    
    if publish_response.ok:
        print("✅ Reel published successfully!"); print("Response:", publish_response.json())
        if cloudinary_public_id_to_delete and CLOUDINARY_CLOUD_NAME: # Check if deletion is needed and possible
            try: 
                print(f"🗑️ Deleting '{cloudinary_public_id_to_delete}' from Cloudinary...")
                cloudinary.api.delete_resources([cloudinary_public_id_to_delete], resource_type="video")
                print("🗑️ Video deleted from Cloudinary.")
            except Exception as e: print(f"⚠️ Failed to delete Cloudinary video: {e}")
        return True
    else: print(f"❌ Failed to publish: {publish_response.text}"); return False


# --- Main Video Generation and Upload Pipeline ---
def run_full_pipeline():
    # --- PART 1: VIDEO CREATION ---
    print("--- Step 1: Selecting and Filtering Pre-existing Background Image ---")
    if not os.path.isdir(PREEXISTING_BACKGROUND_IMAGE_FOLDER):
        print(f"Error: BG folder not found: {PREEXISTING_BACKGROUND_IMAGE_FOLDER}"); return None, None
    img_files = [f for f in os.listdir(PREEXISTING_BACKGROUND_IMAGE_FOLDER) if os.path.isfile(os.path.join(PREEXISTING_BACKGROUND_IMAGE_FOLDER, f)) and os.path.splitext(f)[1].lower() in ALLOWED_IMAGE_EXTENSIONS]
    if not img_files: print(f"Error: No suitable images in {PREEXISTING_BACKGROUND_IMAGE_FOLDER}"); return None, None
    sel_img_name = random.choice(img_files); sel_img_path = os.path.join(PREEXISTING_BACKGROUND_IMAGE_FOLDER, sel_img_name)
    print(f"Selected BG: {sel_img_path}")
    try:
        img_raw = Image.open(sel_img_path)
        img_filtered = _apply_black_filter_to_pil_image(img_raw, AI_BACKGROUND_IMAGE_FILTER_OPACITY)
        orig_w, orig_h = img_filtered.size; new_w = int(orig_w * 1.01)
        print(f"Scaling BG width from {orig_w} to {new_w}...")
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        img_scaled = img_filtered.resize((new_w, orig_h), resample)
        os.makedirs(os.path.dirname(FILTERED_BACKGROUND_FOR_VIDEO_FILENAME) or '.', exist_ok=True)
        img_scaled.save(FILTERED_BACKGROUND_FOR_VIDEO_FILENAME)
        print(f"Filtered BG saved to '{FILTERED_BACKGROUND_FOR_VIDEO_FILENAME}'")
    except Exception as e: print(f"Error processing BG: {e}"); import traceback; traceback.print_exc(); return None, None
    print("BG processing successful.\n")

    print("--- Step 2: Fetching Shloka and Meanings ---")
    used_shlokas = load_used_shlokas(USED_SHLOKAS_FILE)
    skt_raw, hin_raw, eng_raw = "", "", ""
    text_ok = False
    prompt =(
        "Provide a concise Sanskrit shloka (1-2 lines) from Hindu scriptures like the Upanishads, Bhagavad Gita, Mahabharata, Puranas, or other Smriti or Shruti texts."
        " Then, after 'हिंदी अर्थ:', provide an ACCURATE Hindi meaning (1-2 lines) for the shloka."
        " Then, after 'English Meaning:', provide its English meaning (1-2 lines)."
        " Keep all three parts relatively short and suitable for a vertical video format. Ensure the Hindi translation is a correct interpretation of the Sanskrit."
    )
    for _ in range(MAX_SHLOKA_RETRIES):
        resp = get_gemini_text_response(prompt)
        if "Error:" in resp or not resp.strip(): print(f"Gemini error: {resp}"); continue
        s, h, e = split_shloka_hindi_english(resp)
        if not s.strip(): print("Parsed Sanskrit empty. Retrying."); continue
        key_check = normalize_shloka_for_comparison(s.replace(",", " "))
        if key_check not in used_shlokas:
            skt_raw, hin_raw, eng_raw = s, h, e
            add_shloka_to_used(key_check, used_shlokas, USED_SHLOKAS_FILE)
            text_ok = True; print("Unique shloka obtained."); break
        else: print(f"Shloka key '{key_check[:50]}...' used. New one.")
    if not text_ok: print("Failed to get unique shloka."); return None, None

    skt_final, hin_final, eng_final = skt_raw, hin_raw, eng_raw # Keep original eng_raw for caption
    if skt_final: skt_final = skt_final.replace(",", "\n")
    if hin_final: hin_final = hin_final.replace(",", "\n")
    if skt_final.strip(): skt_final = f'"{skt_final.strip()}"'
    if hin_final.strip(): hin_final = f'"{hin_final.strip()}"'
    print(f"--- Processed Sanskrit ---\n{skt_final}\n--- Processed Hindi ---\n{hin_final}\n--- Processed English ---\n{eng_final}\n")

    sfs = calculate_dynamic_font_size(len(skt_final), BASE_FONT_SIZE_SANSKRIT, MIN_FONT_SIZE_SANSKRIT, IDEAL_LENGTH_FOR_BASE_SANSKRIT, MAX_LENGTH_FOR_MIN_SANSKRIT)
    hfs = calculate_dynamic_font_size(len(hin_final), BASE_FONT_SIZE_HINDI, MIN_FONT_SIZE_HINDI, IDEAL_LENGTH_FOR_BASE_HINDI, MAX_LENGTH_FOR_MIN_HINDI)
    efs = calculate_dynamic_font_size(len(eng_final), BASE_FONT_SIZE_ENGLISH, MIN_FONT_SIZE_ENGLISH, IDEAL_LENGTH_FOR_BASE_ENGLISH, MAX_LENGTH_FOR_MIN_ENGLISH)
    print(f"Font sizes - Skt: {sfs}pt, Hin: {hfs}pt, Eng: {efs}pt\n")

    print(f"--- Step 3: Rendering Text to '{RENDERED_TEXT_PNG}' ---")
    if not render_text_to_png_manual_v3(skt_final, hin_final, eng_final, RENDERED_TEXT_PNG, sfs, hfs, efs):
        print(f"Failed to render text."); return None, None
    print(f"Text rendered successfully.\n")
        
    print(f"--- Step 4: Selecting Background Music ---")
    music_file = None
    if BACKGROUND_MUSIC_FILES_LIST:
        avail_music = [f for f in BACKGROUND_MUSIC_FILES_LIST if os.path.exists(f)]
        if avail_music: music_file = random.choice(avail_music); print(f"Selected BGM: {music_file}")
        else: print("Warning: No BGM files found.")
    else: print("BGM list empty.")

    print(f"--- Step 5: Creating Final Video '{FINAL_OUTPUT_VIDEO_FILENAME}' ---")
    video_output_path = os.path.abspath(FINAL_OUTPUT_VIDEO_FILENAME) # Get absolute path
    if not create_video_ffmpeg(FILTERED_BACKGROUND_FOR_VIDEO_FILENAME, RENDERED_TEXT_PNG, music_file, video_output_path):
        print("\nFailed to create final video."); return None, None # Return None for video path if failed
    print(f"\nSuccessfully created video: '{video_output_path}'.")
    return video_output_path, eng_raw # Return path and original English meaning for caption

# --- Main Execution ---
if __name__ == "__main__":
    overall_success = False
    generated_video_path = None
    english_caption_text = None

    for attempt_num in range(1, MAX_OVERALL_ATTEMPTS + 1):
        print(f"\n--- Overall Video Generation Attempt {attempt_num}/{MAX_OVERALL_ATTEMPTS} ---")
        # run_full_pipeline now returns video_path and english_meaning
        video_path_result, english_meaning_result = run_full_pipeline()
        
        if video_path_result and english_meaning_result:
            overall_success = True
            generated_video_path = video_path_result
            english_caption_text = english_meaning_result
            print(f"\n--- Video generation successful on attempt {attempt_num}! Path: {generated_video_path} ---")
            break
        else:
            print(f"--- Video generation failed on attempt {attempt_num}. ---")
            if attempt_num < MAX_OVERALL_ATTEMPTS:
                print(f"Waiting for {RETRY_WAIT_SECONDS} seconds before retrying video creation...")
                time.sleep(RETRY_WAIT_SECONDS)
            else:
                print("\n--- All video generation attempts failed. ---")

    if overall_success and generated_video_path and english_caption_text:
        print(f"\n--- Proceeding to Upload Video: {generated_video_path} ---")
        
        # Construct caption
        final_caption = english_caption_text.strip() + "\n\n" + " ".join(HASHTAGS_LIST)
        print(f"--- Caption for Instagram ---\n{final_caption}\n--------------------------")

        print("📤 Uploading video to Cloudinary...")
        # Use generated_video_path for uploading
        direct_video_url, public_id = upload_to_cloudinary(generated_video_path)

        if direct_video_url and public_id:
            print(f"🔗 Cloudinary URL: {direct_video_url}")
            print("📸 Uploading reel to Instagram...")
            # Pass the constructed caption to the uploader function
            upload_success = upload_reel_to_instagram(direct_video_url, public_id, final_caption)
            if upload_success:
                print("🎉🎉🎉 Full process completed successfully! Video created and uploaded. 🎉🎉🎉")
            else:
                print(" L Video upload to Instagram failed.")
        else:
            print("⚠️ Cloudinary upload failed, cannot upload to Instagram.")
    elif overall_success and not generated_video_path: # Should not happen if pipeline logic is correct
         print("Logic error: Video generation reported success but no path returned.")
    else:
        print("Video creation was not successful. Skipping upload.")

    # Optional: Clean up temporary files
    # Consider deleting RENDERED_TEXT_PNG and FILTERED_BACKGROUND_FOR_VIDEO_FILENAME
    # Be careful not to delete FINAL_OUTPUT_VIDEO_FILENAME if you want to keep it locally after upload
    # if os.path.exists(RENDERED_TEXT_PNG):
    #     try: os.remove(RENDERED_TEXT_PNG)
    #     except OSError as e: print(f"Error removing {RENDERED_TEXT_PNG}: {e}")
    # if os.path.exists(FILTERED_BACKGROUND_FOR_VIDEO_FILENAME):
    #     try: os.remove(FILTERED_BACKGROUND_FOR_VIDEO_FILENAME)
    #     except OSError as e: print(f"Error removing {FILTERED_BACKGROUND_FOR_VIDEO_FILENAME}: {e}")