import os
import requests
import time
import subprocess

API_KEY = os.environ["VAST_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
BASE_URL = "https://console.vast.ai/api/v0"

def test_api_connection():
    """Test basic API connectivity"""
    try:
        result = vast_get("users/current")
        print(f"✅ API connection successful. User: {result.get('username', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def vast_get(path, **params):
    r = requests.get(f"{BASE_URL}/{path}", params=params, headers={"Authorization": f"Bearer {API_KEY}"})
    r.raise_for_status()
    return r.json()

def vast_post(path, payload):
    r = requests.post(f"{BASE_URL}/{path}", json=payload, headers={"Authorization": f"Bearer {API_KEY}"})
    r.raise_for_status()
    return r.json()

def vast_cmd(path, payload):
    r = requests.put(f"{BASE_URL}/{path}", json=payload, headers={"Authorization": f"Bearer {API_KEY}"})
    r.raise_for_status()
    return r.json()

def find_offer():
    # Use q-string format for searching offers
    q_filter = "verified=true,rentable=true,gpu_name=RTX_4090,num_gpus=1,gpu_ram>=24"
    search_params = {
        "q": q_filter,
        "order": "dph_total",
        "limit": "10"
    }
    
    try:
        # Try the offers endpoint first
        offers = vast_get("offers", **search_params)
        if isinstance(offers, list) and len(offers) > 0:
            return offers[0]
        elif isinstance(offers, dict) and "offers" in offers and len(offers["offers"]) > 0:
            return offers["offers"][0]
    except Exception as e:
        print(f"⚠️ Offers endpoint failed: {e}")
    
    try:
        # Fallback to bundles endpoint
        result = vast_get("bundles", **search_params)
        offers = result.get("offers", [])
        if offers and len(offers) > 0:
            return offers[0]
    except Exception as e:
        print(f"⚠️ Bundles endpoint failed: {e}")
    
    raise RuntimeError("No offers found for your specs. Try checking Vast.ai manually or relaxing GPU requirements.")

def create_instance(offer):
    # Handle different field names from the API
    client_id = offer.get("client_id") or offer.get("id")
    min_bid = offer.get("min_bid") or offer.get("dph_total") or offer.get("dph_base", 0.5)
    
    payload = {
        "client_id": client_id,
        "image": "pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime",
        "disk": 60,
        "min_bid": min_bid
    }
    
    print(f"💰 Creating instance with client_id: {client_id}, min_bid: ${min_bid}/hr")
    return vast_post("instances", payload)

def wait_for_running(instance_id):
    while True:
        details = vast_get(f"instances/{instance_id}")
        status = details["instance"]["status_msg"]
        if status == "running":
            ssh = details["instance"]["ssh_cmd"]
            print(f"✅ Instance running! SSH with:\n{ssh}")
            return details["instance"]
        print("⏳ Waiting for instance to start...")
        time.sleep(10)

def setup_and_run(ssh_cmd, video_input):
    remote_outdir = "/workspace/output"

    # Check if input is a URL or local file
    if video_input.startswith("http://") or video_input.startswith("https://"):
        # It's a YouTube URL, pass it directly to the script
        video_arg = video_input
        print(f"📺 Using YouTube URL: {video_input}")
    else:
        # It's a local file, upload it first
        remote_video = "/workspace/input_video.mp4"
        subprocess.run(f"scp -o StrictHostKeyChecking=no {video_input} {ssh_cmd.split()[-1]}:{remote_video}", shell=True)
        video_arg = remote_video
        print(f"📁 Uploaded local file: {video_input}")

    # Run remote setup and pipeline
    setup_script = f"""
    set -e
    sudo apt update && sudo apt install -y ffmpeg git
    pip install huggingface_hub
    git clone https://github.com/Wan-Video/Wan2.2.git
    pip install -r Wan2.2/requirements.txt
    HUGGINGFACE_HUB_TOKEN={HF_TOKEN} huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B

    git clone https://github.com/alexzhaoo/autoclip pipeline
    pip install -r pipeline/requirements.txt
    export WAN22_PATH=/workspace/Wan2.2
    export WAN22_MODEL=ti2v-5B
    export OPENAI_API_KEY={os.environ['OPENAI_API_KEY']}

    mkdir -p {remote_outdir}
    python pipeline/clip.py --video "{video_arg}" --output_dir {remote_outdir}
    """
    subprocess.run(f'{ssh_cmd} "{setup_script}"', shell=True)

    # Download results
    subprocess.run(f"scp -o StrictHostKeyChecking=no -r {ssh_cmd.split()[-1]}:{remote_outdir} ./", shell=True)

def shutdown_instance(instance_id):
    vast_cmd(f"instances/{instance_id}", {"state": "stopped"})
    print("💤 Instance stopped.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_on_vast.py <video_path_or_youtube_url>")
        print("Examples:")
        print("  python run_on_vast.py /path/to/video.mp4")
        print("  python run_on_vast.py 'https://www.youtube.com/watch?v=VIDEO_ID'")
        sys.exit(1)

    video_input = sys.argv[1]

    try:
        print("� Testing API connection...")
        if not test_api_connection():
            print("❌ Cannot connect to Vast.ai API")
            sys.exit(1)
        
        print("�🔍 Finding offer...")
        offer = find_offer()
        
        # Handle different field names for display
        gpu_name = offer.get("gpu_name", "Unknown GPU")
        price = offer.get("min_bid") or offer.get("dph_total") or offer.get("dph_base", "Unknown")
        print(f"Found: {gpu_name} at ${price}/hr")

        print("🚀 Creating instance...")
        instance = create_instance(offer)
        instance_id = instance["instance"]["id"]

        print("⏳ Waiting for instance to be running...")
        inst_details = wait_for_running(instance_id)
        ssh_cmd = inst_details["ssh_cmd"]

        print("⚙️ Setting up environment and running pipeline...")
        setup_and_run(ssh_cmd, video_input)

        print("🛑 Shutting down instance...")
        shutdown_instance(instance_id)
        print("✅ Done!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Tips:")
        print("- Check that VAST_API_KEY environment variable is set")
        print("- Verify your Vast.ai API key is valid")
        print("- Try checking available instances on https://vast.ai manually")
        sys.exit(1)
