import os
import time
import subprocess
from vastai_sdk import VastAI

API_KEY = os.environ["VAST_API_KEY"]
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# Initialize the SDK client
vast = VastAI(api_key=API_KEY)

def test_api_connection():
    try:
        user_info = vast.show_user()
        print(f"✅ API connection successful. User: {user_info.get('username', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def find_offer():
    # Using SDK launch_instance will search and create, but to inspect offers first:
    offers = vast.show_offers(q="verified=true,rentable=true,gpu_name=RTX_4090,num_gpus=1,gpu_ram>=24", order="dph_total", limit=10)
    if not offers:
        raise RuntimeError("No offers found… consider relaxing GPU requirements")
    return offers[0]

def create_instance(offer):
    # The SDK launch_instance can handle searching + creation, but we can also give explicit id
    client_id = offer.get("id")
    print(f"💰 Launching instance with offer ID: {client_id}")
    instance = vast.launch_instance(num_gpus=str(offer.get("num_gpus", 1)),
                                    gpu_name=offer.get("gpu_name", ""),
                                    image="pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime",
                                    disk=60)
    return instance

def wait_for_running(instance_id):
    while True:
        inst = vast.show_instance(id=instance_id)
        status = inst["status_msg"]
        print(f"⏳ Status: {status}")
        if status == "running":
            ssh_cmd = inst["ssh_cmd"]
            print(f"✅ Instance running! SSH with:\n{ssh_cmd}")
            return inst
        time.sleep(10)

def setup_and_run(ssh_cmd, video_input):
    remote_outdir = "/workspace/output"
    if video_input.startswith(("http://", "https://")):
        video_arg = video_input
        print(f"📺 Using URL: {video_input}")
    else:
        remote_video = "/workspace/input_video.mp4"
        target = ssh_cmd.split()[-1]
        subprocess.run(f"scp -o StrictHostKeyChecking=no {video_input} {target}:{remote_video}", shell=True, check=True)
        video_arg = remote_video
        print(f"📁 Uploaded local file: {video_input}")

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
export OPENAI_API_KEY={OPENAI_KEY}
mkdir -p {remote_outdir}
python pipeline/clip.py --video "{video_arg}" --output_dir {remote_outdir}
"""
    subprocess.run(f'{ssh_cmd} "{setup_script}"', shell=True, check=True)
    subprocess.run(f"scp -o StrictHostKeyChecking=no -r {ssh_cmd.split()[-1]}:{remote_outdir} ./", shell=True, check=True)

def shutdown_instance(instance_id):
    vast.stop_instance(id=instance_id)
    print("💤 Instance stopped.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run_on_vast.py <video_path_or_url>")
        sys.exit(1)

    video_input = sys.argv[1]

    try:
        print("🔗 Testing API connection...")
        if not test_api_connection():
            sys.exit(1)

        print("🔍 Searching for suitable offer…")
        offer = find_offer()
        print(f"Offer: GPU {offer.get('gpu_name')} – offer ID {offer.get('id')}")

        print("🚀 Launching instance…")
        instance = create_instance(offer)
        instance_id = instance["id"]

        print("⏳ Waiting for instance to run…")
        inst_details = wait_for_running(instance_id)
        ssh_cmd = inst_details["ssh_cmd"]

        print("⚙️ Running setup and pipeline…")
        setup_and_run(ssh_cmd, video_input)

        print("🛑 Stopping instance…")
        shutdown_instance(instance_id)
        print("✅ Done.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Tips:\n- Ensure VAST_API_KEY is set and valid\n- Consider relaxing GPU requirements or checking the Vast.ai web UI manually")
        sys.exit(1)
