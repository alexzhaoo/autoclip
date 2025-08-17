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
    """Test basic API connectivity"""
    try:
        user_info = vast.show_user()
        print(f"✅ API connection successful. User: {user_info.get('username', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def find_offer():
    """List available machines and pick one that matches requirements"""
    # Using list_machines to search instead of a non-existent show_offers
    machines = vast.show_machines(quiet=True)
    filtered = [m for m in machines if m.get("gpu_name") == "RTX_4090" and m.get("num_gpus", 0) >= 1 and m.get("gpu_ram", 0) >= 24]
    if not filtered:
        raise RuntimeError("No offers found… consider relaxing GPU requirements")
    return filtered[0]

def create_instance(offer):
    """Launch instance using the offer ID"""
    print(f"💰 Launching instance with offer ID: {offer['id']}")
    instance = vast.launch_instance(
        num_gpus=str(offer.get("num_gpus", 1)),
        gpu_name=offer.get("gpu_name", ""),
        image="pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime",
        disk=60
    )
    return instance

def wait_for_running(instance_id):
    """Poll instance status until running"""
    while True:
        inst = vast.show_instance(id=instance_id)
        status = inst.get("status_msg", "")
        print(f"⏳ Status: {status}")
        if status.lower() == "running":
            ssh_cmd = inst.get("ssh_cmd", "")
            print(f"✅ Instance running! SSH with:\n{ssh_cmd}")
            return inst
        time.sleep(10)

def setup_and_run(ssh_cmd, video_input):
    """Setup environment and run Wan2.2 pipeline"""
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
sudo apt update && sudo apt install -y ffmpeg git python3-pip
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
    vast.destroy_instance(ID=instance_id)
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
        instance_id = instance.get("id")

        print("⏳ Waiting for instance to run…")
        inst_details = wait_for_running(instance_id)
        ssh_cmd = inst_details.get("ssh_cmd", "")

        print("⚙️ Running setup and pipeline…")
        setup_and_run(ssh_cmd, video_input)

        print("🛑 Stopping instance…")
        shutdown_instance(instance_id)
        print("✅ Done.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Tips:\n- Ensure VAST_API_KEY is set and valid\n- Consider relaxing GPU requirements or checking the Vast.ai web UI manually")
        sys.exit(1)
