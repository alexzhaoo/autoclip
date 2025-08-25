import os
import time
import subprocess
from vastai_sdk import VastAI

API_KEY = os.environ["VAST_API_KEY"]
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
SSH_KEY_PATH = os.environ.get("VASTAI_SSH_KEY")

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

def create_instance():
    """Launch a new instance directly with fixed GPU/image"""
    print("💰 Launching instance with RTX 5090")
    instance = vast.launch_instance(
        num_gpus="1",
        gpu_name="RTX_5090",
        image="pytorch/pytorch",
        disk=100
    )
    print(instance)
    return instance

def wait_for_running(instance_id):
    """Poll instance status until running"""
    while True:
        inst_details = vast.show_instance(id=instance_id)
        ssh_port = inst_details["ssh_port"]
        ssh_host = inst_details["ssh_host"]
        print(f"SSH: {ssh_host}:{ssh_port}")
        status = inst_details.get("cur_state", "")
        print(f"⏳ Status: {status}")
        if status.lower() == "running":
            return inst_details
        time.sleep(5)


def setup_and_run(inst_details, video_input):
    """Setup environment and run Wan2.2 pipeline"""
    remote_outdir = "/workspace/output"
    ssh_port = inst_details["ssh_port"]
    ssh_host = inst_details["ssh_host"]

    # Handle video input
    if video_input.startswith(("http://", "https://")):
        video_arg = video_input
        print(f"📺 Using URL: {video_input}")
    else:
        remote_video = "/workspace/input_video.mp4"
        subprocess.run(
            f"scp -P {ssh_port} -o StrictHostKeyChecking=no "
            f"{video_input} root@{ssh_host}:{remote_video}",
            shell=True,
            check=True,
        )
        video_arg = remote_video
        print(f"📁 Uploaded local file: {video_input}")

    # Remote setup script
    setup_script = f"""
set -e
cd /workspace

# Fix DNS resolution issues
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
echo "nameserver 8.8.4.4" | sudo tee -a /etc/resolv.conf

sudo apt update && sudo apt install -y ffmpeg git python3-pip git-lfs curl

# Install latest yt-dlp
pip install --upgrade yt-dlp

pip install huggingface_hub
git clone https://github.com/Wan-Video/Wan2.2.git
pip install -r Wan2.2/requirements.txt

# Download model to the workspace directory (not inside Wan2.2)
HUGGINGFACE_HUB_TOKEN={HF_TOKEN} huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /workspace/Wan2.2-TI2V-5B
cd /workspace

git clone https://github.com/alexzhaoo/autoclip pipeline
pip install -r pipeline/requirements.txt

# Test YouTube download first
echo "Testing YouTube download..."
yt-dlp --print filename "https://www.youtube.com/watch?v=hCW2NHbWNwA&t=327s" || echo "YouTube download may fail"

export WAN22_PATH=/workspace/Wan2.2
export WAN22_MODEL=ti2v-5B
export OPENAI_API_KEY={OPENAI_KEY}
mkdir -p {remote_outdir}
python pipeline/clip.py --video "{video_arg}" --output_dir {remote_outdir}
"""

    print("📝 Creating setup script on remote machine...")
    # Write setup script to remote machine and execute it
    script_write_cmd = f'ssh -p {ssh_port} -o StrictHostKeyChecking=no root@{ssh_host} "cat > /tmp/setup.sh << \'EOF\'\n{setup_script}\nEOF"'
    subprocess.run(script_write_cmd, shell=True, check=True)
    
    print("🔧 Making script executable and running setup...")
    script_exec_cmd = f'ssh -p {ssh_port} -o StrictHostKeyChecking=no root@{ssh_host} "chmod +x /tmp/setup.sh && bash /tmp/setup.sh"'
    subprocess.run(script_exec_cmd, shell=True, check=True)
    
    print("📥 Downloading results...")
    subprocess.run(
        f"scp -P {ssh_port} -o StrictHostKeyChecking=no "
        f"-r root@{ssh_host}:{remote_outdir} ./",
        shell=True,
        check=True,
    )

def shutdown_instance(instance_id):
    vast.destroy_instance(ID=instance_id)
    print("💤 Instance destroyed.")

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

        print("🚀 Launching instance…")
        instance = create_instance()
        instance_id = instance.get("new_contract")
        print(instance_id)

        print("⏳ Waiting for instance to run…")
        inst_details = wait_for_running(instance_id)

        print("⚙️ Running setup and pipeline…")
        time.sleep(45)  # Wait for instance to stabilize
        setup_and_run(inst_details, video_input)

        print("🛑 Destroying instance…")
        shutdown_instance(instance_id)
        print("✅ Done.")

    except Exception as e:
        print(f"❌ Error: {e}")
    
        sys.exit(1)
