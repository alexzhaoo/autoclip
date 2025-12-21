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
    gpu_name = os.environ.get("VAST_GPU_NAME", "RTX_5090")
    image = os.environ.get("VAST_IMAGE", "pytorch/pytorch")
    disk_gb = int(os.environ.get("VAST_DISK_GB", "200"))
    print(f"💰 Launching instance with {gpu_name}")
    instance = vast.launch_instance(
        num_gpus="1",
        gpu_name=gpu_name,
        image=image,
        disk=disk_gb,
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

pip install --upgrade pip

# Install latest yt-dlp
pip install --upgrade yt-dlp

git clone https://github.com/alexzhaoo/autoclip pipeline
cd /workspace/pipeline

export OPENAI_API_KEY={OPENAI_KEY}
export HF_TOKEN={HF_TOKEN}

# Setup LightX2V + download Wan2.2 T2V A14B + Dual-LoRA into ./models/
chmod +x ./setup_wan.sh
./setup_wan.sh

source /workspace/pipeline/.venv/bin/activate

# Download NLTK data with multiple methods and fallbacks
echo "📚 Downloading NLTK data..."
python -c "
import os
import nltk
import ssl

# Set NLTK data path
nltk_data_dir = '/workspace/nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# SSL context fix for downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK datasets
datasets = ['punkt', 'punkt_tab', 'stopwords']
for dataset in datasets:
    try:
        nltk.download(dataset, download_dir=nltk_data_dir, quiet=False)
        print(f'✅ Downloaded {dataset}')
    except Exception as e:
        print(f'⚠️ Failed to download {dataset}: {e}')

print(f'NLTK data path: {nltk.data.path}')
"

# Set NLTK_DATA environment variable
export NLTK_DATA=/workspace/nltk_data

# Test YouTube download first
echo "Testing YouTube download..."
yt-dlp --print filename "https://www.youtube.com/watch?v=hCW2NHbWNwA&t=327s" || echo "YouTube download may fail"

export WAN22_BACKEND=lightx2v
export WAN22_MODELS_DIR=/workspace/pipeline/models
mkdir -p {remote_outdir}
python /workspace/pipeline/clip.py --video "{video_arg}" --output_dir {remote_outdir}
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
