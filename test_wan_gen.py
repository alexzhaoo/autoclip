#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import traceback

def main():
    parser = argparse.ArgumentParser(
        description="Test LightX2V Wan2.2 Video Generation Setup"
    )
    parser.add_argument("--prompt", type=str, default="A cinematic drone shot of a futuristic city with neon lights, smooth motion", help="Prompt for generation")
    parser.add_argument("--output", type=str, default="test_wan_gen.mp4", help="Output video path")
    parser.add_argument("--models-dir", type=str, default=os.getenv("WAN22_MODELS_DIR", "./models"), help="Path to models directory")
    
    args = parser.parse_args()
    
    print(f"🔍 Testing Wan2.2 LightX2V Generator Setup")
    print(f"  • Models Dir: {args.models_dir}")
    print(f"  • Output Path: {args.output}")
    print(f"  • Prompt: {args.prompt}")
    print("-" * 50)
    
    # 1. Imports
    try:
        print("⏳ Importing video_gen...", end="", flush=True)
        from video_gen import Wan22LightX2VGenerator
        print(" ✅")
    except ImportError as e:
        print(f"\n❌ Failed to import video_gen: {e}")
        print(f"Ensure you are in the correct directory and 'video_gen.py' exists.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during import: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Initialization
    try:
        print("⏳ Initializing Generator (loading models)...", flush=True)
        # Force environment variables if not set, to match typical production pipeline
        if not os.getenv("WAN22_MODELS_DIR"):
            os.environ["WAN22_MODELS_DIR"] = args.models_dir
            
        generator = Wan22LightX2VGenerator(models_dir=args.models_dir)
        print("✅ Generator Initialized Successfully!")
    except Exception as e:
        print(f"\n❌ Initialization Failed: {e}")
        traceback.print_exc()
        print("\n💡 Troubleshooting Tips:")
        print("  1. Check if 'cfg_p_size' error? -> We patched video_gen.py to force parallel=False.")
        print("  2. Check if 'FlashAttn' error? -> We patched video_gen.py to force sdpa.")
        print("  3. Check if 'default process group' error? -> We patched video_gen.py to skip double init.")
        sys.exit(1)

    # 3. Generation
    try:
        print(f"\n⏳ Generating video Clip...", flush=True)
        output_path = Path(args.output).resolve()
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_path = generator.generate_clip(
            prompt=args.prompt,
            output_path=str(output_path)
        )
        
        if Path(result_path).exists():
             print(f"\n✅ Generation Successful! Video saved to: {result_path}")
             file_size = os.path.getsize(result_path) / (1024 * 1024)
             print(f"  • Size: {file_size:.2f} MB")
        else:
             print(f"\n❌ Generation apparently finished but file not found at: {result_path}")
             
    except Exception as e:
        print(f"\n❌ Generation Failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
