#!/usr/bin/env python3
"""
Test script to generate a single B-roll video using production_broll.py
This creates a short test video to verify the B-roll generation works.
"""

import os
import json
from production_broll import create_production_pipeline

def create_test_clip_data():
    """Create test clip data with sample transcript segments"""
    return {
        "hook": "Test B-roll generation",
        "caption": "Testing the B-roll system",
        "transcript_text": "This is a test to see if the B-roll generation system works properly with Wan2.2",
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "This is a test to see if the B-roll generation system works properly with Wan2.2"
            },
            {
                "start": 3.0,
                "end": 6.0,
                "text": "We want to verify that the video generation pipeline is functioning correctly"
            },
            {
                "start": 6.0,
                "end": 9.0,
                "text": "This will help us confirm everything is set up before using OpenAI tokens"
            }
        ],
        "duration": 9.0
    }

def main():
    print("🎬 Testing B-roll Video Generation")
    print("=" * 50)

    # Create test clip data
    test_clip = create_test_clip_data()
    print(f"📝 Test clip: {test_clip['hook']}")
    print(f"📏 Duration: {test_clip['duration']} seconds")
    print(f"📊 Segments: {len(test_clip['segments'])}")

    try:
        # Create the production pipeline
        print("\n🔧 Initializing production pipeline...")
        pipeline = create_production_pipeline()
        print("✅ Pipeline initialized successfully")

        # Process the test clip
        print("\n🎬 Generating B-roll video...")
        success = pipeline.process_clip(test_clip)

        if success:
            print("\n🎉 SUCCESS! B-roll video generation works!")
            print("📁 Check the current directory for generated broll_*.mp4 files")

            # List generated files
            import glob
            broll_files = glob.glob("broll_*.mp4")
            if broll_files:
                print(f"📹 Generated {len(broll_files)} B-roll video(s):")
                for file in broll_files:
                    size = os.path.getsize(file) / 1024
                    print(f"  - {file}: {size:.1f}KB")
            else:
                print("⚠️ No broll_*.mp4 files found in current directory")
        else:
            print("\n❌ B-roll generation failed")
            return False

    except Exception as e:
        print(f"\n❌ Error during B-roll generation: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)