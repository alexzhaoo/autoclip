#!/usr/bin/env python3
"""Test script for LTX-2 video generation.

This script tests the LTX-2 Fast generator with various configurations
to ensure proper functionality.
"""

import os
import sys
import argparse

# Ensure we're using the local video_gen module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_basic_generation():
    """Test basic LTX-2 video generation."""
    print("\n" + "="*60)
    print("🎬 Test 1: Basic LTX-2 Generation")
    print("="*60)
    
    from video_gen import LTX2FastGenerator, create_ltx2_generator
    
    try:
        # Create generator with default settings
        print("\n📦 Creating LTX-2 generator (480p, 16:9)...")
        generator = create_ltx2_generator(
            resolution="480p",
            aspect_ratio="16:9",
            fast_mode=True,
        )
        print("✅ Generator created successfully!")
        
        # Test prompt
        test_prompt = (
            "Smooth aerial dolly shot gliding over a modern city at dusk, "
            "camera moving forward steadily, warm lights glowing from windows, "
            "soft atmospheric haze, symmetrical composition"
        )
        
        output_file = "test_ltx2_basic.mp4"
        
        print(f"\n📝 Prompt: {test_prompt}")
        print(f"📁 Output: {output_file}")
        print(f"⏱️  Duration: 2 seconds\n")
        
        result = generator.generate_clip(
            prompt=test_prompt,
            output_path=output_file,
            duration=2.0,
        )
        
        print(f"\n✅ Success! Video saved to: {result}")
        
        # Check file size
        file_size = os.path.getsize(result) / 1024
        print(f"📊 File size: {file_size:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vertical_video():
    """Test 9:16 vertical video generation for mobile."""
    print("\n" + "="*60)
    print("🎬 Test 2: Vertical Video (9:16) for TikTok/Reels")
    print("="*60)
    
    from video_gen import create_ltx2_generator
    
    try:
        print("\n📦 Creating LTX-2 generator (480p, 9:16)...")
        generator = create_ltx2_generator(
            resolution="480p",
            aspect_ratio="9:16",
            fast_mode=True,
        )
        print("✅ Generator created successfully!")
        
        test_prompt = (
            "Close-up tracking shot of a barista's hands crafting latte art, "
            "steam rising from the cup, warm cafe lighting, smooth camera movement"
        )
        
        output_file = "test_ltx2_vertical.mp4"
        
        print(f"\n📝 Prompt: {test_prompt}")
        print(f"📁 Output: {output_file}\n")
        
        result = generator.generate_clip(
            prompt=test_prompt,
            output_path=output_file,
            duration=2.0,
        )
        
        print(f"\n✅ Success! Video saved to: {result}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hd_generation():
    """Test 720p HD generation."""
    print("\n" + "="*60)
    print("🎬 Test 3: HD Generation (720p)")
    print("="*60)
    
    from video_gen import create_ltx2_generator
    
    try:
        print("\n📦 Creating LTX-2 generator (720p, 16:9)...")
        generator = create_ltx2_generator(
            resolution="720p",
            aspect_ratio="16:9",
            fast_mode=True,
        )
        print("✅ Generator created successfully!")
        
        test_prompt = (
            "Cinematic wide shot of ocean waves crashing on rocky shoreline, "
            "golden hour lighting, camera slowly panning across the scene, "
            "spray misting in the air, dramatic clouds"
        )
        
        output_file = "test_ltx2_hd.mp4"
        
        print(f"\n📝 Prompt: {test_prompt}")
        print(f"📁 Output: {output_file}")
        print(f"⚠️  Note: 720p requires more VRAM and time\n")
        
        result = generator.generate_clip(
            prompt=test_prompt,
            output_path=output_file,
            duration=2.0,
        )
        
        print(f"\n✅ Success! Video saved to: {result}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_broll_pipeline():
    """Test B-roll generation through the production pipeline."""
    print("\n" + "="*60)
    print("🎬 Test 4: B-roll Pipeline Integration")
    print("="*60)
    
    from production_broll import LTX2VideoGenerator
    
    try:
        print("\n📦 Creating LTX-2 B-roll generator...")
        generator = LTX2VideoGenerator(
            resolution="480p",
            aspect_ratio="16:9",
            fast_mode=True,
        )
        print("✅ Generator created successfully!")
        
        test_prompt = (
            "Abstract flowing particles in dark void, "
            "bioluminescent colors swirling and morphing, "
            "smooth continuous motion, symmetrical patterns"
        )
        
        output_file = "test_ltx2_broll.mp4"
        
        print(f"\n📝 Prompt: {test_prompt}")
        print(f"📁 Output: {output_file}")
        print(f"⏱️  Duration: 2.5 seconds\n")
        
        success = generator.generate_broll_video(
            prompt=test_prompt,
            duration=2.5,
            output_path=output_file,
        )
        
        if success:
            print(f"\n✅ Success! B-roll video generated: {output_file}")
            return True
        else:
            print("\n❌ Generation returned False")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_durations():
    """Test different video durations."""
    print("\n" + "="*60)
    print("🎬 Test 5: Different Durations")
    print("="*60)
    
    from video_gen import LTX2FastGenerator
    
    durations = [1.0, 2.0, 3.0]
    results = []
    
    for duration in durations:
        try:
            print(f"\n⏱️  Testing {duration}s duration...")
            
            generator = LTX2FastGenerator()
            
            test_prompt = f"Smooth rotating abstract geometric shapes, soft lighting, {duration}s duration test"
            output_file = f"test_ltx2_duration_{int(duration)}s.mp4"
            
            result = generator.generate_clip(
                prompt=test_prompt,
                output_path=output_file,
                duration=duration,
            )
            
            print(f"   ✅ {duration}s: Success!")
            results.append((duration, True))
            
        except Exception as e:
            print(f"   ❌ {duration}s: Failed - {e}")
            results.append((duration, False))
    
    print(f"\n📊 Duration Test Summary:")
    for duration, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {duration}s")
    
    return all(success for _, success in results)


def main():
    parser = argparse.ArgumentParser(description="Test LTX-2 Video Generation")
    parser.add_argument("--test", type=int, default=0, 
                        help="Run specific test (1-5). 0 = run all tests.")
    parser.add_argument("--skip-hd", action="store_true",
                        help="Skip HD test (saves time/VRAM)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 LTX-2 Fast Video Generation Test Suite")
    print("="*60)
    
    # Check environment
    print("\n🔍 Environment Check:")
    print(f"   Python: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("   ❌ PyTorch not installed!")
        sys.exit(1)
    
    try:
        import diffusers
        print(f"   Diffusers: {diffusers.__version__}")
    except ImportError:
        print("   ❌ Diffusers not installed!")
        sys.exit(1)
    
    # Run tests
    test_map = {
        1: ("Basic Generation", test_basic_generation),
        2: ("Vertical Video", test_vertical_video),
        3: ("HD Generation", test_hd_generation),
        4: ("B-roll Pipeline", test_broll_pipeline),
        5: ("Different Durations", test_different_durations),
    }
    
    results = []
    
    if args.test == 0:
        # Run all tests
        for test_num, (test_name, test_func) in test_map.items():
            if args.skip_hd and test_num == 3:
                print(f"\n⏭️  Skipping HD test (--skip-hd)")
                continue
            try:
                success = test_func()
                results.append((test_name, success))
            except KeyboardInterrupt:
                print("\n\n⚠️  Tests interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Test {test_num} crashed: {e}")
                results.append((test_name, False))
    else:
        # Run specific test
        if args.test in test_map:
            test_name, test_func = test_map[args.test]
            success = test_func()
            results.append((test_name, success))
        else:
            print(f"❌ Invalid test number: {args.test}")
            print(f"Valid options: {list(test_map.keys())}")
            sys.exit(1)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
