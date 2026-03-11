#!/usr/bin/env python3
"""Standalone test for LTX-2 Fast video generation.

This script tests the LTX-2 video generator without needing to run
the full AutoClipper pipeline. Useful for verifying:
- Model loads correctly
- GPU is accessible
- Generation produces valid output
"""

import os
import sys
import torch

def test_basic_cuda():
    """Test basic CUDA availability."""
    print("=" * 60)
    print("STEP 1: CUDA Basics")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test basic tensor operation
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x.t())
    print(f"   Test tensor operation: ✅ (shape: {y.shape})")
    
    del x, y
    torch.cuda.empty_cache()
    return True


def test_ltx2_import():
    """Test importing LTX-2 from diffusers."""
    print("\n" + "=" * 60)
    print("STEP 2: LTX-2 Import")
    print("=" * 60)
    
    try:
        from diffusers import LTX2Pipeline
        print("✅ LTX2Pipeline imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import LTX2Pipeline: {e}")
        print("\nFix: pip install -U diffusers>=0.32.0")
        return False


def test_ltx2_generation(resolution="480p", aspect_ratio="16:9", fast_mode=True):
    """Test actual LTX-2 video generation."""
    print("\n" + "=" * 60)
    print(f"STEP 3: LTX-2 Generation ({resolution} {aspect_ratio}, fast_mode={fast_mode})")
    print("=" * 60)
    
    output_path = f"test_ltx2_{resolution}_{aspect_ratio.replace(':', '')}.mp4"
    
    # Clean up any previous test
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"   Cleaned up previous test file")
    
    try:
        from video_gen import create_ltx2_generator
        
        print("   Creating generator...")
        generator = create_ltx2_generator(
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            fast_mode=fast_mode
        )
        
        print("   Generating test video (this may take 1-5 minutes)...")
        print("   Prompt: 'A serene lake at sunset, gentle ripples on water'")
        
        import time
        start = time.time()
        
        result = generator.generate_clip(
            prompt="A serene lake at sunset, gentle ripples on water, mountains in background",
            output_path=output_path,
            duration=2.0,  # 2 seconds
            seed=42  # Reproducible
        )
        
        elapsed = time.time() - start
        
        # Verify output
        if not os.path.exists(output_path):
            print(f"❌ Output file not created: {output_path}")
            return False
        
        file_size = os.path.getsize(output_path)
        if file_size < 10000:  # Less than 10KB is probably corrupt
            print(f"❌ Output file too small ({file_size} bytes) - likely corrupted")
            return False
        
        print(f"✅ Generation successful!")
        print(f"   File: {output_path}")
        print(f"   Size: {file_size / 1024:.1f} KB")
        print(f"   Time: {elapsed:.1f}s ({elapsed/generator.config.num_inference_steps:.1f}s per step)")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """Report VRAM usage."""
    print("\n" + "=" * 60)
    print("STEP 4: Memory Summary")
    print("=" * 60)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved:  {reserved:.2f} GB")
        print(f"   Total:     {total:.2f} GB")
        print(f"   Free:      {total - allocated:.2f} GB")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LTX-2 Fast Video Generation Test")
    print("=" * 60)
    
    # Parse arguments
    resolution = "480p"
    aspect_ratio = "16:9"
    if len(sys.argv) > 1:
        resolution = sys.argv[1]  # 480p, 720p, 1080p
    if len(sys.argv) > 2:
        aspect_ratio = sys.argv[2]  # 16:9 or 9:16
    
    print(f"\nTest configuration:")
    print(f"   Resolution: {resolution}")
    print(f"   Aspect Ratio: {aspect_ratio}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Run tests
    results = []
    
    results.append(("CUDA Basics", test_basic_cuda()))
    results.append(("LTX-2 Import", test_ltx2_import()))
    results.append(("LTX-2 Generation", test_ltx2_generation(resolution, aspect_ratio)))
    test_memory_usage()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✅ All tests passed! LTX-2 is working correctly.")
        print(f"\nTest video saved to: test_ltx2_{resolution}_{aspect_ratio.replace(':', '')}.mp4")
    else:
        print("\n❌ Some tests failed. Check errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
