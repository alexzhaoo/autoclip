#!/usr/bin/env python3
"""Production B-roll pipeline with LTX-2 Fast video generation.

This module provides a complete pipeline for analyzing video content,
identifying B-roll opportunities with GPT, and generating high-quality
B-roll videos using Lightricks' LTX-2 Fast model.
"""

import os
import json
import traceback
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class BRollRegion:
    """Represents a region suitable for B-roll replacement"""
    start_time: float
    end_time: float
    duration: float
    reason: str
    confidence: float
    prompt: Optional[str] = None
    broll_path: Optional[str] = None


class ProductionBRollAnalyzer:
    """Enhanced B-roll analyzer with GPT integration"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY")
        )
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    def analyze_content_for_broll(self, segments: List[Dict]) -> List[BRollRegion]:
        """Analyze content using proven GPT method to identify B-roll opportunities"""
        print("🧠 Using GPT to analyze transcript for B-roll opportunities...")
        
        # Process in chunks for better results
        chunk_size = 4
        all_regions = []
        
        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (len(segments) + chunk_size - 1) // chunk_size
            
            print(f"  Analyzing chunk {chunk_num}/{total_chunks}...")
            regions = self._analyze_chunk_with_gpt(chunk)
            all_regions.extend(regions)
            
            # Small delay to respect rate limits
            time.sleep(0.5)
        
        # Apply timing constraints: 1 B-roll every ~10 seconds, 1-2 seconds each
        filtered_regions = self._apply_timing_constraints(all_regions)
        
        print(f"  ✅ Found {len(all_regions)} candidates, filtered to {len(filtered_regions)} optimal placements")
        return filtered_regions
    
    def _apply_timing_constraints(self, regions: List[BRollRegion], min_spacing: float = 2.0) -> List[BRollRegion]:
        """Apply timing constraints: maintain spacing between B-rolls, limit duration
        
        Args:
            regions: List of B-roll regions to filter
            min_spacing: Minimum seconds between B-roll placements (default 2.0)
        """
        if not regions:
            return regions
        
        # Sort by start time
        sorted_regions = sorted(regions, key=lambda r: r.start_time)
        
        filtered = []
        last_broll_time = -min_spacing  # Allow B-roll at start
        
        for region in sorted_regions:
            # Check spacing constraint
            if region.start_time - last_broll_time >= min_spacing:
                # Constrain duration to 1-3 seconds
                max_duration = min(3.0, region.duration)
                constrained_region = BRollRegion(
                    start_time=region.start_time,
                    end_time=min(region.start_time + max_duration, region.end_time),
                    duration=max_duration,
                    prompt=region.prompt,
                    reason=region.reason,
                    confidence=region.confidence
                )
                
                filtered.append(constrained_region)
                last_broll_time = region.start_time
                print(f"    ✅ Kept B-roll at {region.start_time:.1f}s")
            else:
                spacing = region.start_time - last_broll_time
                print(f"    ⏭️  Filtered out B-roll at {region.start_time:.1f}s (spacing {spacing:.1f}s < {min_spacing:.1f}s)")
        
        return filtered
    
    def _analyze_chunk_with_gpt(self, chunk: List[Dict]) -> List[BRollRegion]:
        """Analyze a chunk of segments with GPT"""
        # Create transcript with timing information
        chunk_start = chunk[0].get("start", 0) if chunk else 0
        chunk_end = chunk[-1].get("end", 0) if chunk else 0
        
        # Build transcript with word-level timestamps for precise alignment
        transcript_lines = []
        for seg in chunk:
            if "words" in seg and seg["words"] and len(seg["words"]) > 0:
                first_word_time = seg["words"][0].get("start", 0)
                last_word_time = seg["words"][-1].get("end", 0)
                words_text = " ".join([w.get("word", "") for w in seg["words"]])
                transcript_lines.append(f"[{first_word_time:.2f}s - {last_word_time:.2f}s] {words_text}")
            else:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                seg_text = seg.get("text", "")
                transcript_lines.append(f"[{seg_start:.2f}s - {seg_end:.2f}s] {seg_text}")
        
        transcript_with_timing = "\n".join(transcript_lines)
        clean_transcript = " ".join([seg.get("text", "").strip() for seg in chunk])

        # Enhanced prompt optimized for LTX-2 generation
        prompt = f"""
            Analyze this video transcript and identify 3 short B-roll opportunities (2-4 seconds) that enhance viewer engagement.
            
            FULL CONTEXT (Read this first to understand the story flow):
            "{clean_transcript}"

            TIMED TRANSCRIPT (Use this ONLY for precise start/end times):
            {transcript_with_timing}
            
            CRITICAL TIMING RULES:
            1. START_TIME must be within the time ranges shown above
            2. END_TIME must be 2-4 seconds after START_TIME
            3. B-roll duration should be 2-4 seconds
            4. Choose the EXACT moment when the relevant word/phrase begins speaking
            5. Times must be between {chunk_start:.1f} and {chunk_end:.1f}
            6. Choose moments where visuals would enhance understanding
            
            For each B-roll opportunity, provide:

            START_TIME – MUST be an exact timestamp from the TIMED TRANSCRIPT above (in seconds)
            END_TIME – MUST be a timestamp from the TIMED TRANSCRIPT above (in seconds)
            VISUAL_PROMPT – vivid, cinematic (LTX-2 optimized)
            reason - why this clip fits and which transcript line it illustrates

            Create RELEVANT visuals for LTX-2 video generation using this EXACT structure:
            
            LTX-2 PROMPT BEST PRACTICES:
            
            1. FORMAT: Write as a SINGLE FLOWING PARAGRAPH (not bullet points)
            2. LENGTH: Keep under 200 words
            3. TENSE: Use present tense verbs only
            4. STRUCTURE (in order):
               - Start with MAIN ACTION in one clear sentence
               - Add specific movements and gestures (visual cues, not emotions)
               - Describe subject appearance precisely
               - Include background/environment details
               - Specify camera angle and movement
               - Describe lighting and atmosphere
            
            5. CRITICAL RULES:
               - SINGLE SUBJECT only (one person or main object)
               - NO detailed faces (use hands, silhouettes, or back of head)
               - NO text, logos, or signs (LTX-2 can't generate readable text)
               - NO complex physics (juggling, fast chaotic motion)
               - NO multiple interacting people
               - NO emotional labels like "sad" or "happy" - describe visual cues instead
            
            6. WHAT WORKS WELL:
               - Atmosphere: fog, mist, golden hour, rain, reflections
               - Lighting: backlighting, soft rim light, warm tungsten, neon glow
               - Camera: slow dolly, gentle pan, static tripod, tracking shot
               - Settings: noir, analog film look, cinematic color grading
            
            EXAMPLE GOOD PROMPTS:
            
            TOPIC: "Stressed at work"
            PROMPT: "A person sits at a cluttered desk, their hands typing rapidly on a laptop keyboard. Papers and coffee cups scatter across the surface. The camera slowly pushes in from a medium shot, revealing a single desk lamp casting warm amber light across the workspace. Outside the window behind them, city lights blur in the darkness. The scene feels atmospheric with subtle film grain and soft shadows. Gentle ambient office sounds. Static camera with slow dolly movement, shallow depth of field."
            
            TOPIC: "Coffee and productivity"
            PROMPT: "Steam rises slowly from a ceramic coffee cup sitting on a wooden desk. Morning sunlight streams through a nearby window, catching the wisps of steam in golden light. The camera orbits slowly around the cup at 45 degrees, keeping the focus sharp on the rim while the background blurs into soft bokeh. Dust particles drift through the sunbeam. Warm color palette with soft contrast, peaceful morning atmosphere."
            
            TOPIC: "City life"
            PROMPT: "Rain falls gently on a wet city street at night, neon signs reflecting in puddles on the pavement. The camera tracks slowly forward at eye level as raindrops create ripples in the water. Colorful neon lights from storefronts cast red and blue glows across the wet surfaces. Steam rises from a street vent in the background. Moody noir atmosphere with high contrast lighting, cinematic color grading, subtle halation around bright lights."
            
            WRITE THE PROMPT AS ONE PARAGRAPH. NO BULLET POINTS. NO LISTS.

            Return ONLY a JSON array:
            [
              {{
                "start_time": 2.5,
                "end_time": 5.0,
                "visual_prompt": "detailed cinematic prompt here",
                "reason": "brief reason here"
              }}
            ]
            """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-5-mini-2025-08-07",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert cinematographer and video editor specializing in LTX-2 AI video generation. You write prompts that follow Lightricks' official guidelines: single flowing paragraph, present tense, under 200 words, starting with main action, including camera movement, lighting, and atmosphere. You avoid text/logos, detailed faces, multiple people, complex physics, and emotional labels. You focus on visual storytelling with single subjects, atmospheric lighting, and clean camera language."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            return self._parse_gpt_response(response.choices[0].message.content, chunk)
            
        except Exception as e:
            print(f"    ❌ GPT analysis failed: {e}")
            return []
    
    def _parse_gpt_response(self, response: str, chunk: List[Dict]) -> List[BRollRegion]:
        """Parse GPT response and validate timestamps"""
        try:
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if not json_match:
                print("    ⚠️ No JSON array found in GPT response")
                return []
            
            broll_data = json.loads(json_match.group())
            regions = []
            
            # Build list of valid boundaries for validation
            valid_times = set()
            for seg in chunk:
                valid_times.add(seg.get("start", 0))
                valid_times.add(seg.get("end", 0))
                if "words" in seg and seg["words"]:
                    for word in seg["words"]:
                        valid_times.add(word.get("start", 0))
                        valid_times.add(word.get("end", 0))
            valid_times = sorted(list(valid_times))
            
            for item in broll_data:
                try:
                    start_time = item["start_time"]
                    end_time = item["end_time"]
                    
                    chunk_start = chunk[0].get("start", 0) if chunk else 0
                    chunk_end = chunk[-1].get("end", 0) if chunk else 0
                    
                    if start_time < chunk_start or end_time > chunk_end:
                        print(f"    ⚠️ Skipping B-roll with out-of-range times")
                        continue
                    
                    if end_time <= start_time:
                        print(f"    ⚠️ Skipping B-roll with invalid duration")
                        continue
                    
                    region = BRollRegion(
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        reason=item["reason"],
                        confidence=0.9,
                        prompt=item["visual_prompt"]
                    )
                    regions.append(region)
                    print(f"    📍 B-roll at {start_time:.1f}s-{end_time:.1f}s ({region.duration:.1f}s)")
                except KeyError as e:
                    print(f"    ⚠️ Missing field in GPT response: {e}")
                    continue
            
            return regions
            
        except json.JSONDecodeError as e:
            print(f"    ❌ Failed to parse GPT JSON: {e}")
            return []
        except Exception as e:
            print(f"    ❌ Error parsing GPT response: {e}")
            return []


class LTX2VideoGenerator:
    """LTX-2 Fast video generator for B-roll production."""
    
    def __init__(
        self,
        resolution: str = "480p",
        aspect_ratio: str = "16:9",
        fast_mode: bool = True,
    ):
        """Initialize LTX-2 video generator.
        
        Args:
            resolution: One of "480p", "720p", "1080p"
            aspect_ratio: "16:9" or "9:16"
            fast_mode: Use faster generation settings
        """
        from video_gen import create_ltx2_generator
        
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        
        try:
            self._gen = create_ltx2_generator(
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                fast_mode=fast_mode,
            )
            print(f"🎬 LTX-2 Video Generator ready:")
            print(f"  - Resolution: {resolution} ({aspect_ratio})")
            print(f"  - Fast mode: {fast_mode}")
            print(f"  - Model: Lightricks/LTX-2")
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "401" in str(e):
                print("\n❌ HuggingFace authentication failed!")
                print("   Fix: huggingface-cli login")
                print("   Or:  export HF_TOKEN='your-token'")
            elif "diffusers" in error_msg and "upgrade" in error_msg:
                print("\n❌ diffusers version issue!")
                print("   Fix: pip install -U diffusers>=0.32.0")
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LTX-2 generator: {e}")
    
    def generate_broll_video(self, prompt: str, duration: float, output_path: str) -> bool:
        """Generate B-roll video with LTX-2.
        
        Args:
            prompt: Visual description for the B-roll
            duration: Target duration in seconds
            output_path: Where to save the video
            
        Returns:
            True if successful, False otherwise
        """
        print(f"🎬 Generating B-roll: {prompt[:60]}... ({duration:.1f}s)")
        
        try:
            self._gen.generate_broll_for_region(
                prompt=prompt,
                duration=duration,
                output_path=output_path,
            )
            print(f"  ✅ Generated: {output_path}")
            return True
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                print(f"  ❌ CUDA Out of Memory!")
                print(f"      Try: export LTX2_RESOLUTION=480p")
                print(f"      Or:  Use a GPU with more VRAM")
            elif "authentication" in error_msg or "401" in str(e):
                print(f"  ❌ HuggingFace authentication failed!")
                print(f"      Run: huggingface-cli login")
            else:
                print(f"  ❌ LTX-2 generation error: {e}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"  ❌ LTX-2 generation error: {e}")
            import traceback
            traceback.print_exc()
            return False


# Backwards compatibility aliases
class Wan22VideoGenerator(LTX2VideoGenerator):
    """Backwards compatibility wrapper. Redirects to LTX2VideoGenerator."""
    
    def __init__(self, *args, **kwargs):
        print("[DEPRECATED] Wan22VideoGenerator is deprecated. Using LTX2VideoGenerator instead.", flush=True)
        # Remove old Wan2.2 specific args
        kwargs.pop("wan22_path", None)
        kwargs.pop("model_type", None)
        super().__init__(*args, **kwargs)


class Wan22LightX2VVideoGenerator(LTX2VideoGenerator):
    """Backwards compatibility wrapper. Redirects to LTX2VideoGenerator."""
    
    def __init__(self, *args, **kwargs):
        print("[DEPRECATED] Wan22LightX2VVideoGenerator is deprecated. Using LTX2VideoGenerator instead.", flush=True)
        kwargs.pop("models_dir", None)
        super().__init__(*args, **kwargs)


class ProductionBRollPipeline:
    """Complete production B-roll pipeline with LTX-2 integration"""
    
    def __init__(
        self,
        resolution: str = "480p",
        aspect_ratio: str = "16:9",
        fast_mode: bool = True,
    ):
        """Initialize the production pipeline.
        
        Args:
            resolution: Video resolution (480p, 720p, 1080p)
            aspect_ratio: "16:9" or "9:16"
            fast_mode: Use faster generation settings
        """
        self.analyzer = ProductionBRollAnalyzer()
        self.generator = LTX2VideoGenerator(
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            fast_mode=fast_mode,
        )
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio

    def process_clip(self, clip_data: Dict) -> bool:
        """Process a complete clip with B-roll generation"""
        print(f"\n🎬 Processing clip: {clip_data.get('hook', 'Unknown')[:50]}...")
        
        try:
            # Extract segments from clip data
            segments = clip_data.get("segments", [])
            if not segments:
                print("❌ No segments found in clip data")
                return False
            
            # Analyze content for B-roll opportunities
            broll_regions = self.analyzer.analyze_content_for_broll(segments)
            
            if not broll_regions:
                print("⚠️ No B-roll opportunities identified")
                return False
            
            print(f"✅ Found {len(broll_regions)} B-roll opportunities")
            
            # Generate B-roll videos
            successful_generations = 0
            generation_errors = []
            
            for i, region in enumerate(broll_regions, 1):
                print(f"\n📹 Generating B-roll {i}/{len(broll_regions)}:")
                print(f"  Time: {region.start_time:.1f}s - {region.end_time:.1f}s ({region.duration:.1f}s)")
                print(f"  Prompt: {region.prompt[:80]}...")
                
                output_path = f"broll_{i}_{int(region.start_time)}s.mp4"
                
                try:
                    success = self.generator.generate_broll_video(
                        region.prompt,
                        region.duration,
                        output_path
                    )
                    
                    if success:
                        region.broll_path = output_path
                        successful_generations += 1
                        print(f"  ✅ Generated: {output_path}")
                    else:
                        error_msg = f"B-roll generation returned False for region {i}"
                        generation_errors.append(error_msg)
                        print(f"  ❌ {error_msg}")
                        
                except Exception as e:
                    error_msg = f"Exception during B-roll {i} generation: {e}"
                    generation_errors.append(error_msg)
                    print(f"  ❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
            
            # Check if we have any successful generations
            if successful_generations == 0:
                error_summary = f"All B-roll generation failed. Errors: {'; '.join(generation_errors)}"
                raise RuntimeError(error_summary)
            
            # Warn if some generations failed
            if generation_errors:
                print(f"\n⚠️ {len(generation_errors)} B-roll generations failed:")
                for error in generation_errors:
                    print(f"  - {error}")
            
            print(f"\n✅ Successfully generated {successful_generations}/{len(broll_regions)} B-roll videos")
            
            # Report results
            for region in broll_regions:
                if region.broll_path and os.path.exists(region.broll_path):
                    file_size = os.path.getsize(region.broll_path) / 1024
                    print(f"  📁 {region.broll_path}: {file_size:.1f}KB")
            
            return successful_generations > 0
            
        except Exception as e:
            print(f"❌ Error processing clip: {e}")
            import traceback
            traceback.print_exc()
            return False


# Configuration
PRODUCTION_CONFIG = {
    "resolution": os.getenv("LTX2_RESOLUTION", "1080p"),  # Default to 1080p for quality
    "aspect_ratio": os.getenv("LTX2_ASPECT_RATIO", "9:16"),  # Default to vertical for TikTok
    "fast_mode": os.getenv("LTX2_FAST_MODE", "false").lower() in ("true", "1", "yes"),  # Default to quality mode (30 steps)
}


def create_production_pipeline(
    resolution: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    fast_mode: Optional[bool] = None,
) -> ProductionBRollPipeline:
    """Factory function to create production B-roll pipeline with LTX-2.
    
    Args:
        resolution: Video resolution (480p, 720p, 1080p, 4K)
        aspect_ratio: "16:9" or "9:16"
        fast_mode: Use faster generation settings
        
    Returns:
        Configured ProductionBRollPipeline instance
    """
    return ProductionBRollPipeline(
        resolution=resolution or PRODUCTION_CONFIG["resolution"],
        aspect_ratio=aspect_ratio or PRODUCTION_CONFIG["aspect_ratio"],
        fast_mode=fast_mode if fast_mode is not None else PRODUCTION_CONFIG["fast_mode"],
    )


# Backwards compatibility
# Keep old function name but redirect to new implementation
create_production_pipeline_with_wan22 = create_production_pipeline


def setup_ltx2_environment():
    """Helper function to set up LTX-2 environment"""
    print("🔧 Setting up LTX-2 Environment")
    print("=" * 50)
    
    print("1. Install dependencies:")
    print("   pip install diffusers>=0.32.0 transformers accelerate")
    
    print("\n2. Set environment variables:")
    print("   export OPENAI_API_KEY=your_openai_api_key")
    print("   export LTX2_RESOLUTION=480p  # Options: 480p, 720p, 1080p, 4K")
    print("   export LTX2_ASPECT_RATIO=16:9  # Options: 16:9, 9:16")
    print("   export LTX2_FAST_MODE=true  # Use faster generation")
    
    print("\n3. Model will be auto-downloaded from HuggingFace on first use:")
    print("   - Lightricks/LTX-2")
    print("   - Requires HuggingFace CLI login: huggingface-cli login")
    
    print("\n4. Test installation:")
    print("   python -c \"from production_broll import create_production_pipeline; pipeline = create_production_pipeline()\"")
    
    print("\nRecommended GPU requirements:")
    print("- RTX 4090 (24GB VRAM) for 720p and below")
    print("- RTX 5090 / H100 for 1080p and 4K")
    print("- Enable LTX2_FAST_MODE=true for faster generation")


# Keep old function name for backwards compatibility
setup_wan22_environment = setup_ltx2_environment


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production B-roll Generator with LTX-2 Fast")
    parser.add_argument("--setup", action="store_true", help="Show setup instructions")
    parser.add_argument("--resolution", default="480p", help="Resolution: 480p, 720p, 1080p, 4K")
    parser.add_argument("--aspect-ratio", default="16:9", help="Aspect ratio: 16:9 or 9:16")
    parser.add_argument("--fast-mode", action="store_true", help="Enable fast mode")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_ltx2_environment()
        exit(0)
    
    print("🎬 Production B-roll System with LTX-2 Fast")
    print("=" * 50)
    
    try:
        pipeline = create_production_pipeline(
            resolution=args.resolution,
            aspect_ratio=args.aspect_ratio,
            fast_mode=args.fast_mode,
        )
        print("✅ Pipeline ready for production use")
        print("\nTo use this pipeline:")
        print("1. Import: from production_broll import create_production_pipeline")
        print("2. Create: pipeline = create_production_pipeline()")
        print("3. Process: pipeline.process_clip(your_clip_data)")
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n💡 Tips:")
    print("- Use --setup to see installation instructions")
    print("- Set LTX2_ASPECT_RATIO=9:16 for TikTok/Reels")
    print("- Set LTX2_FAST_MODE=true for faster generation")
