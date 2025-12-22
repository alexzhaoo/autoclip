#!/usr/bin/env python3


import os
import json
import subprocess
import requests
import tempfile
import shutil
from pathlib import Path
import traceback
from typing import List, Dict, Optional, Tuple
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
            min_spacing: Minimum seconds between B-roll placements (default 2.0, was 3.0)
        """
        if not regions:
            return regions
        
        # Sort by start time
        sorted_regions = sorted(regions, key=lambda r: r.start_time)
        
        filtered = []
        last_broll_time = -min_spacing  # Allow B-roll at start
        
        for region in sorted_regions:
            # Check spacing constraint (reduced from 3s to 2s for more placements)
            if region.start_time - last_broll_time >= min_spacing:
                # Constrain duration to 1-3 seconds (increased max from 2s to 3s)
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
                print(f"    ✅ Kept B-roll at {region.start_time:.1f}s (spacing: {region.start_time - (last_broll_time if filtered else -min_spacing):.1f}s)")
            else:
                spacing = region.start_time - last_broll_time
                print(f"    ⏭️  Filtered out B-roll at {region.start_time:.1f}s (spacing {spacing:.1f}s < {min_spacing:.1f}s)")
        
        return filtered
    
    def _analyze_chunk_with_gpt(self, chunk: List[Dict]) -> List[BRollRegion]:
        """Analyze a chunk of segments with GPT-4"""
        # Create transcript with timing information
        chunk_start = chunk[0].get("start", 0) if chunk else 0
        chunk_end = chunk[-1].get("end", 0) if chunk else 0
        
        # Build transcript with word-level timestamps for precise alignment
        transcript_lines = []
        for seg in chunk:
            # Use word-level timestamps if available for better precision
            if "words" in seg and seg["words"] and len(seg["words"]) > 0:
                # Format: show first word timestamp, then words, then last word timestamp
                # This is cleaner than timestamp on every word
                first_word_time = seg["words"][0].get("start", 0)
                last_word_time = seg["words"][-1].get("end", 0)
                words_text = " ".join([w.get("word", "") for w in seg["words"]])
                transcript_lines.append(f"[{first_word_time:.2f}s - {last_word_time:.2f}s] {words_text}")
            else:
                # Fallback to segment-level if words not available
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                seg_text = seg.get("text", "")
                transcript_lines.append(f"[{seg_start:.2f}s - {seg_end:.2f}s] {seg_text}")
        
        transcript_with_timing = "\n".join(transcript_lines)
        
        # Create clean transcript for context
        clean_transcript = " ".join([seg.get("text", "").strip() for seg in chunk])

        # Enhanced prompt optimized for Wan2.2 generation with precise timing alignment
        prompt = f"""
            Analyze this video transcript and identify 3 short B-roll opportunities (2-4 seconds) that enhance viewer engagement.
            
            FULL CONTEXT (Read this first to understand the story flow):
            "{clean_transcript}"

            TIMED TRANSCRIPT (Use this ONLY for precise start/end times):
            {transcript_with_timing}
            
            CRITICAL TIMING RULES:
            1. START_TIME must be within the time ranges shown above (e.g., if you see [2.54s - 5.32s], choose any time between 2.54 and 5.32)
            2. END_TIME must be 2-4 seconds after START_TIME and fall within or end at a phrase boundary
            3. B-roll duration should be 2-4 seconds
            4. Choose the EXACT moment when the relevant word/phrase begins speaking
            5. Times must be between {chunk_start:.1f} and {chunk_end:.1f}
            6. Choose moments where visuals would enhance understanding without disrupting the narrative flow
            
            For each B-roll opportunity, provide:

            START_TIME – MUST be an exact timestamp from the TIMED TRANSCRIPT above (in seconds)
            END_TIME – MUST be a timestamp from the TIMED TRANSCRIPT above (in seconds)
            VISUAL_PROMPT – vivid, cinematic (add the words "Symmetrical Composition" at the end of each prompt) 
            reason - why this clip fits and which transcript line it illustrates

            Suggest clear, engaging visuals with DYNAMIC MOVEMENT that reflect the transcript's ideas, themes, or emotions.
            IMPORTANT: Use the FULL CONTEXT to understand the meaning of split sentences. Do not generate clips for isolated fragments.
            ALWAYS specify motion: camera movement (pan, tilt, zoom, dolly, tracking) AND/OR subject movement (flowing, rotating, walking, falling, rising, transforming).
            Ensure Symmetrical Composition where natural.

            Use diverse visual categories WITH MOTION:
            - Lifestyle & work: people walking, gesturing, objects being manipulated
            - Technology: screens animating, data flowing, lights pulsing, interfaces transitioning
            - Nature: wind blowing, water flowing, clouds drifting, leaves rustling, birds flying
            - Creative processes: materials transforming, liquids mixing, brush strokes appearing
            - Abstract: particles flowing, shapes morphing, waves rippling, energy radiating


            Specify camera angle (e.g., close-up, wide, tracking), lighting, color, and textures.
            Avoid: Overused visuals like generic close-ups of hands unless the transcript explicitly describes manual actions.

            Example VISUAL_PROMPTs (NOTE THE MOTION VERBS):
            Smooth aerial dolly shot gliding over a city at dusk, camera moving forward steadily, windows glowing, soft haze drifting, birds flying across symmetrical skyline.
            Macro shot of vibrant ink droplets spreading and swirling dynamically through clear water, colors bleeding outward, symmetrical flow patterns forming and evolving.
            Fluid tracking shot following a scientist walking through a bright lab, camera moving smoothly alongside, reflections shifting on glass panels, balanced composition.
            Slow sweeping pan across rolling ocean waves under golden light, water continuously rippling, foam flowing forward, camera tilting gently, symmetrical horizon line.


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
                model="gpt-4.1-2025-04-14",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert video editor specializing in engaging short-form content and AI video generation. You create detailed, cinematic prompts optimized for Wan2.2 text-to-video generation across diverse visual categories including scientific, abstract, nature, lifestyle, technology, and artistic themes."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            return self._parse_gpt_response(response.choices[0].message.content, chunk)
            
        except Exception as e:
            print(f"    ❌ GPT analysis failed: {e}")
            return []
    
    def _parse_gpt_response(self, response: str, chunk: List[Dict]) -> List[BRollRegion]:
        """Parse GPT response and validate timestamps align with segment boundaries"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                print("    ⚠️ No JSON array found in GPT response")
                return []
            
            broll_data = json.loads(json_match.group())
            regions = []
            
            # Build list of valid word and segment boundaries for validation
            valid_times = set()
            for seg in chunk:
                valid_times.add(seg.get("start", 0))
                valid_times.add(seg.get("end", 0))
                # Add word-level timestamps
                if "words" in seg and seg["words"] and len(seg["words"]) > 0:
                    for word in seg["words"]:
                        valid_times.add(word.get("start", 0))
                        valid_times.add(word.get("end", 0))
            valid_times = sorted(list(valid_times))
            
            # Ensure we have at least chunk boundaries
            if not valid_times:
                valid_times = [chunk_start, chunk_end]
            
            for item in broll_data:
                try:
                    start_time = item["start_time"]
                    end_time = item["end_time"]
                    
                    # Validate timestamps are reasonable
                    chunk_start = chunk[0].get("start", 0) if chunk else 0
                    chunk_end = chunk[-1].get("end", 0) if chunk else 0
                    
                    if start_time < chunk_start or end_time > chunk_end:
                        print(f"    ⚠️ Skipping B-roll with out-of-range times: {start_time:.1f}s-{end_time:.1f}s (chunk: {chunk_start:.1f}s-{chunk_end:.1f}s)")
                        continue
                    
                    if end_time <= start_time:
                        print(f"    ⚠️ Skipping B-roll with invalid duration: {start_time:.1f}s-{end_time:.1f}s")
                        continue
                    
                    # Snap to nearest word/segment boundaries (within 0.3s tolerance for word precision)
                    def snap_to_boundary(time, boundaries, tolerance=0.3):
                        closest = min(boundaries, key=lambda x: abs(x - time))
                        if abs(closest - time) <= tolerance:
                            return closest
                        return time
                    
                    original_start = start_time
                    original_end = end_time
                    start_time = snap_to_boundary(start_time, valid_times)
                    end_time = snap_to_boundary(end_time, valid_times)
                    
                    if start_time != original_start or end_time != original_end:
                        print(f"    🔧 Adjusted timing: {original_start:.2f}s-{original_end:.2f}s → {start_time:.2f}s-{end_time:.2f}s (snapped to word boundaries)")
                    
                    region = BRollRegion(
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        reason=item["reason"],
                        confidence=0.9,  # High confidence for GPT suggestions
                        prompt=item["visual_prompt"]
                    )
                    regions.append(region)
                    print(f"    📍 B-roll at {start_time:.1f}s-{end_time:.1f}s ({region.duration:.1f}s): {item['reason'][:50]}")
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

class Wan22VideoGenerator:
    """Wan2.2 video generator with multiple model support"""
    
    def __init__(self, wan22_path: Optional[str] = None, model_type: str = "ti2v-5B", aspect_ratio: str = "16:9"):
        self.wan22_path = wan22_path or os.getenv("WAN22_PATH", "/workspace/Wan2.2" if os.path.exists("/workspace") else "./Wan2.2")
        self.model_type = model_type
        self.aspect_ratio = aspect_ratio
        
        # ti2v-5B model configuration with dynamic aspect ratio support
        if aspect_ratio == "9:16":
            size = "704*1280"    # 9:16 portrait for TikTok/Reels
            aspect_desc = "9:16 portrait"
        else:
            size = "704*1280"    # 16:9 landscape (matches your working command)
            aspect_desc = "16:9 landscape"
        
        self.model_config = {
            "task": "ti2v-5B",
            "size": size,
            "min_vram": 24,
            "description": f"Text/Image-to-video, 5B parameters, RTX 4090 compatible, {aspect_desc}"
        }
        
        # Validate Wan2.2 installation
        try:
            self.wan22_available = self._validate_wan22_installation()
        except Exception as e:
            raise RuntimeError(f"Wan2.2 validation failed: {e}")
        
        print("🎬 Wan2.2 Video Generator ready:")
        print(f"  - Wan2.2 path: {self.wan22_path}")
        print(f"  - Model: {model_type} ({self.model_config['description']})")
        print("  - Wan2.2 validated: ✅")
    
    def _validate_wan22_installation(self) -> bool:
        """Validate Wan2.2 installation and model availability"""
        try:
            # Check if Wan2.2 directory exists
            wan22_dir = Path(self.wan22_path)
            if not wan22_dir.exists():
                raise FileNotFoundError(f"Wan2.2 directory not found: {self.wan22_path}")
            
            # Check if generate.py exists
            generate_script = wan22_dir / "generate.py"
            if not generate_script.exists():
                raise FileNotFoundError(f"generate.py not found in {self.wan22_path}")
            
            # Check if ti2v-5B model directory exists
            workspace_dir = Path("/workspace") if os.path.exists("/workspace") else Path(".")
            model_dir = workspace_dir / "Wan2.2-TI2V-5B"
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}. Please download the ti2v-5B model to {model_dir}")
            
            print("✅ Wan2.2 installation validated: ti2v-5B")
            return True
            
        except Exception as e:
            print(f"❌ Error validating Wan2.2 installation: {e}")
            raise  # Re-raise the exception instead of returning False
    
    def generate_broll_video(self, prompt: str, duration: float, output_path: str) -> bool:
        """Generate B-roll video with Wan2.2"""
        print(f"🎬 Generating B-roll: {prompt[:60]}... ({duration:.1f}s)")
        
        print(f"  🌟 Using Wan2.2 {self.model_type} generation...")
        return self._generate_with_wan22(prompt, duration, output_path)
    
    def _generate_with_wan22(self, prompt: str, duration: float, output_path: str) -> bool:
        """Generate video using Wan2.2 ti2v-5B model"""
        try:
            config = self.model_config
            
            # Model is in workspace root, not inside Wan2.2 directory
            workspace_dir = Path("/workspace") if os.path.exists("/workspace") else Path(".")
            model_dir = workspace_dir / "Wan2.2-TI2V-5B"
            
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build Wan2.2 command matching your working configuration
            cmd = [
                "python", 
                str(Path(self.wan22_path) / "generate.py"),
                "--task", config["task"],
                "--size", config["size"],
                "--ckpt_dir", str(model_dir),
                # "--offload_model", "True",
                # "--convert_model_dtype", 
                # "--t5_cpu",
                "--prompt", prompt
            ]
            
            # For RTX 5090 (24GB+), keep T5 on GPU for better performance
            # Only use t5_cpu if you have memory issues
            # if self.model_type == "ti2v-5B":
            #     cmd.extend(["--t5_cpu"])
            
            # Add prompt extension if API key is available
            dash_api_key = os.getenv("DASH_API_KEY")
            if dash_api_key:
                cmd.extend([
                    "--use_prompt_extend",
                    "--prompt_extend_method", "dashscope"
                ]) 
            
            print(f"    🔧 Running: {' '.join(cmd[:6])}... (full command with {len(cmd)} args)")
            print(f"    📂 Working directory: {self.wan22_path}")
            
            # Record existing MP4 files before generation to identify new ones
            wan22_dir = Path(self.wan22_path)
            existing_files = set()
            
            # Check multiple possible output locations
            search_dirs = [
                wan22_dir,  # Main Wan2.2 directory
                wan22_dir / "output",  # Common output subdirectory 
                wan22_dir / "outputs",  # Alternative output subdirectory
                wan22_dir / "generated",  # Alternative output subdirectory
            ]
            
            for search_dir in search_dirs:
                if search_dir.exists():
                    existing_files.update(search_dir.glob("**/*.mp4"))
            
            print(f"    📊 Found {len(existing_files)} existing MP4 files before generation")
            
            # Set environment
            env = os.environ.copy()
            env["PYTHONPATH"] = self.wan22_path
            if dash_api_key:
                env["DASH_API_KEY"] = dash_api_key
            
            print("    ⏳ Starting Wan2.2 generation (model loading may take 30-120 seconds)...")
            start_time = time.time()
            
            # Run generation and capture output for better error reporting
            result = subprocess.run(
                cmd,
                cwd=self.wan22_path,
                env=env,
                capture_output=True,  # Capture output for error reporting
                text=True,
                timeout=1500  # 25 minute timeout
            )
            
            elapsed_time = time.time() - start_time
            print(f"    ⏱️ Generation process completed in {elapsed_time:.1f} seconds (exit code: {result.returncode})")
            
            if result.returncode == 0:
                # Find new MP4 files created during generation
                new_files = set()
                for search_dir in search_dirs:
                    if search_dir.exists():
                        new_files.update(search_dir.glob("**/*.mp4"))
                
                generated_files = list(new_files - existing_files)
                
                print(f"    🔍 Found {len(generated_files)} new MP4 files after generation")
                for f in generated_files:
                    file_size = f.stat().st_size if f.exists() else 0
                    print(f"        📹 {f} ({file_size/1024:.1f}KB)")
                
                if generated_files:
                    # Get the most recent video file
                    latest_video = max(generated_files, key=lambda p: p.stat().st_mtime)
                    
                    print(f"    📁 Using latest generated file: {latest_video}")
                    
                    # Move to desired output path
                    try:
                        shutil.move(str(latest_video), output_path)
                        print(f"    📦 Moved file to: {output_path}")
                    except Exception as e:
                        # If move fails, try copy instead
                        try:
                            shutil.copy2(str(latest_video), output_path)
                            print(f"    📦 Copied file to: {output_path}")
                            # Try to remove original after successful copy
                            try:
                                latest_video.unlink()
                            except:
                                pass  # Don't fail if we can't clean up
                        except Exception as e2:
                            raise RuntimeError(f"Failed to move/copy generated video from {latest_video} to {output_path}: {e2}")
                    
                    # Verify file exists and has content
                    if not os.path.exists(output_path):
                        raise RuntimeError(f"Output file was not created at {output_path}")
                    
                    file_size = os.path.getsize(output_path)
                    if file_size <= 10000:  # At least 10KB
                        raise RuntimeError(f"Generated file too small ({file_size} bytes) - likely corrupted or generation failed")
                    
                    print(f"  ✅ Wan2.2 B-roll generated successfully: {output_path} ({file_size/1024:.1f}KB)")
                    return True
                else:
                    # No new files found - provide detailed debugging info
                    print(f"    ❌ No new MP4 files found after generation!")
                    print(f"    🔍 Searched directories:")
                    for search_dir in search_dirs:
                        if search_dir.exists():
                            all_files = list(search_dir.glob("**/*"))
                            mp4_files = list(search_dir.glob("**/*.mp4"))
                            print(f"        📂 {search_dir}: {len(all_files)} total files, {len(mp4_files)} MP4 files")
                        else:
                            print(f"        📂 {search_dir}: (doesn't exist)")
                    
                    raise RuntimeError(f"No new MP4 files found in expected directories after successful generation")
            else:
                error_msg = f"Wan2.2 generation failed (exit code {result.returncode})"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nSTDOUT: {result.stdout}"
                print(f"    ❌ {error_msg}")
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            print("  ⏱️ Wan2.2 generation timed out")
            return False
        except Exception as e:
            print(f"  ❌ Wan2.2 generation error: {e}")
            return False


class Wan22LightX2VVideoGenerator:
    """Wan2.2 T2V 4-step distill generator via LightX2V (Dual-LoRA)."""

    def __init__(self, models_dir: Optional[str] = None):
        from video_gen import Wan22LightX2VGenerator

        self._gen = Wan22LightX2VGenerator(models_dir=models_dir or os.getenv("WAN22_MODELS_DIR", "./models"))

    def generate_broll_video(self, prompt: str, duration: float, output_path: str) -> bool:
        try:
            _ = duration
            self._gen.generate_clip(prompt=prompt, output_path=output_path)
            return True
        except Exception as e:
            print(f"  ❌ LightX2V generation error: {e}")
            return False

class ProductionBRollPipeline:
    """Complete production B-roll pipeline with Wan2.2 integration"""
    
    def __init__(self, wan22_path: Optional[str] = None, model_type: str = "ti2v-5B", aspect_ratio: str = "16:9"):
        self.analyzer = ProductionBRollAnalyzer()
        self.generator = self._init_generator(wan22_path, model_type, aspect_ratio)
        self.wan22_path = wan22_path
        self.model_type = model_type
        self.aspect_ratio = aspect_ratio

    def _init_generator(self, wan22_path: Optional[str], model_type: str, aspect_ratio: str):
        prefer_lightx2v = os.getenv("WAN22_BACKEND", "lightx2v").lower() == "lightx2v"
        if prefer_lightx2v:
            try:
                print("🌟 Using LightX2V Wan2.2 T2V 4-step distill (Dual-LoRA) backend")
                return Wan22LightX2VVideoGenerator()
            except Exception as e:
                tb = traceback.format_exc()
                print(
                    "⚠️ LightX2V backend unavailable, falling back to legacy Wan2.2 generate.py: "
                    f"{type(e).__name__}: {e}\n{tb}"
                )

                # Only attempt legacy fallback if the Wan2.2 repo exists.
                effective_wan22_path = wan22_path or os.getenv(
                    "WAN22_PATH", "/workspace/Wan2.2" if os.path.exists("/workspace") else "./Wan2.2"
                )
                try:
                    if not Path(effective_wan22_path).exists():
                        raise FileNotFoundError(f"Wan2.2 directory not found: {effective_wan22_path}")
                except Exception as path_err:
                    raise RuntimeError(
                        "B-roll generation backend initialization failed. "
                        "LightX2V failed to initialize, and legacy Wan2.2 fallback is unavailable. "
                        f"LightX2V error: {e}. "
                        f"Legacy check error: {path_err}. "
                        "Fix options: (1) rerun ./setup_wan.sh and ensure 'python -c \"from lightx2v import LightX2VPipeline\"' works, "
                        "or (2) set WAN22_BACKEND=legacy and clone Wan2.2 at WAN22_PATH."
                    )

        return Wan22VideoGenerator(wan22_path, model_type, aspect_ratio)
    
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
            
            # Generate final composite video
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
    "wan22_path": os.getenv("WAN22_PATH", "/workspace/Wan2.2" if os.path.exists("/workspace") else "./Wan2.2"),
    "model_type": os.getenv("WAN22_MODEL", "t2v-A14B"),
    "aspect_ratio": os.getenv("BROLL_ASPECT_RATIO", "16:9"),  # Use 16:9 landscape since that's working
}

def create_production_pipeline() -> ProductionBRollPipeline:
    """Factory function to create production B-roll pipeline with Wan2.2"""
    return ProductionBRollPipeline(
        wan22_path=PRODUCTION_CONFIG["wan22_path"],
        model_type=PRODUCTION_CONFIG["model_type"],
        aspect_ratio=PRODUCTION_CONFIG["aspect_ratio"]
    )

def setup_wan22_environment():
    """Helper function to set up Wan2.2 environment"""
    print("🔧 Setting up Wan2.2 Environment")
    print("=" * 50)
    
    wan22_path = PRODUCTION_CONFIG["wan22_path"]
    
    print(f"1. Clone Wan2.2 repository to: {wan22_path}")
    print("   git clone https://github.com/Wan-Video/Wan2.2.git")
    print("   cd Wan2.2")
    print("   pip install -r requirements.txt")
    
    print("\n2. Download ti2v-5B model:")
    print("   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B")
    print("   (Requires ~24GB VRAM, works with RTX 4090)")
    
    print("\n3. Set environment variables:")
    print(f"   export WAN22_PATH={wan22_path}")
    print("   export WAN22_MODEL=ti2v-5B")
    print("   export BROLL_ASPECT_RATIO=9:16  # Use 9:16 for TikTok/Reels, 16:9 for landscape")
    print("   export OPENAI_API_KEY=your_openai_api_key")
    print("   export DASH_API_KEY=your_dashscope_api_key  # Optional, for prompt extension")
    
    print("\n4. Test installation:")
    print("   python -c \"from production_broll import create_production_pipeline; pipeline = create_production_pipeline()\"")
    
    print("\nRecommended GPU requirements:")
    print("- RTX 4090 (24GB VRAM) or better for ti2v-5B model")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production B-roll Generator with Wan2.2 ti2v-5B")
    parser.add_argument("--setup", action="store_true", help="Show setup instructions")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_wan22_environment()
        exit(0)
    
    print("🎬 Production B-roll System with Wan2.2 ti2v-5B")
    print("=" * 50)
    
    try:
        pipeline = create_production_pipeline()
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
    print("- Set BROLL_ASPECT_RATIO=9:16 for TikTok/Reels (default)")
    print("- Set DASH_API_KEY for enhanced prompt generation")