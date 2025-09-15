#!/usr/bin/env python3
"""
Production B-roll Generator v2.0
Integrates with Wan2.2 video generation models for high-quality video generation
Includes fast local fallback for development and testing
"""

import os
import json
import subprocess
import requests
import tempfile
import shutil
from pathlib import Path
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
    
    def _apply_timing_constraints(self, regions: List[BRollRegion]) -> List[BRollRegion]:
        """Apply timing constraints: 1 B-roll every ~10 seconds, max 2 seconds duration each"""
        if not regions:
            return regions
        
        # Sort by start time
        sorted_regions = sorted(regions, key=lambda r: r.start_time)
        
        filtered = []
        last_broll_time = -10.0  # Allow B-roll at start
        
        for region in sorted_regions:
            # Check 10-second spacing constraint
            if region.start_time - last_broll_time >= 10.0:
                # Constrain duration to 1-2 seconds
                max_duration = min(2.0, region.duration)
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
        
        return filtered
    
    def _analyze_chunk_with_gpt(self, chunk: List[Dict]) -> List[BRollRegion]:
        """Analyze a chunk of segments with GPT-4"""
        # Create transcript text from chunk
        transcript_text = " ".join([seg.get("text", "") for seg in chunk])
        
        # Enhanced prompt optimized for Wan2.2 generation
        prompt = f"""
        Analyze this video transcript and identify 2-3 specific moments where B-roll footage would enhance viewer engagement.

        TRANSCRIPT:
        {transcript_text}

        For each B-roll opportunity, provide:
        1. START_TIME: When to start B-roll (in seconds from chunk beginning)
        2. END_TIME: When to end B-roll  
        3. VISUAL_PROMPT: Detailed description optimized for Wan2.2 text-to-video model
        4. REASON: Why this moment needs B-roll

        Requirements for VISUAL_PROMPT (optimized for Wan2.2):
        - Highly detailed, cinematic descriptions suitable for AI video generation
        - Choose from these visual categories based on content:
          * Scientific: molecules, cells, brain activity, biological processes
          * Abstract: geometric patterns, flowing energy, conceptual animations
          * Nature: landscapes, weather, organic textures, natural phenomena
          * Lifestyle: hands working, objects in motion, daily activities (no faces)
          * Technology: digital interfaces, data visualization, futuristic concepts
          * Artistic: paint mixing, ink flowing, creative processes, artistic textures
        - Avoid human faces or talking people (audio continues over B-roll)
        - Include specific camera movements, lighting, and visual style cues
        - Mention colors, textures, and motion patterns
        - Target 1-2 second segments for optimal quality
        - Use cinematic terminology (close-up, wide shot, tracking shot, etc.)

        Examples of excellent Wan2.2 prompts for different categories:
        - Scientific: "Cinematic close-up of cortisol stress hormone molecules floating through bloodstream, ethereal blue and white scientific visualization with soft volumetric lighting, slow motion particles"
        - Abstract: "Flowing golden energy streams forming geometric patterns against deep purple background, smooth camera rotation, ethereal particle effects, cinematic lighting"
        - Nature: "Macro shot of morning dew droplets on green grass blades, soft natural lighting, gentle breeze movement, shallow depth of field, peaceful atmosphere"
        - Lifestyle: "Close-up of hands writing in notebook with pen, warm lighting, shallow focus on paper texture, smooth camera movement, cozy atmosphere"
        - Technology: "3D holographic data visualization with floating charts and graphs, blue and cyan color palette, smooth camera orbit, futuristic interface elements"
    
        Return ONLY a JSON array:
        [
          {{
            "start_time": 2.5,
            "end_time": 5.0,
            "visual_prompt": "detailed cinematic prompt here",
            "reason": "explanation here"
          }}
        ]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-5-mini-2025-08-07",
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
        """Parse GPT response and create BRollRegion objects"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                print("    ⚠️ No JSON array found in GPT response")
                return []
            
            broll_data = json.loads(json_match.group())
            regions = []
            
            # Calculate chunk start time
            chunk_start = chunk[0].get("start", 0) if chunk else 0
            
            for item in broll_data:
                try:
                    region = BRollRegion(
                        start_time=chunk_start + item["start_time"],
                        end_time=chunk_start + item["end_time"],
                        duration=item["end_time"] - item["start_time"],
                        reason=item["reason"],
                        confidence=0.9,  # High confidence for GPT suggestions
                        prompt=item["visual_prompt"]
                    )
                    regions.append(region)
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
    
    def __init__(self, wan22_path: Optional[str] = None, model_type: str = "ti2v-5B"):
        self.wan22_path = wan22_path or os.getenv("WAN22_PATH", "/workspace/Wan2.2" if os.path.exists("/workspace") else "./Wan2.2")
        self.model_type = model_type
        
        # Model configurations
        self.model_configs = {
            "t2v-A14B": {
                "task": "t2v-A14B",
                "size": "1280*720",
                "min_vram": 80,
                "description": "High-quality text-to-video, 14B parameters"
            },
            "i2v-A14B": {
                "task": "i2v-A14B", 
                "size": "1280*720",
                "min_vram": 80,
                "description": "High-quality image-to-video, 14B parameters"
            },
            "ti2v-5B": {
                "task": "ti2v-5B",
                "size": "1280*704", 
                "min_vram": 24,
                "description": "Text/Image-to-video, 5B parameters, RTX 4090 compatible"
            }
        }
        
        # Validate Wan2.2 installation
        try:
            self.wan22_available = self._validate_wan22_installation()
        except Exception as e:
            raise RuntimeError(f"Wan2.2 validation failed: {e}")
        
        print("🎬 Wan2.2 Video Generator ready:")
        print(f"  - Wan2.2 path: {self.wan22_path}")
        print(f"  - Model type: {model_type} ({self.model_configs[model_type]['description']})")
        print(f"  - Wan2.2 validated: ✅")
    
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
            
            # Check if model directory exists
            model_name_map = {
                "t2v-A14B": "Wan2.2-T2V-A14B",
                "i2v-A14B": "Wan2.2-I2V-A14B", 
                "ti2v-5B": "Wan2.2-TI2V-5B"
            }
            
            # Model is in workspace root, not inside Wan2.2 directory
            workspace_dir = Path("/workspace") if os.path.exists("/workspace") else Path(".")
            model_dir = workspace_dir / model_name_map[self.model_type]
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}. Please download the {self.model_type} model to {model_dir}")
            
            print(f"✅ Wan2.2 installation validated: {self.model_type}")
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
        """Generate video using Wan2.2 models"""
        try:
            config = self.model_configs[self.model_type]
            model_name_map = {
                "t2v-A14B": "Wan2.2-T2V-A14B",
                "i2v-A14B": "Wan2.2-I2V-A14B", 
                "ti2v-5B": "Wan2.2-TI2V-5B"
            }
            
            # Model is in workspace root, not inside Wan2.2 directory
            workspace_dir = Path("/workspace") if os.path.exists("/workspace") else Path(".")
            model_dir = workspace_dir / model_name_map[self.model_type]
            
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build Wan2.2 command
            cmd = [
                "python", 
                str(Path(self.wan22_path) / "generate.py"),
                "--task", config["task"],
                "--size", config["size"],
                "--ckpt_dir", str(model_dir),
                "--prompt", prompt,
                "--offload_model", "True",
                "--convert_model_dtype"
            ]
            
            # Add t5_cpu for 5B model to save memory
            if self.model_type == "ti2v-5B":
                cmd.extend(["--t5_cpu"])
            
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
            
            # Run generation with real-time output to see progress and debug file locations
            result = subprocess.run(
                cmd,
                cwd=self.wan22_path,
                env=env,
                capture_output= True,  # Allow real-time output to see where files are created
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
                print(f"    ❌ {error_msg}")
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            print("  ⏱️ Wan2.2 generation timed out")
            return False
        except Exception as e:
            print(f"  ❌ Wan2.2 generation error: {e}")
            return False

class ProductionBRollPipeline:
    """Complete production B-roll pipeline with Wan2.2 integration"""
    
    def __init__(self, wan22_path: Optional[str] = None, model_type: str = "ti2v-5B"):
        self.analyzer = ProductionBRollAnalyzer()
        self.generator = Wan22VideoGenerator(wan22_path, model_type)
        self.wan22_path = wan22_path
        self.model_type = model_type
    
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
    "model_type": os.getenv("WAN22_MODEL", "ti2v-5B"),  # ti2v-5B, t2v-A14B, i2v-A14B
}

def create_production_pipeline() -> ProductionBRollPipeline:
    """Factory function to create production B-roll pipeline with Wan2.2"""
    return ProductionBRollPipeline(
        wan22_path=PRODUCTION_CONFIG["wan22_path"],
        model_type=PRODUCTION_CONFIG["model_type"]
    )

def setup_wan22_environment():
    """Helper function to set up Wan2.2 environment"""
    print("🔧 Setting up Wan2.2 Environment")
    print("=" * 50)
    
    wan22_path = PRODUCTION_CONFIG["wan22_path"]
    model_type = PRODUCTION_CONFIG["model_type"]
    
    print(f"1. Clone Wan2.2 repository to: {wan22_path}")
    print("   git clone https://github.com/Wan-Video/Wan2.2.git")
    print("   cd Wan2.2")
    print("   pip install -r requirements.txt")
    
    print(f"\n2. Download {model_type} model:")
    if model_type == "ti2v-5B":
        print("   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B")
        print("   (Requires ~24GB VRAM, works with RTX 4090)")
    elif model_type == "t2v-A14B":
        print("   huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B")
        print("   (Requires ~80GB VRAM, high-end GPU needed)")
    elif model_type == "i2v-A14B":
        print("   huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B")
        print("   (Requires ~80GB VRAM, high-end GPU needed)")
    
    print(f"\n3. Set environment variables:")
    print(f"   export WAN22_PATH={wan22_path}")
    print(f"   export WAN22_MODEL={model_type}")
    print("   export OPENAI_API_KEY=your_openai_api_key")
    print("   export DASH_API_KEY=your_dashscope_api_key  # Optional, for prompt extension")
    
    print(f"\n4. Test installation:")
    print("   python -c \"from wan22_broll import create_production_pipeline; pipeline = create_production_pipeline()\"")
    
    print(f"\nRecommended GPU requirements:")
    print("- ti2v-5B: RTX 4090 (24GB VRAM) or better")
    print("- t2v-A14B/i2v-A14B: A100 (80GB VRAM) or H100")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production B-roll Generator with Wan2.2")
    parser.add_argument("--setup", action="store_true", help="Show setup instructions")
    parser.add_argument("--model", choices=["ti2v-5B", "t2v-A14B", "i2v-A14B"], 
                       default="ti2v-5B", help="Wan2.2 model to use")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_wan22_environment()
        exit(0)
    
    # Update config based on args
    PRODUCTION_CONFIG["model_type"] = args.model
    
    print("🎬 Production B-roll System with Wan2.2")
    print("=" * 50)
    print(f"Model: {args.model}")
    
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
    print("- Use --model ti2v-5B for consumer GPUs (RTX 4090)")  
    print("- Use --model t2v-A14B for high-end GPUs (A100/H100)")
    print("- Set DASH_API_KEY for enhanced prompt generation")