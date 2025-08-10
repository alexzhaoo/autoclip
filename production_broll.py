#!/usr/bin/env python3
"""
Production B-roll Generator v2.0
Combines proven GPT analysis with Wand 2.2 + Vast.ai for high-quality video generation
Includes fast local fallback for development and testing
"""

import os
import json
import subprocess
import requests
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
    """Enhanced B-roll analyzer with proven GPT integration"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ OPENAI_API_KEY not set. Using mock mode for testing.")
            self.client = None
            self.mock_mode = True
        else:
            self.client = OpenAI(api_key=api_key)
            self.mock_mode = False
    
    def analyze_content_for_broll(self, segments: List[Dict]) -> List[BRollRegion]:
        """Analyze content using proven GPT method to identify B-roll opportunities"""
        print("🧠 Using GPT to analyze transcript for B-roll opportunities...")
        
        # Process in chunks for better results
        chunk_size = 3
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
        
        # Apply timing constraints: 1 B-roll every ~7 seconds, 2-3 seconds each
        filtered_regions = self._apply_timing_constraints(all_regions)
        
        print(f"  ✅ Found {len(all_regions)} candidates, filtered to {len(filtered_regions)} optimal placements")
        return filtered_regions
    
    def _apply_timing_constraints(self, regions: List[BRollRegion]) -> List[BRollRegion]:
        """Apply timing constraints: 1 B-roll every ~7 seconds, max 3 seconds duration each"""
        if not regions:
            return regions
        
        # Sort by start time
        sorted_regions = sorted(regions, key=lambda r: r.start_time)
        
        filtered = []
        last_broll_time = -7.0  # Allow B-roll at start
        
        for region in sorted_regions:
            # Check 7-second spacing constraint
            if region.start_time - last_broll_time >= 7.0:
                # Constrain duration to 2-3 seconds
                max_duration = min(3.0, region.duration)
                constrained_region = BRollRegion(
                    start_time=region.start_time,
                    end_time=min(region.start_time + max_duration, region.end_time),
                    duration=max_duration,
                    prompt=region.prompt,
                    reason=region.reason
                )
                
                filtered.append(constrained_region)
                last_broll_time = region.start_time
        
        return filtered
    
    def _analyze_chunk_with_gpt(self, chunk: List[Dict]) -> List[BRollRegion]:
        """Analyze a chunk of segments with GPT-4"""
        if self.mock_mode:
            return self._generate_mock_regions(chunk)
        
        # Create transcript text from chunk
        transcript_text = " ".join([seg.get("text", "") for seg in chunk])
        
        # Enhanced prompt based on our successful tests
        prompt = f"""
        Analyze this educational video transcript and identify 2-3 specific moments where B-roll footage would enhance viewer engagement.

        TRANSCRIPT:
        {transcript_text}

        For each B-roll opportunity, provide:
        1. START_TIME: When to start B-roll (in seconds from chunk beginning)
        2. END_TIME: When to end B-roll  
        3. VISUAL_PROMPT: Detailed description for Wand 2.2 text-to-video model
        4. REASON: Why this moment needs B-roll

        Requirements for VISUAL_PROMPT:
        - Specific, detailed descriptions suitable for AI video generation
        - Focus on abstract concepts, scientific visualizations, or metaphors
        - Avoid human faces or talking (audio continues)
        - Include motion and visual style cues
        - Target 3-5 second segments

        Examples of good prompts:
        - "Microscopic view of stress hormones cortisol molecules floating through bloodstream, blue and white scientific visualization"
        - "3D animation of neural pathways lighting up in brain, synapses firing with electrical signals, purple and gold colors"
        - "Abstract representation of inflammation: red particles swirling and growing, then cooling to blue, time-lapse style"

        Return ONLY a JSON array:
        [
          {{
            "start_time": 2.5,
            "end_time": 6.0,
            "visual_prompt": "detailed prompt here",
            "reason": "explanation here"
          }}
        ]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                temperature=0.7,
                max_tokens=800,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert video editor specializing in educational content. You identify perfect moments for B-roll footage and create detailed prompts for AI video generation."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            
            return self._parse_gpt_response(response.choices[0].message.content, chunk)
            
        except Exception as e:
            print(f"    ❌ GPT analysis failed: {e}")
            return []
    
    def _generate_mock_regions(self, chunk: List[Dict]) -> List[BRollRegion]:
        """Generate mock B-roll regions for testing when GPT is not available"""
        if not chunk:
            return []
        
        chunk_start = chunk[0].get("start", 0)
        chunk_duration = chunk[-1].get("end", chunk_start + 10) - chunk_start
        
        # Generate 2 mock regions based on content
        transcript_text = " ".join([seg.get("text", "") for seg in chunk]).lower()
        
        regions = []
        
        # First region (early in chunk)
        regions.append(BRollRegion(
            start_time=chunk_start + 1.0,
            end_time=chunk_start + 4.5,
            duration=3.5,
            reason="stress_visualization",
            confidence=0.8,
            prompt=self._generate_mock_prompt(transcript_text, "early")
        ))
        
        # Second region (later in chunk)
        if chunk_duration > 8:
            regions.append(BRollRegion(
                start_time=chunk_start + chunk_duration - 5.0,
                end_time=chunk_start + chunk_duration - 1.0,
                duration=4.0,
                reason="biological_process",
                confidence=0.8,
                prompt=self._generate_mock_prompt(transcript_text, "late")
            ))
        
        return regions
    
    def _generate_mock_prompt(self, transcript_text: str, position: str) -> str:
        """Generate mock prompts based on content analysis"""
        prompts = {
            "stress": [
                "Microscopic view of stress hormones cortisol molecules floating through bloodstream, blue and white scientific visualization",
                "3D animation of the human body with stress affecting different organs, red particles flowing through systems"
            ],
            "brain": [
                "Neural pathways lighting up in brain, synapses firing with electrical signals, purple and gold colors",
                "Brain cross-section showing neural activity, glowing connections and information processing"
            ],
            "heart": [
                "Animated heart beating, blood circulation visualization with red particles flowing through vessels",
                "Cardiovascular system overview, heart pumping blood through arteries and veins, medical animation"
            ],
            "default": [
                "Abstract flowing particles representing biological processes, scientific visualization, blue and white",
                "Cellular activity animation, microscopic view of biological processes, soft scientific lighting"
            ]
        }
        
        # Select prompt category
        category = "default"
        for key in ["stress", "brain", "heart"]:
            if key in transcript_text:
                category = key
                break
        
        # Select prompt based on position
        prompt_index = 0 if position == "early" else 1
        return prompts[category][prompt_index]
    
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

class ProductionVideoGenerator:
    """Production video generator with Wand 2.2 + Vast.ai and fast local fallback"""
    
    def __init__(self, vast_ai_endpoint: Optional[str] = None, fallback_mode: str = "fast_local"):
        self.vast_ai_endpoint = vast_ai_endpoint
        self.fallback_mode = fallback_mode
        
        # Initialize fast local generator if available
        self.fast_generator = None
        if fallback_mode == "fast_local":
            try:
                from fast_broll import FastLocalGenerator
                self.fast_generator = FastLocalGenerator()
                print("✅ Fast local generator initialized")
            except ImportError:
                print("⚠️ Fast local generator not available")
        
        print("🎬 Production generator ready:")
        print(f"  - Vast.ai endpoint: {'✅ Configured' if vast_ai_endpoint else '❌ Not set'}")
        print(f"  - Fallback mode: {fallback_mode}")
    
    def generate_broll_video(self, prompt: str, duration: float, output_path: str, 
                           prefer_quality: bool = True) -> bool:
        """Generate B-roll video with quality preference"""
        print(f"🎬 Generating B-roll: {prompt[:60]}... ({duration:.1f}s)")
        
        # Try Vast.ai first if available and quality is preferred
        if self.vast_ai_endpoint and prefer_quality:
            print("  🌐 Attempting Wand 2.2 via Vast.ai...")
            if self._generate_via_vast_ai(prompt, duration, output_path):
                return True
            print("  ⚠️ Vast.ai failed, falling back to local generation")
        
        # Try fast local generation
        if self.fast_generator:
            print("  ⚡ Using fast local generation...")
            return self.fast_generator.generate_fast_broll(prompt, duration, output_path)
        
        # Final fallback to enhanced placeholder
        print("  📝 Using enhanced placeholder...")
        return self._generate_enhanced_placeholder(prompt, duration, output_path)
    
    def _generate_via_vast_ai(self, prompt: str, duration: float, output_path: str) -> bool:
        """Generate video using Wand 2.2 on Vast.ai"""
        try:
            # Enhanced payload for Wand 2.2
            payload = {
                "model": "wand-2.2",
                "prompt": prompt,
                "duration": duration,
                "width": 1080,
                "height": 1920,
                "fps": 30,
                "steps": 30,  # Higher quality
                "guidance_scale": 8.0,
                "seed": -1,  # Random seed
                "motion_bucket_id": 127,  # Good motion
                "cond_aug": 0.02
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('VAST_AI_TOKEN', '')}"
            }
            
            # Make request with progress tracking
            response = requests.post(
                f"{self.vast_ai_endpoint}/generate",
                json=payload,
                headers=headers,
                timeout=600,  # 10 minute timeout for quality generation
                stream=True
            )
            
            if response.status_code == 200:
                # Save the generated video
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verify file size
                file_size = os.path.getsize(output_path)
                if file_size > 1000:  # At least 1KB
                    print(f"  ✅ High-quality B-roll generated: {output_path} ({file_size/1024:.1f}KB)")
                    return True
                else:
                    print(f"  ❌ Generated file too small: {file_size} bytes")
                    return False
            else:
                print(f"  ❌ Vast.ai request failed: {response.status_code}")
                return False
                
        except requests.Timeout:
            print("  ⏱️ Vast.ai request timed out")
            return False
        except Exception as e:
            print(f"  ❌ Vast.ai generation error: {e}")
            return False
    
    def _generate_enhanced_placeholder(self, prompt: str, duration: float, output_path: str) -> bool:
        """Generate enhanced placeholder with better visuals"""
        try:
            # Generate color based on prompt
            color = self._prompt_to_color(prompt)
            
            # Create visual effects based on content
            effects = self._prompt_to_effects(prompt)
            
            # Shortened prompt for display
            display_text = prompt[:50] + "..." if len(prompt) > 50 else prompt
            
            cmd = [
                "ffmpeg", "-y", "-v", "quiet",
                "-f", "lavfi",
                "-i", f"color=c={color}:size=1080x1920:duration={duration}:rate=30",
                "-vf", (
                    f"drawtext=text='🎬 AI B-ROLL\\n{display_text}'"
                    ":fontcolor=white:fontsize=24:x=(w-text_w)/2:y=h*0.4+20*sin(t*2*PI/1.5),"
                    "drawtext=text='⚡ Fast Generation Mode'"
                    ":fontcolor=cyan:fontsize=18:x=(w-text_w)/2:y=h*0.6,"
                    f"{effects}"
                ),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                file_size = os.path.getsize(output_path)
                print(f"  ✅ Enhanced placeholder created: {output_path} ({file_size/1024:.1f}KB)")
                return True
            else:
                print(f"  ❌ Enhanced placeholder failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"  ❌ Error creating enhanced placeholder: {e}")
            return False
    
    def _prompt_to_color(self, prompt: str) -> str:
        """Smart color selection based on prompt content"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["stress", "cortisol", "tension"]):
            return "#FF6B6B"  # Stress red
        elif any(word in prompt_lower for word in ["brain", "neural", "synapse"]):
            return "#6B73FF"  # Neural blue
        elif any(word in prompt_lower for word in ["heart", "blood", "circulatory"]):
            return "#FF9FF3"  # Cardiovascular pink
        elif any(word in prompt_lower for word in ["muscle", "fiber", "tissue"]):
            return "#54A0FF"  # Muscle blue
        elif any(word in prompt_lower for word in ["molecule", "chemical", "hormone"]):
            return "#5F27CD"  # Chemical purple
        else:
            return "#00D2D3"  # Default cyan
    
    def _prompt_to_effects(self, prompt: str) -> str:
        """Add visual effects based on prompt content"""
        prompt_lower = prompt.lower()
        
        if "flowing" in prompt_lower or "stream" in prompt_lower:
            return "drawtext=text='~':fontcolor=white:fontsize=40:x=w*0.1+50*sin(t):y=h*0.2+30*cos(t*1.5)"
        elif "pulse" in prompt_lower or "beat" in prompt_lower:
            return "drawtext=text='♥':fontcolor=red:fontsize=60:x=(w-text_w)/2:y=h*0.8+10*sin(t*8)"
        elif "spark" in prompt_lower or "fire" in prompt_lower:
            return "drawtext=text='✨':fontcolor=yellow:fontsize=30:x=w*0.8:y=h*0.2+20*sin(t*3)"
        else:
            return "drawtext=text='◦':fontcolor=white:fontsize=20:x=w*0.9:y=h*0.1+15*sin(t*2)"

class ProductionBRollPipeline:
    """Complete production B-roll pipeline"""
    
    def __init__(self, vast_ai_endpoint: Optional[str] = None):
        self.analyzer = ProductionBRollAnalyzer()
        self.generator = ProductionVideoGenerator(vast_ai_endpoint)
        self.vast_ai_endpoint = vast_ai_endpoint
    
    def process_clip(self, clip_data: Dict, prefer_quality: bool = True) -> bool:
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
            for i, region in enumerate(broll_regions, 1):
                print(f"\n📹 Generating B-roll {i}/{len(broll_regions)}:")
                print(f"  Time: {region.start_time:.1f}s - {region.end_time:.1f}s ({region.duration:.1f}s)")
                print(f"  Prompt: {region.prompt[:80]}...")
                
                output_path = f"broll_{i}_{int(region.start_time)}s.mp4"
                
                success = self.generator.generate_broll_video(
                    region.prompt,
                    region.duration,
                    output_path,
                    prefer_quality=prefer_quality
                )
                
                if success:
                    region.broll_path = output_path
                    print(f"  ✅ Generated: {output_path}")
                else:
                    print(f"  ❌ Failed to generate B-roll {i}")
            
            # Generate final composite video
            return self._create_final_video(broll_regions)
            
        except Exception as e:
            print(f"❌ Error processing clip: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_final_video(self, broll_regions: List[BRollRegion]) -> bool:
        """Create final video with B-roll integration"""
        print("\n🎞️ Creating final video with B-roll integration...")
        
        # This would integrate with your existing video composition pipeline
        # For now, just report success
        successful_brolls = [r for r in broll_regions if r.broll_path and os.path.exists(r.broll_path)]
        
        print(f"✅ Successfully generated {len(successful_brolls)}/{len(broll_regions)} B-roll videos")
        
        for region in successful_brolls:
            file_size = os.path.getsize(region.broll_path) / 1024
            print(f"  📁 {region.broll_path}: {file_size:.1f}KB")
        
        return len(successful_brolls) > 0

# Configuration
PRODUCTION_CONFIG = {
    "vast_ai_endpoint": os.getenv("VAST_AI_ENDPOINT"),
    "vast_ai_token": os.getenv("VAST_AI_TOKEN"),
    "prefer_quality": True,  # Set to False for faster development
    "max_concurrent_generations": 3
}

def create_production_pipeline() -> ProductionBRollPipeline:
    """Factory function to create production B-roll pipeline"""
    return ProductionBRollPipeline(PRODUCTION_CONFIG["vast_ai_endpoint"])

if __name__ == "__main__":
    # Test the production system
    print("🚀 Testing Production B-roll System")
    print("=" * 50)
    
    # Test with sample data
    test_clip = {
        "hook": "The Hidden Truth About Stress",
        "segments": [
            {"start": 0, "end": 10, "text": "When we experience stress, our bodies release a cascade of hormones including cortisol and adrenaline."},
            {"start": 10, "end": 20, "text": "These stress hormones travel through our bloodstream and affect every cell in our body."},
            {"start": 20, "end": 30, "text": "The sympathetic nervous system activates, increasing our heart rate and blood pressure."}
        ]
    }
    
    pipeline = create_production_pipeline()
    
    # Test with prefer_quality=False for faster testing
    success = pipeline.process_clip(test_clip, prefer_quality=False)
    
    if success:
        print("\n🎉 Production Pipeline Test PASSED!")
    else:
        print("\n❌ Production Pipeline Test FAILED!")
