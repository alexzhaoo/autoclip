import json
import glob

def concat_gpt_responses_to_json(output_file="all_gpt_clips.json"):
    all_clips = []
    # Match files like gpt_response1.txt, gpt_response2.txt, etc.
    for fname in sorted(glob.glob("gpt_response*.txt")):
        with open(fname, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # If the file contains a dict with 'clips', extract it
                if isinstance(data, dict) and "clips" in data:
                    clips = data["clips"]
                elif isinstance(data, list):
                    clips = data
                else:
                    print(f"⚠️ Unexpected format in {fname}")
                    continue
                all_clips.extend(clips)
            except Exception as e:
                print(f"❌ Failed to parse {fname}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_clips, f, indent=2, ensure_ascii=False)
    print(f"✅ Concatenated {len(all_clips)} clips into {output_file}")

# Example usage:
concat_gpt_responses_to_json()