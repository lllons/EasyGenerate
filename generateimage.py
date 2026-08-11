import torch
import os
from diffusers import DiffusionPipeline

def main():
    print("==================================================")
    print("                Generator Lab")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 1. LOAD MODEL
    # ---------------------------------------------------------
    print("Loading model into CPU memory... Please wait.")
    pipe = DiffusionPipeline.from_pretrained(
        "Lykon/dreamshaper-8", 
        torch_dtype=torch.float32,
        safety_checker=None, 
        requires_safety_checker=False, 
    )
    pipe = pipe.to("cpu")
    print("Model successfully loaded!\n")

    # ---------------------------------------------------------
    # 2. USER INPUTS (PROMPTS)
    # ---------------------------------------------------------
    user_prompt = input("Enter your PROMPT:\n> ").strip()

    user_neg_prompt = input("\nEnter your NEGATIVE PROMPT (leave blank for default):\n> ").strip()
    if not user_neg_prompt:
        user_neg_prompt = "blurry, low quality, distorted, extra limbs, bad anatomy, text, watermark"

    # ---------------------------------------------------------
    # 3. TINKER PARAMETERS (Edit these numbers to experiment!)
    # ---------------------------------------------------------
    # Image Dimensions (Must be multiples of 8. Default for SD 1.5 is 512x512)
    WIDTH = 512 
    HEIGHT = 512  
    # 8: 64, 128, 256, 512, 768, 1024
    
    # Generation Quality & Speed
    # Higher steps = cleaner image, but takes longer on CPU. (Range: 15 to 50)
    STEPS = 12    

    # Guidance Scale (CFG)
    # Controls how strictly the AI obeys your text prompt vs creative freedom. (Range: 1.0 to 15.0, Default: 7.5)
    GUIDANCE_SCALE = 7.5  

    # Random Seed (-1 for completely random generation every run, or set an integer like 42 to reproduce exact results)
    SEED = 16     

    # Output filename prefix
    OUTPUT_PREFIX = "phto"

    # ---------------------------------------------------------
    # 4. RUN GENERATION
    # ---------------------------------------------------------
    # Set up random generator seed
    generator = None
    if SEED != -1:
        generator = torch.Generator("cpu").manual_seed(SEED)
        print(f"\nUsing locked Seed: {SEED}")
    else:
        print("\nUsing Random Seed...")

    print("\n--- Starting Generation ---")
    print(f"Prompt:          '{user_prompt}'")
    print(f"Negative Prompt: '{user_neg_prompt}'")
    print(f"Resolution:      {WIDTH}x{HEIGHT}")
    print(f"Steps:           {STEPS}")
    print(f"Guidance Scale:  {GUIDANCE_SCALE}")
    print("Processing on CPU... (This will take a few minutes)")

    # Run the model with all user settings
    image = pipe(
        prompt=user_prompt,
        negative_prompt=user_neg_prompt,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]

    # ---------------------------------------------------------
    # 5. SAVE RESULT
    # ---------------------------------------------------------
    file_number = 1
    while os.path.exists(f"{OUTPUT_PREFIX}{file_number}.png"):
        file_number += 1
    output_file = f"{OUTPUT_PREFIX}{file_number}.png"

    image.save(output_file)
    print(f"\nDone! Image successfully saved to: {output_file}")

if __name__ == "__main__":
    main()