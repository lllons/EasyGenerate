import torch
import os
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image


def main():
    print("==================================================")
    print("                  Editor Lab")
    print("==================================================\n")

    # ---------------------------------------------------------
    # 1. LOAD MODEL
    # ---------------------------------------------------------
    print("Loading model into CPU memory... Please wait.")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cpu")
    print("Model successfully loaded!\n")

    # ---------------------------------------------------------
    # 2. USER INPUTS (PROMPTS)
    # ---------------------------------------------------------
    # InstructPix2Pix wants an INSTRUCTION, not a description.
    # Good: "Make it snow"  /  Bad: "a snowy street"
    user_prompt = input("Enter your EDIT INSTRUCTION:\n> ").strip()

    user_neg_prompt = input("\nEnter your NEGATIVE PROMPT (leave blank for default):\n> ").strip()
    if not user_neg_prompt:
        user_neg_prompt = "blurry, low quality, distorted, extra limbs, bad anatomy, text, watermark"

    # ---------------------------------------------------------
    # 3. TINKER PARAMETERS (Edit these numbers to experiment!)
    # ---------------------------------------------------------
    # Source image to edit
    INPUT_IMAGE = "input.png"

    # Image Dimensions (Must be multiples of 8. Default for SD 1.5 is 512x512)
    # NOTE: pix2pix has no width/height argument - the input image is resized
    # to this, and the output comes out at the same size.
    WIDTH = 512
    HEIGHT = 512
    # 8: 64, 128, 256, 512, 768, 1024

    # Generation Quality & Speed
    # Higher steps = cleaner image, but takes longer on CPU. (Range: 15 to 50)
    STEPS = 5

    # Guidance Scale (CFG)
    # Controls how strictly the AI obeys your text prompt vs creative freedom. (Range: 1.0 to 15.0, Default: 7.5)
    GUIDANCE_SCALE = 7.5

    # Image Guidance Scale
    # Controls how closely the result sticks to the ORIGINAL image.
    # Higher = looks more like the input, lower = edits more aggressively. (Range: 1.0 to 2.5, Default: 1.5)
    IMAGE_GUIDANCE_SCALE = 1.5

    # Random Seed (-1 for completely random generation every run, or set an integer like 42 to reproduce exact results)
    SEED = -1

    # Output filename prefix
    OUTPUT_PREFIX = "edit"

    # ---------------------------------------------------------
    # 4. RUN EDIT
    # ---------------------------------------------------------
    # Load and size the source image
    image = Image.open(INPUT_IMAGE).convert("RGB")
    image = image.resize((WIDTH, HEIGHT), Image.LANCZOS)

    # Set up random generator seed
    generator = None
    if SEED != -1:
        generator = torch.Generator("cpu").manual_seed(SEED)
        print(f"\nUsing locked Seed: {SEED}")
    else:
        print("\nUsing Random Seed...")

    print("\n--- Starting Edit ---")
    print(f"Input Image:     '{INPUT_IMAGE}'")
    print(f"Instruction:     '{user_prompt}'")
    print(f"Negative Prompt: '{user_neg_prompt}'")
    print(f"Resolution:      {WIDTH}x{HEIGHT}")
    print(f"Steps:           {STEPS}")
    print(f"Guidance Scale:  {GUIDANCE_SCALE}")
    print(f"Image Guidance:  {IMAGE_GUIDANCE_SCALE}")
    print("Processing on CPU... (This will take a few minutes)")

    # Run the model with all user settings
    result = pipe(
        prompt=user_prompt,
        negative_prompt=user_neg_prompt,
        image=image,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        image_guidance_scale=IMAGE_GUIDANCE_SCALE,
        generator=generator
    ).images[0]

    # ---------------------------------------------------------
    # 5. SAVE RESULT
    # ---------------------------------------------------------
    file_number = 1
    while os.path.exists(f"{OUTPUT_PREFIX}{file_number}.png"):
        file_number += 1
    output_file = f"{OUTPUT_PREFIX}{file_number}.png"

    result.save(output_file)
    print(f"\nDone! Image successfully saved to: {output_file}")


if __name__ == "__main__":
    main()