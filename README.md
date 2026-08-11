# EasyGenerate

**Generate AI images with one python file!**

A single-file, CPU-only text-to-image generator. Run it, type a prompt, get a PNG. No GPU, no web UI, no API keys — just Python and patience.

Built on [Hugging Face Diffusers](https://github.com/huggingface/diffusers) using the **[Lykon/dreamshaper-8](https://huggingface.co/Lykon/dreamshaper-8)** model (a Stable Diffusion 1.5 fine-tune).

---

## Requirements

- Python 3.9+
- ~8 GB RAM free (the model is loaded in `float32`)
- ~5 GB disk for the model weights (downloaded once, cached)

## Install

```bash
git clone https://github.com/lllons/EasyGenerate.git
cd EasyGenerate

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install torch diffusers transformers accelerate safetensors pillow
```

> On Linux, plain `pip install torch` pulls the ~2.5 GB CUDA build. For a much smaller CPU-only wheel:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Run

```bash
python generateimage.py
```

The first run downloads the model (a few GB) into your Hugging Face cache — subsequent runs start straight away.

You'll be asked for two things:

1. **Prompt** — what you want to see.
2. **Negative prompt** — what you *don't* want. Press Enter to accept the default (`blurry, low quality, distorted, extra limbs, bad anatomy, text, watermark`).

Output is saved next to the script as `phto1.png`, `phto2.png`, and so on — existing files are never overwritten.

### Example

```
Enter your PROMPT:
> a lighthouse on a rocky coast at dusk, dramatic clouds, cinematic lighting

Enter your NEGATIVE PROMPT (leave blank for default):
>
```

Expect roughly **1–3 minutes** per image on a modern laptop CPU at the default settings.

---

## Tinkering

All the knobs live in section 3 of `generateimage.py`. Edit the constants and re-run.

| Setting | Default | What it does |
|---|---|---|
| `WIDTH` / `HEIGHT` | `512` | Output size. Must be multiples of 8. SD 1.5 is trained at 512×512 — going much higher tends to produce duplicated limbs and heads, and costs a lot more CPU time. |
| `STEPS` | `12` | Denoising steps. More steps = cleaner detail, linearly slower. 12 is fast-and-rough; 20–30 is the usual sweet spot. |
| `GUIDANCE_SCALE` | `7.5` | How strictly the model obeys the prompt. Low (3–5) is loose and creative, high (10–15) is literal but can look overcooked. |
| `SEED` | `16` | Locks randomness so the same prompt gives the same image. Set to `-1` for a different result every run. |
| `OUTPUT_PREFIX` | `"phto"` | Filename prefix for saved images. |

**Reproducibility tip:** keep `SEED` fixed while you iterate on wording, so you can tell whether a change came from your prompt or just from a new roll of the dice. Switch to `-1` once you're happy and want variations.

## Running on a GPU

The script is pinned to CPU. If you have a CUDA card, change these two lines:

```python
torch_dtype=torch.float16   # was torch.float32
...
pipe = pipe.to("cuda")      # was "cpu"
```

...and update the generator device to match:

```python
generator = torch.Generator("cuda").manual_seed(SEED)
```

On Apple Silicon, use `"mps"` instead of `"cuda"` and keep `float32`.

---

## Notes

- **The NSFW safety checker is disabled** (`safety_checker=None`). That's a deliberate choice for local experimentation — worth knowing before you point anyone else at this.
- Model weights are cached in `~/.cache/huggingface/hub` (`%USERPROFILE%\.cache\huggingface\hub` on Windows). Delete that folder to reclaim the space.
- If the process is killed partway through, you're almost certainly out of RAM. Close browser tabs, or drop the resolution.
- Add `*.png` to your `.gitignore` unless you actually want your generations in version control.

## License

This project is released under the [MIT License](LICENSE).

The model weights are separately governed by the [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
