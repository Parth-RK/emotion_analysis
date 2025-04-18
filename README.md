
# Emotion Analysis Engine 🔥

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Stop guessing how your users feel!** This repo turbo-charges pre-trained Hugging Face Transformers, fine-tuning them to sniff out emotions in text like a digital bloodhound. 🐶

## ✨ What's Inside? (The TL;DR)

*   **🚀 Fine-Tune the Big Guns:** Grab *any* compatible Transformer from Hugging Face (MiniLM, BERT, whatever floats your boat) and bend it to your emotional will.
*   **📊 Data Agnostic-ish:** Chuck in CSV, TSV, or JSONL. It'll figure it out.
*   **✂️ Auto-Split Data:** Too lazy for separate `val`/`test` files? Set paths to `None` in the config, it'll slice 'em up for you.
*   **⚙️ Tweak Everything:** `config.py` is your control panel. Dial in hyperparameters, model names, paths – go wild.
*   **📈 Visualize Training:** See how your model learns (or doesn't) with loss/metric plots. No black boxes here.
*   **🔬 Deep Dive Evals:** Get the gritty details with classification reports and confusion matrices. See where it shines, see where it fails.
*   **📦 Organized Artifacts:** No messy outputs. Everything (model, plots, logs) lands neatly in `artifacts/Transformer/`.
*   **💬 Test Drive:** Run `app.py` for instant gratification. See built-in demos or chat interactively with your new AI therapist (disclaimer: not actual therapy).


## 🎬 `app.py` in Action

Fire up `python app.py` post-training and you'll see something like this before the interactive prompt:


- Running Built-in Examples


Example 1/10: 'I am feeling incredibly happy and excited about the party tonight!'
  --> Predicted: joy (Score: 0.9876)  # Nice!

Example 2/10: 'This movie is making me feel really sad and thoughtful.'
  --> Predicted: sadness (Score: 0.9543) # Makes sense.

... (more examples) ...

 Built-in Examples Finished 
 
--- Interactive Emotion Prediction ---
Using model: nreimers/MiniLM-L6-H384-uncased
Enter text to classify, or type 'quit' or 'exit' to stop.

Enter text: _ # Your turn!

## 🛠️ Get It Running (Setup)

Standard procedure, mostly.

1.  **Clone It:**
    ```bash
    git clone https://github.com/Parth-RK/emotion_analysis.git
    cd emotion_analysis
    ```

2.  **Environment:** (Use Conda/Venv, don't pollute global Python!)
    ```bash
    # Example using Conda
    conda create -n emotion-reactor python=3.10
    conda activate emotion-reactor
    ```

3.  **Dependencies:**

    *   ⚠️ **Crucial Step:** Get PyTorch sorted FIRST, matching your CUDA setup (or CPU).
        ```bash
        # EXAMPLE FOR CUDA 11.8 - CHANGE THIS FOR YOUR SYSTEM!
        pip install torch --index-url https://download.pytorch.org/whl/cu118
        ```
        *(Hit up the [PyTorch site](https://pytorch.org/get-started/locally/) for the command you actually need.)*

    *   Install the rest:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Data:**
    *   Drop `training.csv` (or whatever you name it) in the project root, or update `config.py`.
    *   Add `validation.csv` / `test.csv` or let the config handle splitting (`VALID_FILE_PATH = None`).
    *   Make sure the columns line up with `TEXT_COLUMN_INDEX` / `LABEL_COLUMN_INDEX` in `config.py`.

## ⚙️ Config (`config.py`)

This is where the magic happens. Open `config.py` and fiddle with:

*   `TRANSFORMER_MODEL_NAME`: The core choice. `"nreimers/MiniLM-L6-H384-uncased"`? `"google-bert/bert-base-uncased"`? Your call.
*   `TRAIN/VALID/TEST_FILE_PATH`: Point to your data.
*   `EPOCHS`, `TRAIN_BATCH_SIZE`, `LEARNING_RATE`, `MAX_LEN`: The usual suspects for training.
*   *...and a bunch more settings*

## 🚀 Let's Go! (Usage)

1.  **Train the Beast:**
    *   Double-check `config.py`.
    *   Run it:
        ```bash
        python main.py
        ```
    *   *(Pro Tip: `python main.py --help` shows command-line overrides for quick tweaks).*
    *   Watch the logs. Artifacts land in `artifacts/Transformer/`.

2.  **Talk to Your AI:**
    *   Make sure training finished and `artifacts/Transformer/model/best_model.pt` exists.
    *   Run the app:
        ```bash
        python app.py
        ```
    *   See the demos, then type your own text.
    *   `quit` or `exit` when you're done probing its digital soul.



## 💻 My Specs (No need to laugh, it works)

- Nvidia GeForce MX150 2GB
- Nvidia Graphics Driver 572.83
- Cuda toolkit 11.7
- Python 3.10.0
- PyTorch version: 2.6.0+cu118
```bash 
pip install torch --index-url https://download.pytorch.org/whl/cu118
```








## 🤝 Contribute?

Sure, why not? Fork it, break it, fix it, submit a PR. Issues welcome too.

## 📄 License

MIT. Do what you want, just don't blame me if your AI becomes sentient and judges your music taste.

## 🙏 Shoutouts

*  Hugging Face 🤗 - Absolute legends.
*  PyTorch Crew 🔥
*  Google Research 🌐
*  Kaggle 📊