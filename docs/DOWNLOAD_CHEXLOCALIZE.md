# Downloading CheXlocalize from Azure

The dataset link is a **container-level SAS (Shared Access Signature)** URL. It cannot be opened in a browser; use one of the methods below. The link is valid until **2026-04-14**.

---

## Option 1: AzCopy (recommended, command line)

### Install AzCopy on macOS

```bash
# Homebrew
brew install azcopy
```

If not using Homebrew, download the macOS binary from:  
https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy

### Download the entire container

Replace `YOUR_LOCAL_FOLDER` with the directory where you want the data (e.g. `./data/chexlocalize`).

```bash
azcopy copy 'https://aimistanforddatasets01.blob.core.windows.net/chexlocalize?sv=2019-02-02&sr=c&sig=CQLBxruUoZFW2fE3fqQzTkSnj%2FQ26R8TqsVmvFbE9A4%3D&st=2026-03-15T05%3A52%3A47Z&se=2026-04-14T05%3A57%3A47Z&sp=rl' 'YOUR_LOCAL_FOLDER' --recursive
```

Example (download into `./data/chexlocalize` in your project):

```bash
mkdir -p data/chexlocalize
azcopy copy 'https://aimistanforddatasets01.blob.core.windows.net/chexlocalize?sv=2019-02-02&sr=c&sig=CQLBxruUoZFW2fE3fqQzTkSnj%2FQ26R8TqsVmvFbE9A4%3D&st=2026-03-15T05%3A52%3A47Z&se=2026-04-14T05%3A57%3A47Z&sp=rl' './data/chexlocalize' --recursive
```

- `--recursive` copies all blobs in the container (full dataset).
- Keep the URL in **single quotes** so the shell does not alter `&`, `?`, etc.

---

## Option 2: Azure Storage Explorer (GUI)

1. **Install**  
   https://azure.microsoft.com/en-us/products/storage/storage-explorer/

2. **Connect with SAS**
   - Open Storage Explorer.
   - Right-click **Storage Accounts** (or use the plug icon) → **Connect to Azure Storage**.
   - Choose **Blob container or directory**.
   - Paste the full URL:
     ```
     https://aimistanforddatasets01.blob.core.windows.net/chexlocalize?sv=2019-02-02&sr=c&sig=CQLBxruUoZFW2fE3fqQzTkSnj%2FQ26R8TqsVmvFbE9A4%3D&st=2026-03-15T05%3A52%3A47Z&se=2026-04-14T05%3A57%3A47Z&sp=rl
     ```
   - Click **Next** → **Connect**.

3. **Download**
   - Open the `chexlocalize` container.
   - Select the blobs/folders you need (or select all).
   - Right-click → **Download** and choose your local folder (e.g. `data/chexlocalize`).

---

## After download

- Run `tree -L 2` (or similar) on your CheXlocalize folder and share the structure if you need help with dataset loaders or evaluation paths.
- Use the CheXlocalize repo scripts (`heatmap_to_segmentation.py`, `eval.py`) on the downloaded data as in your plan.
