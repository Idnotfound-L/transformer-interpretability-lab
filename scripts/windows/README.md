# Windows helper scripts

Convenience wrappers for Windows users. Each script changes into the repository root first, so it can be run from anywhere (double-click or from a terminal).

| Script | Purpose |
|---|---|
| `setup_windows.bat` | Create `.venv` and install the base requirements |
| `setup_huggingface_windows.bat` | Install Hugging Face dependencies for pretrained-model experiments |
| `run_protocol_v2_gpt2_windows.bat [seed]` | Run Protocol V2 on GPT-2 (default seed 7) |
| `run_protocol_v2_dashboard_windows.bat` | Launch the Protocol V2 results dashboard |
| `run_sequence_copy_smollm3_windows.bat`, `run_sequence_copy_study_windows.bat` | Legacy Protocol V1 sequence-copy runners |
| `run_hf_research_windows.bat`, `run_hf_dashboard_windows.bat` | Hugging Face research runner and dashboard |
| `run_research_windows.bat`, `run_research_dashboard_windows.bat` | GPT-2 / TransformerLens research runner and dashboard |
| `train_windows.bat`, `run_dashboard_windows.bat` | Train and inspect the small TinyGPT model |

On Linux/macOS, use the equivalent `python ...` commands shown in the top-level README.
