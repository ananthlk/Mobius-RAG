# LLM configs

YAML files in this directory select which LLM backend and model to use for extraction and critique. The app supports multiple LLM modules via a **provider registry**; you can add new backends without changing core code.

**Credentials:** Do not store API keys or secrets in PostgreSQL or in YAML. Use environment variables or a secret manager. See [docs/CREDENTIALS.md](../../docs/CREDENTIALS.md).

## Using a config

- **Env:** Set `LLM_PROVIDER` and provider-specific env vars (see [config.py](../config.py)). The app uses these when no config version is specified.
- **Config version:** When starting a chunking job, you can pass a **config version** (e.g. `default`, `production`). The app loads `app/llm_configs/<version>.yaml` and builds the provider from it. This overrides env-based selection for that job.

## Config file shape

Each YAML file must have:

- **provider** (required): Registered provider name, e.g. `ollama`, `vertex`, `openai`.
- **model** (optional): Model name (provider-specific).
- **options** (optional): Common options (e.g. `temperature`, `num_predict`).
- **&lt;provider&gt;** (optional): Provider-specific block, e.g. `ollama.base_url`, `vertex.project_id`, `openai.api_key`.

Examples: see `default.yaml` (Ollama) and `production.yaml` (Vertex).

## Built-in providers

| Provider | Config block | Notes |
|----------|--------------|--------|
| **ollama** | `ollama.base_url` | Local; options: `num_predict` |
| **vertex** | `vertex.project_id`, `vertex.location` | Google Vertex AI (Gemini); install with `pip install -e ".[vertex]"`; needs env or config |
| **openai** | `openai.api_key`, `openai.base_url` | Optional; requires `pip install openai` and API key |

## Adding a new LLM module

1. **Implement the provider**  
   In `app/services/`, add a module (e.g. `llm_provider_anthropic.py`) that defines a class subclassing `LLMProvider` and implementing:
   - `async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]`
   - `async def generate(self, prompt: str, **kwargs) -> str`

2. **Register it**  
   In `app/services/llm_provider.py`:
   - Add a factory function that takes a config dict and returns an instance of your provider.
   - Call `register_provider("anthropic", _anthropic_factory)` (and optionally wrap in `try/except ImportError` if the dependency is optional).

3. **Add a config file**  
   Add e.g. `app/llm_configs/anthropic.yaml` with `provider: anthropic` and any keys your factory expects (e.g. `anthropic.api_key`, `model`).

4. **Optional: env fallbacks**  
   In `app/config.py` you can add env vars (e.g. `ANTHROPIC_API_KEY`) and have your factory read them when the key is not in the config.

No changes are required in `llm_config.py`; it uses the registry to resolve `provider` by name.
