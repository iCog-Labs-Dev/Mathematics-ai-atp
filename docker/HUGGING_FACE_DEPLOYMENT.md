# Deploy the MCTS API from Hugging Face artifacts

This guide runs the MCTS proving API on a generic Linux Docker host. It uses one image for two sequential operations:

1. A temporary container downloads and verifies a published model release into persistent host storage.
2. A long-running container mounts that release read-only and serves proof-search requests through FastAPI/Uvicorn.

The downloader must finish successfully before you start the API. These steps are explicit; the current Compose file does not start a download service automatically.

## Requirements

- Docker Engine and the Docker Compose plugin on a Linux host, with permission to run Docker commands.
- The application source revision containing `docker/Dockerfile`, `docker-compose.yml`, and `maths_ai/gnn_inference/scripts/fetch_mcts_deployment.py`.
- Network access for the image build and Hugging Face download. The image build downloads and builds the Lean and PLN dependencies, so the initial build can take considerable time and storage.
- Persistent storage for the downloaded bundle, partial-transfer overhead, Docker build layers, and API state. Check your filesystem quota as well as free disk space.
- A trusted published release with a checkpoint matching the image's custom client, REPL, and compiled Mathlib environment. Artifact download does not establish runtime compatibility.
- For GPU serving: an NVIDIA GPU, a compatible host driver, and configured NVIDIA Container Toolkit support for Docker.

Conda, JupyterHub, and host installations of Python, Lean, or PeTTa are not required for this container-based workflow. The image supplies the application and native runtimes.

Run all commands on the Docker host from the application repository root. If you are using a remote Docker context or a Docker daemon outside your shell's container, bind-mount source paths must exist in the **daemon's filesystem**. A path visible in a JupyterLab terminal is not necessarily visible to its Docker daemon.

## Release identifiers and the upload receipt

The repository used in this guide is **`iCogLabs-Dev/Lean-Proof-Search-Checkpoints`**. Pass it explicitly through `--repo` when fetching; the CLI still defaults to the older repository. Its serving releases use this layout:

```text
deployments/mcts/v1.0.0/
├── release.json
├── README.md
├── artifacts/
│   ├── checkpoint.pt
│   ├── prepared/
│   ├── query/
│   ├── index/
│   ├── lemmas.jsonl
│   └── declarations.json
└── config/
    ├── rl.json
    └── service.json
```

Obtain these values from the release publisher:

| Value | Meaning |
| --- | --- |
| Model repository | Hugging Face model repository containing the release |
| Version | Release prefix version, for example `v1.0.0` |
| Full release commit | Exact 40-character Hugging Face commit to download; this is not the application GitHub commit |
| Manifest SHA-256 | Trusted checksum of the published `release.json`, preferably provided through release notes or another trusted channel |
| Application revision | Compatible GitHub code revision used to build the service image |
| Serving device | Whether the release's generated configuration selects `cpu` or `cuda:0` |

**`upload-receipt.json` belongs to the upload workflow.** The upload script saves it locally for the person publishing the release. It contains the resulting commit, manifest digest, and upload-verification information. It is not a model artifact, is not uploaded by default, and is not required on the deployment host. Deployment users only need the published identifiers above; the publisher can share those values without sharing the receipt itself.

If you are also the uploader, you may read those values from your local upload receipt. Otherwise, use the release publisher's instructions and the Hugging Face commit history. Do not expect a deployment user to possess a JupyterHub run directory or upload receipt.

A version branch can move. Pin the full Hugging Face commit for deployment. If a trusted manifest digest has not been supplied, the fetch command can omit `--manifest-sha256`; it then trusts the manifest at the selected commit and still verifies every file against it. That does not provide an independent manifest-identity check.

## 1. Obtain the compatible application source

For a new checkout:

```bash
git clone --branch MCTS-AC-branch \
  https://github.com/iCog-Labs-Dev/Mathematics-ai-atp.git \
  mathematics-ai-atp
cd mathematics-ai-atp
```

Use the application revision recommended by the release publisher. If they provide a specific GitHub commit, check it out before building:

```bash
git checkout REPLACE_WITH_APPLICATION_COMMIT
```

For an existing checkout, enter its root directory and select the compatible revision without discarding local changes. The publishing/download scripts must have been committed and pushed before another host can obtain them through Git.

Check that the required files and Docker are available:

```bash
test -f maths_ai/gnn_inference/scripts/fetch_mcts_deployment.py
test -f docker/Dockerfile
docker version
docker compose version
```

## 2. Build the shared image

```bash
docker build -f docker/Dockerfile -t maths-ai-mcts-api .
```

The Dockerfile's existing `COPY maths_ai/ /workspace/maths_ai/` includes the downloader. No separate downloader image is required. The same image contains the model-loading code, API, custom Pantograph client, Lean runtime, and provisioned PLN runtime.

The build does not download the model release. The commands below perform that operation separately. A successful image build does not establish that its independently built REPL/Mathlib bytes match the release inventory.

## 3. Choose the release and persistent host directory

Replace the commit and checksum placeholders with real values. These exports apply to your current terminal:

```bash
export HF_MODEL_REPO="iCogLabs-Dev/Lean-Proof-Search-Checkpoints"
export HF_RELEASE_VERSION="v1.0.0"
export HF_RELEASE_COMMIT="REPLACE_WITH_FULL_HUGGING_FACE_COMMIT"
export HF_RELEASE_MANIFEST="REPLACE_WITH_RELEASE_JSON_SHA256"

export DEPLOY_PARENT="$HOME/mcts-deployments"
export DEPLOY_ROOT="$DEPLOY_PARENT/$HF_RELEASE_VERSION"
mkdir -p "$DEPLOY_PARENT"
```

`DEPLOY_PARENT` must be an absolute path on persistent host storage that the Docker daemon can mount. You can use another location, such as `/srv/mcts-deployments`, if your user can write to it. Do not use `/tmp` for a deployment you need to retain.

The output `DEPLOY_ROOT` must be absent for the first download. The downloader refuses to overwrite an existing deployment.

## 4. Launch the download container

For a public repository:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --entrypoint /opt/venv/bin/python \
  -v "$DEPLOY_PARENT:/downloads" \
  maths-ai-mcts-api \
  -m maths_ai.gnn_inference.scripts.fetch_mcts_deployment \
  --repo "$HF_MODEL_REPO" \
  --revision "$HF_RELEASE_COMMIT" \
  --manifest-sha256 "$HF_RELEASE_MANIFEST" \
  --version "$HF_RELEASE_VERSION" \
  --output "/downloads/$HF_RELEASE_VERSION"
```

This runs in the foreground and:

- Resolves and downloads the specified commit and release prefix.
- Verifies every serving file's checksum and size.
- Validates checkpoint, prepared metadata, query encoder, index, corpus, and inventory bindings using the serving loaders.
- Publishes the completed deployment through a relative symlink to a persistent sibling backing directory.
- Exits successfully and removes the temporary container through `--rm`.

The bind mount preserves downloaded files on the host after the container exits. Running the downloader as your host user avoids creating root-owned host files.

For a private or restricted repository, provide an externally set `HF_TOKEN` with read access and add `-e HF_TOKEN` before `maths-ai-mcts-api`. The token must be available in the host environment; do not put its literal value into the Dockerfile, image, release configuration, or shared instructions. A write token is not needed for downloading.

If the command fails, stop here and inspect its error. Do not start the API against a partial download. The error identifies a retained backing directory for retry. Rerun the same download command with an additional argument after the output argument:

```text
--resume /downloads/.v1.0.0.bundle-REPLACE_WITH_RETAINED_DIRECTORY
```

Use the actual **container path** printed by the failed download, with the same host mount and original arguments. Completed verified files are reused. A new invocation without `--resume` can allocate another backing directory and consume more space.

After success, the host layout is approximately:

```text
$DEPLOY_PARENT/
├── .v1.0.0.bundle-<unique-id>/
│   ├── artifacts/
│   ├── config/
│   ├── release.json
│   └── download-receipt.json
└── v1.0.0 -> .v1.0.0.bundle-<unique-id>
```

Keep the hidden backing directory. Deleting it breaks the published deployment link. `download-receipt.json` is generated on the deployment host and records verified download identities; it is distinct from the uploader's local `upload-receipt.json`.

## 5. Set the API mounts and file permissions

```bash
export MATHS_AI_ARTIFACTS="$DEPLOY_ROOT/artifacts"
export MATHS_AI_CONFIG="$DEPLOY_ROOT/config"

test -f "$MATHS_AI_ARTIFACTS/checkpoint.pt"
test -f "$MATHS_AI_CONFIG/service.json"
test -f "$MATHS_AI_CONFIG/rl.json"
```

The API image runs as user ID 1000. Its user must be able to traverse the mounted artifact/config directories and read their files. The download container may have used a different host user ID. For these deliberately shared serving artifacts and generated configurations, you can grant read/traversal access explicitly:

```bash
chmod -R a+rX "$MATHS_AI_ARTIFACTS" "$MATHS_AI_CONFIG"
```

On a host requiring restricted local access, arrange equivalent read access for container user ID 1000 through your ownership/group policy instead. Do not grant access to your private upload receipt or token files. Hosts enforcing SELinux may also require bind-mount labeling appropriate to their policy.

Compose mounts these directories read-only at `/artifacts` and `/config`. It creates a separate writable named volume, `maths-ai-mcts-state`, at `/state` for evaluation jobs and runtime state.

## 6. Launch one API container

Choose the service matching the release's generated `config/rl.json` device settings. Check `device`, `collection_devices`, `update_device`, and `retrieval.device` if unsure. Starting the GPU service does not automatically rewrite a CPU configuration, or the reverse.

For a CPU-configured release:

```bash
HF_TOKEN= docker compose up -d mcts-api
```

For a release configured for one GPU at `cuda:0`:

```bash
HF_TOKEN= docker compose --profile gpu up -d mcts-api-gpu
```

These commands use the already built `maths-ai-mcts-api` image. They clear the download token for this invocation because the model files are already local and the default evaluation dataset is public. Supply a separate read token only if your chosen evaluation dataset or another runtime operation genuinely requires private Hub access.

Run **one** service. The CPU and GPU services share the host port and state volume; they are alternatives, not two replicas. Keep `MATHS_AI_ARTIFACTS` and `MATHS_AI_CONFIG` exported for later Compose commands in this terminal. Re-export them when using a new terminal.

## 7. Inspect logs and readiness

For CPU serving:

```bash
docker logs --tail=100 maths-ai-mcts-api
```

For GPU serving:

```bash
docker logs --tail=100 maths-ai-mcts-api-gpu
```

For either service, with the default port:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/ready
```

`/health` reports API liveness. `/ready` remains unavailable while initialization runs and reports failure if model or runtime validation fails. Startup can take minutes. Readiness also verifies the provisioned PLN runtime when the configured profiles require it.

The declaration inventory is bound to the **REPL's source commit and toolchain and the compiled Mathlib contents**. The image keeps the REPL's Git checkout so that its commit can be read; a REPL that is not in a clean checkout of its commit can only match an inventory bound to its exact executable bytes, as releases published before this binding were. If readiness reports an environment mismatch, deploy a compatible runtime or use the verified environment-migration workflow to publish a new matched release. Do not edit inventory hashes or disable validation to make startup pass.

A trusted checkpoint is required: PyTorch checkpoint loading can execute pickle content. Valid checksums establish identity, not that an unknown publisher is safe.

## 8. Submit a proof request

After readiness succeeds, use a profile present in the mounted `service.json`:

```bash
curl --fail http://localhost:8000/prove \
  -H 'Content-Type: application/json' \
  -d '{
    "goal": "True",
    "hypotheses": [],
    "search_profile": "puct",
    "proposal_mode": "sample",
    "seed": 42,
    "timeout_s": 60
  }'
```

The generated configuration also provides `best_first`, `puct_pln`, and `best_first_pln`. Select one of the PLN profiles to exercise PLN-enabled search. A trained policy is not guaranteed to solve the example within its search budget; inspect the response's certification and termination information.

The default Compose port is bound to `127.0.0.1:8000`. Remote users cannot reach it directly. Use an appropriately configured authenticated proxy/TLS setup for shared access; the service itself has no authentication. Set `MATHS_AI_API_PORT` before starting Compose if a different host port is needed, and use that port in requests.

## Stop or update the deployment

Stop the selected service with its corresponding command:

```bash
docker compose stop mcts-api
# Or, for GPU serving:
docker compose --profile gpu stop mcts-api-gpu
```

Stopping/removing a container does not remove the downloaded host artifacts. Preserve the named state volume if you need saved evaluation results. Do not use volume-removal commands casually.

For a new model version, download and verify it into a new persistent output, stop the old service, update the two mount variables, and recreate the selected API service. Do not modify the currently mounted release in place. Follow the existing state/restart policy: unfinished evaluation jobs become interrupted rather than automatically resuming.

## Related instructions and validation limits

- [Publishing model releases and detailed retry instructions](../maths_ai/gnn_inference/docs/mcts_hugging_face_deployment.md)
- [Image contents, API endpoints, and service lifecycle](DOCKER.md)

These instructions match the repository's Dockerfile, Compose configuration, and downloader CLI. The download tooling has offline fake-Hub tests and a tiny real model/tokenizer/FAISS round-trip test. A full image build and deployment with the selected release still need verification on the target Docker host.
