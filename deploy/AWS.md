# Deploying to AWS EC2

## Read this before picking an instance

The pipeline needs about **960 MB of RAM** with the models loaded — measured on a real run, not
estimated. That single number decides everything:

| Instance | RAM | Verdict |
| --- | --- | --- |
| `t2.micro` / `t3.micro` | 1 GB | **Too small.** This is what the classic 12-month free tier covers, and it will OOM or thrash. |
| **`t4g.small`** (ARM Graviton) | **2 GB** | **Use this.** Cheapest thing that actually works, and the `Dockerfile` is arm64-ready. |
| `t3.small` (x86) | 2 GB | Works too, ~25% more expensive for the same memory. |

`t4g.small` runs roughly **$12/month**. AWS accounts opened since mid-2025 start with credits
(commonly $100), which covers months of runtime — check **Billing → Free Tier** in the console to
see exactly what your account has. Either way this is not strictly free, so do step 1.

That 960 MB no longer grows with the size of the photo. The heavy stages run over overlapping
tiles past `CHUNK_THRESHOLD_PIXELS`, so peak memory follows the tile and not the upload: on a
24 MP photo the classical stages peak at **315 MB instead of 3.4 GB**, measured. A big file on a
small instance is now a question of how long it takes, not whether the box survives it. See
*Large files, and more than one person at a time* in the README for how that works.

Two operational consequences worth knowing before you size the disk:

* **Uploads are bigger than they were** — 64 MB and 80 MP, up from 20 MB and 40 MP, because
  there is no longer a memory reason to refuse them. A photo over the pixel budget is fitted to
  it and processed rather than rejected.
* **Each job keeps three files, not two** — the original, the result, and an un-styled copy that
  makes changing or removing the beautification filter free. All three are deleted
  `RESULT_RETENTION_MINUTES` after the job finishes. Budget disk for roughly
  `3 × largest expected result × jobs completed per 30 minutes`; the 8 GB default volume is
  ample for a handful of users, and `RESULT_RETENTION_MINUTES` is the dial if it is not.

---

## 1. Set a billing alarm — do this first

A new AWS account with no guardrail is how people end up with a surprise bill.

**Billing and Cost Management → Budgets → Create budget**

- Template: **Zero spend budget** (alerts on the first cent beyond free/credits), or a monthly
  cost budget of e.g. $15
- Enter your email

Do it now, before launching anything.

## 2. Launch the instance

**EC2 → Instances → Launch instances**

| Field | Value |
| --- | --- |
| Name | `beautify` |
| AMI | **Ubuntu Server 24.04 LTS**, architecture **64-bit (Arm)** |
| Instance type | **t4g.small** |
| Key pair | *Create new* → RSA → `.pem` → **it downloads once; keep it** |
| Network → Allow HTTPS | ✅ |
| Network → Allow HTTP | ✅ |
| Storage | **20 GiB** gp3 (the image plus weights needs more than the 8 GiB default) |

Pick the region nearest you first (top-right selector) — `ap-south-1` Mumbai for India,
`eu-west-1` Ireland for Europe. Latency and price both depend on it.

Launch, then copy the **Public IPv4 address**.

> Unlike Oracle, there is no second firewall inside the VM. The security group is the only one,
> and ticking "Allow HTTP" above is all that is needed.

## 3. Connect

```bash
chmod 400 beautify.pem                       # macOS/Linux
ssh -i beautify.pem ubuntu@<PUBLIC-IP>
```

On Windows PowerShell:

```powershell
icacls .\beautify.pem /inheritance:r /grant:r "$($env:USERNAME):R"
ssh -i .\beautify.pem ubuntu@<PUBLIC-IP>
```

## 4. Install Docker

**Which OS did you actually launch?** The EC2 wizard defaults to *Amazon Linux*, so it is easy to
end up there even when aiming for Ubuntu. `cat /etc/os-release` settles it. The commands differ,
and Docker's convenience script fails on Amazon Linux with
`ERROR: Unsupported distribution 'amzn'`.

### Amazon Linux 2023 (login user: `ec2-user`)

```bash
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker

# The compose plugin is not part of Amazon Linux's docker package.
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)"   -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```

### Ubuntu (login user: `ubuntu`)

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
```

Either way, confirm both are present before continuing:

```bash
docker --version && docker compose version
```

## 5. Deploy

```bash
git clone https://github.com/<your-user>/<your-repo>.git beautify
cd beautify
docker compose up -d --build
```

The first build installs PyTorch and downloads ~528 MB of weights. On a `t4g.small` expect
**15–25 minutes**. Watch it:

```bash
docker compose logs -f
```

Wait for:

```
All weights ready (527.6 MB total, 5 downloaded).
beautify.registry registry loaded: {'ready': True, ... 'gfpgan': True}
```

### If the build is killed

2 GB is comfortable at runtime but tight while pip unpacks PyTorch. If the build dies with
`Killed`, add swap once and rebuild:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 6. Verify

```bash
curl http://<PUBLIC-IP>/health
```

`"ready"` must be `true` and `"mockMode"` must be `false`. If `ready` is false the app **refuses**
to process anything rather than silently returning a resized copy — a broken deploy announces
itself instead of shipping fake results.

Then open `http://<PUBLIC-IP>/` and enhance a photo. Expect **15–40 s** per image: this is CPU
inference on two cores.

---

## Custom domain with HTTPS

1. Point an `A` record at the instance's public IP. (Give the instance an **Elastic IP** first —
   EC2 → Elastic IPs → Allocate → Associate — or the address changes on every stop/start.)
2. Open port 443 in the security group if it is not already.
3. On the instance:

```bash
sudo apt install -y caddy
```

`/etc/caddy/Caddyfile`:

```
your-domain.com {
    reverse_proxy 127.0.0.1:80
}
```

```bash
sudo systemctl restart caddy
```

Caddy obtains and renews the certificate automatically.

## Pulling new code onto a running instance

The login user depends on the AMI: `ec2-user` on Amazon Linux, `ubuntu` on Ubuntu. Or use
**EC2 → Connect → EC2 Instance Connect** for a browser terminal and skip the key entirely.

```bash
ssh -i beautify.pem ec2-user@<PUBLIC-IP>    # ubuntu@ on an Ubuntu AMI
cd ~/beautify                               # `ls ~` if you are not sure where it went

curl -s localhost/health | grep -o '"build":"[^"]*"'   # note this, before

git pull
docker compose up -d --build                           # REBUILD - see below
docker compose logs -f                                 # ctrl-C once it says ready

curl -s localhost/health | grep -o '"build":"[^"]*"'   # must have CHANGED
```

**`--build` is not optional.** The `Dockerfile` does `COPY . .`, so the application code is baked
into the image when it is built. `git pull` updates the files on the host and changes nothing the
container can see; `docker compose restart` restarts the *old* image. Both look like a successful
deploy and neither is one. Only `up -d --build` rebuilds the image and replaces the container.

That is the failure this project's `build` field exists to catch. It is a fingerprint of the
Python that the running process actually loaded, so:

| `build` after the deploy | Meaning |
| --- | --- |
| **different** from before | The new code is live. |
| **same** as before | The container was never replaced. You skipped `--build`, or the build failed and Compose kept the old image running — read `docker compose logs`. |
| field **missing** | You are on a build from before this field existed, so certainly not the latest. |

A second, content-level check that needs no note-taking — the current build serves thirteen
looks:

```bash
curl -s localhost/api/filters | grep -o '"id"' | wc -l    # 13
curl -s localhost/api/filters | grep -o 'Studio Noir'
```

Nothing escapes the rebuild here. `web/` is served from disk on every request, but in Docker
that disk is *inside the image*, so HTML, CSS and JS need the rebuild exactly like the Python
does.

> Running from source instead of Docker (`run.ps1`, or `uvicorn` directly) has the same trap in a
> sharper form: `web/` really is live and a browser refresh picks it up, while `app/*.py` is held
> in the running process and does not. You get a new-looking page driven by the old pipeline,
> which is the single most confusing state this project can be in. **Restart the process after
> touching anything under `app/`** - and `curl localhost:8000/health` will tell you whether you
> did, via `build`.

### Nothing came back up

```bash
docker compose ps                 # is it running, restarting, or gone?
docker compose logs --tail=80     # the actual reason
```

A build killed for memory during `pip install torch` is the common one on a 2 GB box — add swap
(see *If the build is killed* above) and rebuild. To go back to what was working:

```bash
git log --oneline -5
git checkout <last-good-sha>
docker compose up -d --build
```

The images and results in `.data/` are scratch and are wiped on restart anyway, so a rollback
loses nothing but the jobs in flight.

---

## Changing configuration (`.env`)

Every setting has a working default, so the app runs with no `.env` at all. Add one when you want
to override something.

```bash
cd ~/beautify
cp .env.example .env      # first time only - it is gitignored, so `git pull` never touches it
nano .env

docker compose up -d      # no --build needed: env is read at container start, not at build
```

Confirm it took:

```bash
docker compose exec beautify env | grep AUTO_UPSCALE_MAX_SIDE
```

Two things about this file are worth knowing, because both have bitten people:

* **`.env` is excluded from the image** (`.dockerignore`), which is deliberate — per-host tuning
  and anything secret should not be baked into a built artefact. It reaches the container through
  the `env_file:` entry in `docker-compose.yml`, and *only* through it. Before that entry existed,
  a `.env` sitting next to the code on a server did nothing whatsoever.

  It needs **Compose v2.24+**, because it uses the long `path:` / `required:` form. Check first -
  an older plugin fails to parse the file rather than ignoring the line, which turns a routine
  deploy into an outage:

  ```bash
  docker compose version                      # v2.24.0 or newer

  # too old? replace the plugin (Amazon Linux path shown; Ubuntu uses the same command)
  sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
  ```
* **Compose reads a file of the same name for its own `${VAR}` substitution** inside
  `docker-compose.yml`. Same filename, same directory, unrelated job. Do not be surprised that one
  file feeds two mechanisms.

Values set in the `environment:` block of `docker-compose.yml` **override** `.env`. That
precedence is deliberate: a stray `PORT` in a hand-edited `.env` cannot detach the app from the
port the compose file publishes.

`git pull` will never overwrite your `.env` — it is gitignored. It *can* update `.env.example`, so
after a pull that adds a setting, diff them:

```bash
diff <(grep -o '^[A-Z_]*' .env.example | sort -u) <(grep -o '^[A-Z_]*' .env | sort -u)
```

The settings most worth changing on a small instance are in **Tuning** below.

## Day-to-day

```bash
docker compose logs -f       # watch
docker compose ps            # status and health
docker compose restart       # restart the SAME image (config already in env; no code change)
docker compose down          # stop
docker system prune -f       # reclaim disk from old images after a few deploys
```

`restart: unless-stopped` brings the container back after a reboot.

**To stop paying**, terminate the instance (EC2 → Instance state → Terminate) and release any
Elastic IP — an unattached Elastic IP is billed on its own.

## Tuning

| Variable | Default | Effect |
| --- | --- | --- |
| `AUTO_UPSCALE_MAX_SIDE` | `1600` | Lower it to make big photos faster |
| `RESULT_RETENTION_MINUTES` | `30` | How long a result survives — and how much disk is in use |
| `MAX_UPLOAD_BYTES` | `67108864` | Upload ceiling (64 MB) |
| `CHUNK_THRESHOLD_PIXELS` | `4000000` | Past this, the heavy stages run in tiles |
| `CHUNK_TILE_SIZE` | `768` | **Lower this first if the box is memory-starved.** Peak memory scales with it |
| `MODEL_CHUNK_TILE_SIZE` | `384` | Same, for the super-resolution model. 256 is safe on 1 GB |
| `MAX_QUEUED_JOBS` | `64` | How many people can be waiting before uploads are refused |
| `JOB_TIMEOUT_SECONDS_PER_MEGAPIXEL` | `90` | Raise it if a slow instance times out on big photos |
| `KEEP_UNFILTERED_BASE` | `true` | Set `false` to trade instant filter switching for disk |

Do **not** add uvicorn workers: each loads its own ~960 MB copy of the models. Do not raise
`WORKER_CONCURRENCY` either — one heavy job at a time is what keeps the box alive under load,
and the queue handles the rest. If jobs are queueing more than you would like, a bigger instance
is the answer, not more workers.

### If a large photo is slow

That is the design. The priority is **stability → successful processing → output quality →
speed**: a photo that takes two minutes and finishes beats one that takes thirty seconds and
takes the server with it. The progress bar reports real stage boundaries and creeps across each
tiled stage, and anyone waiting behind someone else is told their position in the queue.

If you want it genuinely faster, in order of effect: a GPU instance (an order of magnitude, and
nothing else needs to change), then a lower `AUTO_UPSCALE_MAX_SIDE`, then more vCPUs.
