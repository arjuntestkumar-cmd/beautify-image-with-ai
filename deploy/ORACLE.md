# Deploying to an Oracle Cloud Always Free VM

Why this host: the pipeline needs about **960 MB of RAM** with the models loaded (measured, not
estimated), which is why free 512 MB tiers OOM. Oracle's Always Free **Ampere A1** shape gives up
to 4 OCPUs and 24 GB of RAM at no cost and with no time limit — far more headroom than anything
else free, and always-on rather than sleeping.

One thing to know up front: A1 is **arm64**, not x86. The `Dockerfile` detects this and installs
the correct PyTorch wheel automatically, so you do not have to do anything about it.

---

## 1. Create the instance

In the Oracle Cloud console → **Compute → Instances → Create instance**:

| Setting | Value |
| --- | --- |
| Image | Ubuntu 22.04 (or 24.04) |
| Shape | **VM.Standard.A1.Flex** (Ampere, Always Free eligible) |
| OCPUs / memory | 2 OCPU / 12 GB is plenty; 4 / 24 is the free maximum |
| Boot volume | 50 GB default is fine (the image is a few GB) |
| SSH keys | upload your public key, or let it generate one — **save the private key** |

If you get *"Out of host capacity"*, that is Oracle's free ARM pool being full in that region.
Try another availability domain, or retry later — it is a known and common annoyance.

## 2. Open port 80 — **both** firewalls

This is where nearly everyone gets stuck. Oracle Ubuntu images block everything except SSH at the
instance level *as well as* in the cloud firewall. You must open both.

**a) Cloud side** — Instance → *Virtual cloud network* → *Security List* → **Add ingress rule**:

```
Source CIDR : 0.0.0.0/0
IP protocol : TCP
Destination port range : 80
```

**b) Instance side** — SSH in and persist an iptables rule:

```bash
ssh ubuntu@<your-public-ip>

sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo netfilter-persistent save
```

Skip either one and the site simply times out with no useful error.

## 3. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker            # or log out and back in
```

## 4. Deploy

```bash
git clone https://github.com/<your-user>/<your-repo>.git beautify
cd beautify
docker compose up -d --build
```

The first build installs PyTorch and downloads ~528 MB of weights, so give it 10–20 minutes on
2 OCPUs. Subsequent restarts are instant because the weights are baked into the image.

## 5. Verify it is really doing AI

```bash
docker compose logs -f
```

You want to see:

```
beautify.registry registry loaded: {'ready': True, ... 'models': {'realesrgan:general': True,
                                    'realesrgan:wdn': True, 'gfpgan': True}}
```

Then from your own machine:

```bash
curl http://<your-public-ip>/health
```

`"ready"` must be `true` and `"mockMode"` must be `false`. If `ready` is false the app **refuses**
to process anything rather than quietly returning a resized copy — so a broken deployment
announces itself instead of shipping fake results.

Open `http://<your-public-ip>/` and enhance a photo.

---

## Optional: a domain with HTTPS

With a domain pointed at the instance's IP, Caddy gets you an automatic certificate. Open port
443 in **both** firewalls exactly as in step 2, then:

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

Caddy obtains and renews the certificate on its own.

## Keeping it running

`restart: unless-stopped` in `docker-compose.yml` brings the container back after a reboot or a
crash, so there is nothing else to configure.

To deploy new code:

```bash
git pull
docker compose up -d --build
```

## Tuning

Set these under `environment:` in `docker-compose.yml`:

| Variable | Default | Effect |
| --- | --- | --- |
| `AUTO_UPSCALE_MAX_SIDE` | `1600` | Lower it to make big photos faster (they skip the 2× upscale) |
| `RESULT_RETENTION_MINUTES` | `30` | How long an original and result survive |
| `MAX_UPLOAD_BYTES` | `67108864` | Upload ceiling (64 MB) |
| `CHUNK_TILE_SIZE` | `768` | Lower it first if the machine is memory-starved; peak memory scales with it |
| `MAX_QUEUED_JOBS` | `64` | How many people can be waiting before uploads are refused |
| `JOB_TIMEOUT_SECONDS` | `900` | Raise if slow hardware trips it |

Do **not** raise `WORKER_CONCURRENCY` or add uvicorn workers: each one loads its own ~960 MB copy
of the models, and inference already saturates the CPU.

## What to expect

- **Speed.** Ampere cores are decent but this is still CPU inference: expect roughly 15–40 s per
  photo depending on size and whether a face is present.
- **Memory.** ~960 MB resident of your 12–24 GB. Comfortable.
- **Storage.** Uploads and results live in the container and are deleted 30 minutes after each
  job, and on every restart.
