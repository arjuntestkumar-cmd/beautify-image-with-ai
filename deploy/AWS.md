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

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
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

## Day-to-day

```bash
cd ~/beautify && git pull && docker compose up -d --build   # deploy new code
docker compose logs -f                                       # watch
docker compose down                                          # stop
```

`restart: unless-stopped` brings the container back after a reboot.

**To stop paying**, terminate the instance (EC2 → Instance state → Terminate) and release any
Elastic IP — an unattached Elastic IP is billed on its own.

## Tuning

| Variable | Default | Effect |
| --- | --- | --- |
| `AUTO_UPSCALE_MAX_SIDE` | `1600` | Lower it to make big photos faster |
| `RESULT_RETENTION_MINUTES` | `30` | How long a result survives |
| `MAX_UPLOAD_BYTES` | `20971520` | Upload ceiling |

Do **not** add uvicorn workers: each loads its own ~960 MB copy of the models.
