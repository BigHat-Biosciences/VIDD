# Building a BigHat container on a util box

Written while building `vidd-antibody` on 2026-07-27. Nothing in `bonobo`,
`ProDifEvo-Refinement`, `bh-ai` or `bh-experimental` documents the *host* side of
a container build — only the per-repo `bh-deploy.sh` and bonobo's "Deploying to
ECR" note, both of which assume a working Docker host. Everything below was
derived by hand on a fresh `bh compute create` box and cost about an hour.
Most of it is not VIDD-specific.

## The short version

```bash
bh aws-login --admin                  # SSO, 12h token
bh compute create                     # or reuse a running box
# get on (see "Access" -- there are two paths and they are NOT interchangeable)
bash /tmp/setup_box2.sh               # host prep: instance store + docker + nvidia toolkit
rsync the repo up                     # NOT git pull; see "Getting the source up"
PUSH=0 bash scripts/bh-deploy.sh      # build without pushing
bash scripts/smoke_in_container.sh    # single-GPU smoke
```

## The AMI does not have Docker

`bh compute create` hardcodes `ami-0fbaa4ab156b08a7d`
(`bh-ai/aws_tools/src/bh/aws_tools/cli_ext.py:186`), built from
`BhOnDemandImageRecipe` (`bh-ai/stacks/cdk/stacks/image_builder.py:119`). That
recipe installs **pyenv and Python 3.9.17 and nothing else**. There is no Docker,
no `nvidia-container-toolkit`, and no `docker` group. Every person building a
container on a fresh box installs these by hand. Budget for it.

## The root volume is 20G and it is already full

`bh compute create` sets **no `BlockDeviceMappings`**, so you inherit the Ubuntu
parent image's default root volume: 20 GiB. The AMI itself uses ~18G of it (12G
`/usr`, 5.1G `/opt`), leaving roughly 500 MB free on a box that has never run
anything. `scp` of a 3 KB script can fail on a fresh box.

**You probably cannot grow it.** `BigHatReader-automation` — which is what
`bh aws-login --admin` actually gives you — has no `ec2:ModifyVolume`:

```
UnauthorizedOperation ... not authorized to perform: ec2:ModifyVolume
```

Enlarging the EBS volume is an eng ask. Plan to fit in what you have.

**Do not try to build in that space.** Measured, from ECR: `rerd-antibody` is
13.99 GB compressed and `bonobo` is 17.4 GB compressed. Uncompressed, plus build
intermediates, these need well north of 30 GB.

### Use the instance store — this is the whole trick

`g5.4xlarge` (and the other `g5`/`g6e` types in the `bh compute` menu) ship a
large NVMe instance store, unformatted and unmounted. On a `g5.4xlarge` that is
558.8 GB at `/dev/nvme1n1`. Format it, mount it, and put Docker's `data-root`
there **before installing Docker**, so `dockerd` never populates
`/var/lib/docker` on the full root volume:

```bash
sudo mkfs.ext4 -F -m 0 -L dockerstore /dev/nvme1n1
sudo mkdir -p /mnt/store && sudo mount /dev/nvme1n1 /mnt/store
printf '{\n  "data-root": "/mnt/store/lib-docker"\n}\n' | sudo tee /etc/docker/daemon.json
```

Also bind-mount apt's lists and cache onto it. They were consuming 234 MB of the
~500 MB of root we had, and `apt-get update` grows them:

```bash
sudo mount --bind /mnt/store/apt-lists /var/lib/apt/lists
sudo mount --bind /mnt/store/apt-cache /var/cache/apt
```

Formatting and mounting need no packages, which is why this ordering works on a
disk that is already full.

**The instance store is ephemeral.** Its contents do not survive a stop/start,
and `bh compute create` tags every box `autoshutdown=true`
(`cli_ext.py:204`) with no documented shutdown window. An image built here is
disposable until it is pushed to ECR. Do not build at the end of the day and
expect it to be there in the morning.

## Docker 29 ignores your data-root for image layers

This is the one that would have killed the build 30 GB in. Docker 29 defaults to
the **containerd snapshotter** (`docker info` shows
`Storage Driver: overlayfs` / `driver-type: io.containerd.snapshotter.v1`).
Under that driver, `data-root` in `daemon.json` is still reported faithfully as
`Docker Root Dir`, but actual image layers are written to
**`/var/lib/containerd`**, which `data-root` does not cover. A single CUDA base
image put 384 MB there while `Docker Root Dir` stayed at 232 KB.

Relocate containerd too, not just Docker:

```bash
sudo systemctl stop docker.service docker.socket containerd.service
sudo rsync -aH --delete /var/lib/containerd/ /mnt/store/containerd/
sudo rm -rf /var/lib/containerd && sudo mkdir -p /var/lib/containerd
sudo mount --bind /mnt/store/containerd /var/lib/containerd
sudo systemctl start containerd.service docker.service
```

Verify with a real pull rather than trusting `docker info` — measure both
filesystems before and after. A correct setup puts ~all of the image on the
instance store and single-digit KB on root:

```
root-before=482104K store-before=599960K
root-after =482092K store-after =5682876K     # 5.08G to the store, 12K to root
```

Both bind mounts (this and apt's) are lost on reboot and are not in `/etc/fstab`.
That is deliberate — the instance store is ephemeral anyway — but it means a
rebooted box needs the setup script re-run. It is idempotent.

## Two gotchas that will waste your time

**`blkid` lies to non-root.** As `ubuntu` it exits 0 with no output on an
unformatted device; as root it exits 2. A guard like
`if ! blkid /dev/nvme1n1; then mkfs; fi` therefore skips the `mkfs` and the mount
fails with "wrong fs type, bad option, bad superblock". Always `sudo blkid`.

**`pgrep -f <pattern>` matches your own command line.** If you drive the box over
SSH, the remote `bash -c` running your check contains the pattern you are
searching for, so `pgrep -f build.sh` reports the build as alive forever — and
`pkill -f build.sh` kills your own SSH session (exit 255). Use `pgrep -x -f
"bash /full/path.sh"` for an exact full-cmdline match, or kill by explicit PID. I
hit this three separate times in one night; it silently turns a monitor loop into
an infinite one and makes a finished job look hung.

**`apt.systemd.daily` will fight you.** Ubuntu's periodic timer takes
`/var/lib/apt/lists/lock` in the background, and under `set -e` any `apt-get`
call — including `apt-get clean` — dies instantly with "Could not get lock".
Clearing `/var/lib/apt/lists/*` to reclaim space is itself enough to trigger a
refresh that then blocks you for 10+ minutes. Wait the lock out rather than
killing it mid-write; killing `apt-get` while it is writing lists risks leaving
apt in a broken state. See `wait_for_apt()` in the setup script.

## Access: two paths, and they are not interchangeable

1. **Shared long-lived util box** — `bh aws-utility` / `bh aws-gpu-utility`
   print creds and a private IP; the PEM comes from SSM (key names
   `ad-hoc-compute` / `ad-hoc-gpu-compute`). SSH as `ssm-user`. Requires VPN.
2. **Your own `bh compute create` box** — has **no key pair at all**
   (`KeyName: null`) and is **not SSM-registered**, so neither a PEM nor
   `aws ssm start-session` works. Get in with EC2 Instance Connect, as `ubuntu`:

```bash
aws ec2-instance-connect send-ssh-public-key --region us-east-1 \
    --instance-id <id> --instance-os-user ubuntu \
    --availability-zone us-east-1a \
    --ssh-public-key file://$HOME/.ssh/id_ed25519.pub
ssh -i ~/.ssh/id_ed25519 ubuntu@<private-ip>      # key is valid ~60s
```

That grant exists only because the box carries `bh:access-tier=utility`, which
`bh compute create` sets at provision time. Reader is *denied* writing that tag
key, so a box that lacks it cannot be made SSH-able without an admin
(`bh-ai/stacks/cdk/stacks/reader_policy.py:244`,
`bh-ai/workarounds/backfill_access_tier_tag.py`).

Still requires VPN — Instance Connect pushes the key via the API, but the SSH
itself goes to the private IP.

## Getting the source up: rsync, not git

Container work is frequently uncommitted. For VIDD, `origin/ab-mvp` had neither
`Dockerfile` nor `scripts/entrypoint.sh`, so `git clone` on the box would have
produced nothing buildable. Prefer rsync while iterating, and commit only once
the build is green:

```bash
rsync -az --delete --exclude '.git' --exclude '__pycache__' --exclude '.idea' \
      --exclude 'output' --exclude '.mber' \
      -e "ssh -i ~/.ssh/id_ed25519" ~/repos/VIDD/ ubuntu@<ip>:/home/ubuntu/VIDD/
```

## Keeping the box alive

`bh-ai/stacks/cdk/stacks/ec2_auto_shutdown.py` puts a CloudWatch alarm on every
instance tagged `autoshutdown=true`: `CPUUtilization <= 3.0%` for **180
consecutive 1-minute periods (3 hours)**, action `arn:aws:automate:<region>:ec2:stop`.
Confirm yours exists — the stack enumerates instances at deploy time, so a very
new box may not have one yet:

```bash
aws cloudwatch describe-alarms --region us-east-1 \
  --query "MetricAlarms[?Dimensions[?Value=='<instance-id>']].{name:AlarmName,state:StateValue}"
```

Because `datapoints_to_alarm == evaluation_periods == 180`, **one** minute above
3% resets the streak, so a light periodic burn is enough. This matters far more
than usual when Docker's storage is on the instance store: a stop **destroys the
image and the whole host setup**, it does not pause them. Nothing here is durable
until it is pushed to ECR.

## Verify GPU passthrough before the real build

Cheap, and it fails fast if `nvidia-container-toolkit` did not configure:

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Pushing

`docker push` needs the ECR repo to already exist — the deploy script does not
create it. Check first:

```bash
aws ecr describe-repositories --region us-east-1 --repository-names <name>
```

ECR repos are declared as YAML under
`bh-ai/stacks/config/<env>/<region>/ecr/`, so a new one is an eng ask. Use
`PUSH=0` to build and smoke-test while that is in flight.

Launchers resolve images by **digest**, not tag
(`bh.aws_tools.sagemaker.get_image_uri` → `describe_images` →
`<registry>/<repo>@<digest>`), so re-pushing `:latest` does not silently change
what an already-queued job runs.
