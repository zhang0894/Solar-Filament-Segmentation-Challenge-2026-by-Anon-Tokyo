# Publishing this delivery

The release archive contains an explicit publication allowlist. It excludes credentials, original competition data, execution caches and private machine configuration. The trained checkpoints are distributed separately through the public Kaggle dataset linked in `artifacts/checkpoints.json`.

If the repository was delivered as a ZIP rather than pushed automatically, extract `AnonTokyo_Delivery.zip`, enter its `AnonTokyo/` directory, and authenticate Git on your machine. Then publish to the empty repository:

```bash
git init -b main
git add .
git commit -m "Release Anon Tokyo solar filament solution"
git remote add origin https://github.com/zhang0894/Solar-Filament-Segmentation-Challenge-2026-by-Anon-Tokyo.git
git push -u origin main
```

Use your existing Git author identity. If the remote has since acquired commits, clone it first and copy the delivery files into that checkout; do not force-push over existing work.

After publishing, open the repository and the report link in a signed-out browser. Complete the organizer's Google form using `ORGANIZER_CHECKLIST.md`. A Git push and a Kaggle prediction submission do not replace that form.
