"""download exactly the eight study snapshots. no credentials required."""
from pathlib import Path
import urllib.request,json,zipfile,hashlib,datetime
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'data'/'raw';RAW.mkdir(parents=True,exist_ok=True)
DATES=['2019-10-31','2020-10-31','2021-04-12','2022-11-16','2023-04-26','2023-10-25','2024-10-23','2025-11-20','2026-07-15']
records=[]
for date in DATES:
 url=f'https://data.cms.gov/provider-data/sites/default/files/dataset-archives/theme/dialysis-facilities/dialysis-facilities_{date}.zip'
 target=RAW/f'facility_{date}.csv'
 with urllib.request.urlopen(url,timeout=120) as r:
  import io
  content=r.read()
 with zipfile.ZipFile(io.BytesIO(content)) as z:
  n=next(n for n in z.namelist() if n=='23ew-n7w9.csv' or n=='DFC_SOCRATA_FAC_DATA.csv' or ('DFC_FACILITY.csv' in n and 'ICH' not in n))
  target.write_bytes(z.read(n))
 records.append(dict(snapshot=date,url=url,archive_member=n,file=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 print(date,'downloaded',flush=True)
(RAW/'sources.json').write_text(json.dumps(records,indent=2))
