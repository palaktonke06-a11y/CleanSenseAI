from __future__ import annotations
import hashlib, json, math, os, re, sqlite3, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

SUPPORTED = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".parquet"}


def persist_upload(uploaded, dest_dir: Path, max_mb: int = 1024) -> Path:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    size = int(getattr(uploaded, "size", 0))
    if size > max_mb * 1024 * 1024:
        raise ValueError(f"File is larger than configured limit ({max_mb} MB).")
    path = dest_dir / (hashlib.sha256((uploaded.name + str(size)).encode()).hexdigest()[:12] + suffix)
    uploaded.seek(0)
    with path.open("wb") as f:
        while True:
            block = uploaded.read(8 * 1024 * 1024)
            if not block: break
            f.write(block)
    uploaded.seek(0)
    if path.stat().st_size == 0: raise ValueError("The uploaded file is empty.")
    return path


def _read_csv_sample(path: Path, n: int) -> pd.DataFrame:
    errors=[]
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try: return pd.read_csv(path, nrows=n, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e: errors.append(str(e))
    raise ValueError("Could not decode CSV with common encodings: " + "; ".join(errors[-1:]))


def read_sample(path: Path, n: int = 50000) -> pd.DataFrame:
    ext=path.suffix.lower()
    if ext == ".csv": return _read_csv_sample(path,n)
    if ext in (".xlsx", ".xls"): return pd.read_excel(path, nrows=n)
    if ext == ".parquet": return pd.read_parquet(path).head(n)
    if ext == ".jsonl":
        rows=[]
        with path.open("r",encoding="utf-8") as f:
            for i,line in enumerate(f):
                if i>=n: break
                if line.strip(): rows.append(json.loads(line))
        return pd.json_normalize(rows)
    obj=json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj,dict): obj=obj.get("data",obj)
    return pd.json_normalize(obj if isinstance(obj,list) else [obj]).head(n)


def count_rows(path: Path) -> int | None:
    if path.suffix.lower()==".csv":
        with path.open("rb") as f: return max(0,sum(1 for _ in f)-1)
    if path.suffix.lower()==".jsonl":
        with path.open("rb") as f: return sum(1 for line in f if line.strip())
    return None


def _sensitive_flags(s: pd.Series) -> list[str]:
    vals=s.dropna().astype(str).head(2000)
    if vals.empty:return []
    patterns={"Email":r"^[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}$","Phone-like":r"^[+\d][\d\s().-]{7,}$","Aadhaar-like":r"^\d{4}[ -]?\d{4}[ -]?\d{4}$","PAN-like":r"^[A-Z]{5}\d{4}[A-Z]$"}
    return [name for name,pat in patterns.items() if vals.str.match(pat,na=False).mean()>=.6]


def profile_dataset(path: Path, sample_rows: int=50000, chunk_size: int=100000) -> dict[str,Any]:
    sample=read_sample(path,sample_rows)
    if sample.empty or len(sample.columns)==0: raise ValueError("Dataset has no readable rows or columns.")
    total=count_rows(path)
    # Exact CSV aggregate missingness and duplicates are computed by streaming; duplicate count is exact only within this file.
    exact_missing=None; exact_duplicates=None
    if path.suffix.lower()==".csv":
        missing=0; rows=0; dup=0
        # Disk-backed fingerprint index makes duplicate counting global across chunk boundaries.
        with tempfile.TemporaryDirectory(prefix="cleansense_profile_") as tmp:
            db_path=Path(tmp)/"fingerprints.sqlite"
            conn=sqlite3.connect(db_path)
            conn.execute("CREATE TABLE seen (fingerprint TEXT PRIMARY KEY)")
            try:
                for chunk in pd.read_csv(path,chunksize=chunk_size,low_memory=False):
                    missing += int(chunk.isna().sum().sum()); rows += len(chunk)
                    fingerprints=[]
                    for row in chunk.itertuples(index=False,name=None):
                        payload=json.dumps([None if pd.isna(v) else str(v) for v in row],ensure_ascii=False,separators=(",",":"))
                        fingerprints.append((hashlib.sha256(payload.encode()).hexdigest(),))
                    before_count=conn.total_changes
                    conn.executemany("INSERT OR IGNORE INTO seen VALUES (?)",fingerprints)
                    inserted=conn.total_changes-before_count
                    dup += len(chunk)-inserted
                    conn.commit()
            finally: conn.close()
        exact_missing=missing; exact_duplicates=dup
        if total is None: total=rows
    cols=[]
    for name in sample.columns:
        s=sample[name]; non=s.dropna()
        cols.append({"name":str(name),"dtype":str(s.dtype),"missing":int(s.isna().sum()),"missing_pct":round(float(s.isna().mean()*100),2),"unique":int(s.nunique(dropna=True)),"unique_pct":round(float(s.nunique(dropna=True)/max(1,len(non))*100),2),"sensitive_flags":_sensitive_flags(s),"sample_values":[str(v)[:70] for v in non.head(3)]})
    sample_missing=int(sample.isna().sum().sum()); sample_dup=int(sample.duplicated().sum())
    denominator=max(1,sample.shape[0]*sample.shape[1]); miss_pct=(exact_missing/ max(1,(total or len(sample))*len(sample.columns))*100) if exact_missing is not None else sample_missing/denominator*100
    dup_pct=(exact_duplicates/max(1,total)*100) if exact_duplicates is not None else sample_dup/max(1,len(sample))*100
    score=max(0,min(100,round(100-.65*miss_pct-.35*dup_pct,1)))
    numeric=int(len(sample.select_dtypes(include=np.number).columns)); categorical=int(len(sample.select_dtypes(include=["object","category","bool"]).columns))
    return {"filename":path.name,"total_rows":int(total if total is not None else len(sample)),"total_rows_exact":total is not None,"sampled_rows":len(sample),"columns_count":len(sample.columns),"missing_values":int(exact_missing if exact_missing is not None else sample_missing),"missing_pct":round(miss_pct,2),"duplicate_rows":int(exact_duplicates if exact_duplicates is not None else sample_dup),"duplicate_pct":round(dup_pct,2),"health_score":score,"numeric_columns":numeric,"categorical_columns":categorical,"datetime_columns":int(len(sample.select_dtypes(include=["datetime","datetimetz"]).columns)),"columns":cols,"sample":sample,"sampled":bool(total is not None and total>len(sample)),"notice":"CSV row count, missing values and global exact duplicate count are streamed with a disk-backed fingerprint index. Column distributions and value-level signals use a profile sample." if path.suffix.lower()==".csv" else "For this format, profiling uses a bounded sample. Full row count and global aggregates may not be available."}


def create_insights(profile: dict) -> list[dict[str,str]]:
    out=[]
    if profile["missing_pct"]>0: out.append({"level":"warning","title":"Missing data detected","detail":f"{profile['missing_values']:,} missing cells ({profile['missing_pct']}%). Review column-wise fill strategies before modelling."})
    if profile["duplicate_rows"]>0: out.append({"level":"warning","title":"Duplicate records found","detail":f"{profile['duplicate_rows']:,} duplicate rows were identified in the profile scan."})
    sensitive=[c for c in profile["columns"] if c["sensitive_flags"]]
    if sensitive: out.append({"level":"warning","title":"Potential sensitive information","detail":"Potential personal-data patterns in: " + ", ".join(c["name"] for c in sensitive) + ". Review before sharing or exporting."})
    high=[c for c in profile["columns"] if c["unique_pct"]>=95 and c["unique"]>30]
    if high: out.append({"level":"info","title":"High-cardinality columns","detail":"Review these columns for IDs or near-unique values: " + ", ".join(c["name"] for c in high[:8])})
    if profile["health_score"]>=90: out.append({"level":"success","title":"Strong baseline data health","detail":"The current quality indicators look healthy. Check domain rules and leakage before training models."})
    if not out: out.append({"level":"success","title":"No major basic quality alerts","detail":"No missingness or duplicate alerts were detected. Continue with domain-specific validation."})
    return out


def clean_csv_chunked(source: Path, output: Path, actions: list[dict], chunksize: int=100000) -> dict:
    if source.suffix.lower()!=".csv":
        df=read_sample(source,10**9)
        cleaned,log=clean_dataframe(df,actions)
        output.parent.mkdir(parents=True,exist_ok=True); cleaned.to_csv(output,index=False)
        return {"rows_before":len(df),"rows_after":len(cleaned),"removed":len(df)-len(cleaned),"log":log,"mode":"in-memory format conversion"}
    output.parent.mkdir(parents=True,exist_ok=True)
    before=0; after=0; log=[]
    # A disk-backed SQLite fingerprint index catches duplicates across chunk boundaries.
    db=output.with_suffix(".dedupe.sqlite")
    conn=sqlite3.connect(db)
    conn.execute("CREATE TABLE IF NOT EXISTS seen (fingerprint TEXT PRIMARY KEY)")
    first=True
    try:
        for chunk in pd.read_csv(source,chunksize=chunksize,low_memory=False):
            before+=len(chunk)
            for action in actions:
                typ=action["type"]; col=action.get("column")
                if typ=="fill_missing_median" and col in chunk and pd.api.types.is_numeric_dtype(chunk[col]):
                    val=action.get("value")
                    if val is None: val=pd.to_numeric(chunk[col],errors="coerce").median()
                    chunk[col]=chunk[col].fillna(val)
                elif typ=="fill_missing_mode" and col in chunk:
                    val=action.get("value")
                    if val is None:
                        mode=chunk[col].mode(dropna=True); val=mode.iloc[0] if not mode.empty else "Unknown"
                    chunk[col]=chunk[col].fillna(val)
                elif typ=="standardize_text" and col in chunk and (pd.api.types.is_object_dtype(chunk[col]) or pd.api.types.is_string_dtype(chunk[col])):
                    chunk[col]=chunk[col].map(lambda x: re.sub(r"\s+"," ",x).strip() if isinstance(x,str) else x)
                elif typ=="drop_duplicates":
                    keep=[]
                    for row in chunk.itertuples(index=False,name=None):
                        fp=hashlib.sha256(json.dumps([None if pd.isna(v) else str(v) for v in row],ensure_ascii=False,separators=(",",":" )).encode()).hexdigest()
                        try: conn.execute("INSERT INTO seen VALUES (?)",(fp,)); keep.append(True)
                        except sqlite3.IntegrityError: keep.append(False)
                    chunk=chunk.loc[keep]
            after+=len(chunk)
            chunk.to_csv(output,index=False,mode="w" if first else "a",header=first); first=False
        conn.commit()
    finally:
        conn.close(); db.unlink(missing_ok=True)
    if first: pd.DataFrame().to_csv(output,index=False)
    return {"rows_before":before,"rows_after":after,"removed":before-after,"log":[{"action":a["type"],"column":a.get("column") or "All rows"} for a in actions],"mode":"chunked CSV processing with disk-backed duplicate index"}


def clean_dataframe(df: pd.DataFrame, actions: list[dict]) -> tuple[pd.DataFrame,list[dict]]:
    out=df.copy(); log=[]
    for a in actions:
        typ=a["type"]; col=a.get("column"); before=len(out)
        if typ=="fill_missing_median" and col in out:
            val=pd.to_numeric(out[col],errors="coerce").median(); out[col]=out[col].fillna(val)
        elif typ=="fill_missing_mode" and col in out:
            m=out[col].mode(dropna=True); out[col]=out[col].fillna(m.iloc[0] if len(m) else "Unknown")
        elif typ=="standardize_text" and col in out:
            out[col]=out[col].map(lambda x:re.sub(r"\s+"," ",x).strip() if isinstance(x,str) else x)
        elif typ=="drop_duplicates": out=out.drop_duplicates()
        elif typ=="drop_column" and col in out: out=out.drop(columns=[col])
        log.append({"action":typ,"column":col or "All rows","rows_removed":before-len(out)})
    return out,log


def build_summary(profile: dict, insights: list[dict]) -> str:
    return (f"The dataset contains {profile['total_rows']:,} rows and {profile['columns_count']:,} columns. "
            f"Its current Data Health Score is {profile['health_score']}/100, with {profile['missing_values']:,} missing cells "
            f"({profile['missing_pct']}%) and {profile['duplicate_rows']:,} duplicate rows ({profile['duplicate_pct']}%). "
            f"The profile sample contains {profile['numeric_columns']} numeric and {profile['categorical_columns']} categorical columns. "
            f"The scan generated {len(insights)} quality insight(s). Review sensitive-data flags before sharing the dataset.")
