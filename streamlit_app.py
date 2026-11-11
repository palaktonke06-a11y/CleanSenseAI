from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import pandas as pd
import plotly.express as px
import streamlit as st
from engine import persist_upload, profile_dataset, create_insights, clean_csv_chunked, clean_dataframe, build_summary, read_sample

st.set_page_config(page_title="CleanSense AI | Data Quality Studio",page_icon="🧹",layout="wide",initial_sidebar_state="expanded")
STORE=ROOT/"storage"; UPLOADS=STORE/"uploads"; OUTPUTS=STORE/"outputs"

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--cs-border:rgba(128,145,170,.24)}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.block-container{max-width:1540px;padding-top:1.35rem;padding-bottom:3rem}
section[data-testid="stSidebar"]{border-right:1px solid var(--cs-border)}
.cs-hero{position:relative;overflow:hidden;padding:28px 30px;border:1px solid var(--cs-border);border-radius:22px;background:linear-gradient(115deg,rgba(37,99,235,.18),rgba(16,185,129,.10),rgba(139,92,246,.10));margin-bottom:20px;animation:rise .65s ease-out}
.cs-hero:after{content:'';position:absolute;width:260px;height:260px;right:-70px;top:-145px;border-radius:50%;background:radial-gradient(circle,rgba(59,130,246,.28),transparent 68%);animation:glow 5s ease-in-out infinite alternate}
.cs-hero h1{font-family:'Space Grotesk',sans-serif;font-size:2.2rem;letter-spacing:-1.2px;margin:0}.cs-hero p{opacity:.78;margin:6px 0 0;font-size:1rem}
.cs-kpi{border:1px solid var(--cs-border);border-radius:17px;padding:17px 18px;min-height:120px;background:linear-gradient(145deg,rgba(128,145,170,.075),rgba(128,145,170,.025));transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;animation:rise .55s ease both}
.cs-kpi:hover{transform:translateY(-4px);border-color:rgba(59,130,246,.55);box-shadow:0 12px 28px rgba(15,23,42,.10)}.cs-kpi .label{font-size:.82rem;opacity:.72}.cs-kpi .value{font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700;margin-top:8px;line-height:1.15}.cs-kpi .hint{font-size:.75rem;opacity:.62;margin-top:7px}
.cs-panel{border:1px solid var(--cs-border);border-radius:18px;padding:17px 19px;background:rgba(128,145,170,.035);height:100%}
.cs-insight{border:1px solid var(--cs-border);border-left:4px solid #3b82f6;border-radius:12px;padding:13px 15px;margin:8px 0;background:rgba(128,145,170,.045)}.cs-insight.warning{border-left-color:#f59e0b}.cs-insight.success{border-left-color:#10b981}.cs-insight .t{font-weight:700}.cs-insight .d{font-size:.88rem;opacity:.78;margin-top:3px}
.cs-pill{display:inline-block;border:1px solid var(--cs-border);border-radius:999px;padding:4px 10px;font-size:.75rem;opacity:.8}
@keyframes rise{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:translateY(0)}}@keyframes glow{from{transform:translate(0,0)}to{transform:translate(-30px,22px)}}
.stTabs [data-baseweb="tab-list"]{gap:8px}.stTabs [data-baseweb="tab"]{border-radius:10px;padding:9px 15px}
</style>""",unsafe_allow_html=True)
st.markdown("<div class='cs-hero'><h1>🧹 CleanSense AI</h1><p>Explainable Data Quality · Smart Cleaning · ML Readiness Studio</p><span class='cs-pill'>LOCAL WORKSPACE</span> <span class='cs-pill'>AUDIT-FRIENDLY</span> <span class='cs-pill'>LARGE CSV AWARE</span></div>",unsafe_allow_html=True)

for key,default in {"source_path":None,"profile":None,"clean_path":None,"clean_result":None,"filename":None}.items():
    if key not in st.session_state: st.session_state[key]=default

def kpi(label,value,hint=""):
    st.markdown(f"<div class='cs-kpi'><div class='label'>{label}</div><div class='value'>{value}</div><div class='hint'>{hint}</div></div>",unsafe_allow_html=True)

def section(title,subtitle=None):
    st.markdown(f"### {title}")
    if subtitle: st.caption(subtitle)

with st.sidebar:
    st.markdown("## Workspace")
    st.success("Local processing engine ready")
    st.caption("Frontend and data engine run in one Streamlit process. No separate API terminal is required.")
    if st.button("＋ New dataset",use_container_width=True):
        for k in ("source_path","profile","clean_path","clean_result","filename"): st.session_state[k]=None
        st.rerun()
    st.divider()
    st.markdown("**Processing profile**")
    sample_limit=st.select_slider("Profile sample rows",options=[10000,25000,50000,100000,200000],value=50000,format_func=lambda x:f"{x:,}")
    chunk_size=st.select_slider("CSV processing chunk",options=[25000,50000,100000,200000],value=100000,format_func=lambda x:f"{x:,}")
    st.caption("Large CSVs are profiled and cleaned in chunks. Metrics based on samples are labelled clearly.")
    st.divider()
    st.caption("CleanSense AI v2.0 • Local-first • No paid AI service")

if not st.session_state.source_path:
    section("Start with your dataset","Upload a file to generate a data-quality profile, plain-language summary and actionable cleaning plan.")
    upload=st.file_uploader("Choose a dataset",type=["csv","xlsx","xls","json","jsonl","parquet"],help="Supported formats: CSV, Excel, JSON, JSONL and Parquet.")
    c1,c2,c3=st.columns(3)
    with c1:
        with st.container(border=True): st.markdown("**01 · Profile**"); st.write("Understand structure, missingness, duplicates, cardinality and sensitive-data signals.")
    with c2:
        with st.container(border=True): st.markdown("**02 · Improve**"); st.write("Review suggested actions, approve changes and keep a before/after record.")
    with c3:
        with st.container(border=True): st.markdown("**03 · Prepare**"); st.write("Check model readiness and export the cleaned dataset and audit summary.")
    if upload and st.button("Analyze dataset",type="primary",use_container_width=True):
        try:
            with st.spinner("Saving file and profiling data safely..."):
                path=persist_upload(upload,UPLOADS)
                prof=profile_dataset(path,sample_rows=sample_limit,chunk_size=chunk_size)
                st.session_state.source_path=str(path); st.session_state.profile=prof; st.session_state.filename=upload.name
            st.success("Dataset analyzed successfully."); st.rerun()
        except Exception as e: st.error(f"Could not analyze dataset: {e}")
    st.stop()

profile=st.session_state.profile; source=Path(st.session_state.source_path); insights=create_insights(profile)
section("Dataset command center",st.session_state.filename or source.name)
if profile.get("notice"): st.info(profile["notice"])
metrics=st.columns(5)
with metrics[0]: kpi("Total rows",f"{profile['total_rows']:,}","Exact count" if profile['total_rows_exact'] else "Sample-based count")
with metrics[1]: kpi("Columns",f"{profile['columns_count']:,}","Detected fields")
with metrics[2]: kpi("Missing cells",f"{profile['missing_values']:,}",f"{profile['missing_pct']}% of scanned cells")
with metrics[3]: kpi("Duplicate rows",f"{profile['duplicate_rows']:,}",f"{profile['duplicate_pct']}% of scanned rows")
with metrics[4]: kpi("Data health",f"{profile['health_score']}/100","Quality indicator")

summary=build_summary(profile,insights)
tab_overview,tab_profile,tab_clean,tab_ml,tab_export=st.tabs(["✦ Overview","▦ Data profile","⚙ Cleaning studio","◈ ML readiness","⇩ Exports"])
with tab_overview:
    left,right=st.columns([1.25,1])
    with left:
        with st.container(border=True):
            section("Dataset executive summary")
            st.write(summary)
            st.download_button("Download summary (.txt)",summary,file_name="cleansense_dataset_summary.txt",mime="text/plain")
        with st.container(border=True):
            section("Quality dimensions")
            qdf=pd.DataFrame({"Dimension":["Completeness","Uniqueness"],"Score":[max(0,100-profile['missing_pct']),max(0,100-profile['duplicate_pct'])]})
            fig=px.bar(qdf,x="Score",y="Dimension",orientation="h",range_x=[0,100],text="Score",color="Dimension",template="plotly_dark" if st.get_option("theme.base")=="dark" else "plotly_white")
            fig.update_layout(height=230,showlegend=False,margin=dict(l=8,r=12,t=8,b=8)); st.plotly_chart(fig,use_container_width=True)
    with right:
        with st.container(border=True):
            section("Smart quality insights")
            for item in insights:
                st.markdown(f"<div class='cs-insight {item['level']}'><div class='t'>{item['title']}</div><div class='d'>{item['detail']}</div></div>",unsafe_allow_html=True)
        with st.container(border=True):
            section("Column type mix")
            type_df=pd.DataFrame({"Type":["Numeric","Categorical","Other"],"Columns":[profile['numeric_columns'],profile['categorical_columns'],max(0,profile['columns_count']-profile['numeric_columns']-profile['categorical_columns'])]})
            fig=px.pie(type_df,names="Type",values="Columns",hole=.58,template="plotly_dark" if st.get_option("theme.base")=="dark" else "plotly_white")
            fig.update_layout(height=250,margin=dict(l=5,r=5,t=8,b=5)); st.plotly_chart(fig,use_container_width=True)
with tab_profile:
    section("Column-level intelligence","Column statistics are computed from the configured profile sample.")
    cdf=pd.DataFrame(profile["columns"])
    show_cols=[c for c in ["name","dtype","missing","missing_pct","unique","unique_pct","sensitive_flags","sample_values"] if c in cdf.columns]
    st.dataframe(cdf[show_cols],use_container_width=True,hide_index=True,height=min(560,100+len(cdf)*38))
    if not cdf.empty:
        fig=px.bar(cdf.sort_values("missing_pct",ascending=False).head(20),x="name",y="missing_pct",title="Top columns by missingness",template="plotly_dark" if st.get_option("theme.base")=="dark" else "plotly_white")
        fig.update_layout(height=330,xaxis_title="Column",yaxis_title="Missing (%)"); st.plotly_chart(fig,use_container_width=True)
    sensitive=cdf[cdf.sensitive_flags.map(bool)] if "sensitive_flags" in cdf else pd.DataFrame()
    if not sensitive.empty: st.warning("Potential sensitive columns detected. Review access and masking requirements before exporting or sharing.")
with tab_clean:
    section("Review → approve → apply","Select only the operations you want. Column deletion is deliberately separate and requires confirmation.")
    cdf=pd.DataFrame(profile["columns"]); actions=[]
    for _,r in cdf.iterrows():
        col=r["name"]
        if r["missing_pct"]>0:
            if str(r["dtype"]).startswith(("int","float")): actions.append({"type":"fill_missing_median","column":col,"label":f"Fill missing values in {col} (median)","risk":"Low"})
            else: actions.append({"type":"fill_missing_mode","column":col,"label":f"Fill missing values in {col} (mode)","risk":"Low"})
        if str(r["dtype"]) in ("object","string"): actions.append({"type":"standardize_text","column":col,"label":f"Normalize whitespace in {col}","risk":"Low"})
    if profile["duplicate_rows"]>0: actions.append({"type":"drop_duplicates","column":None,"label":"Remove exact duplicate rows (global CSV deduplication)","risk":"Moderate"})
    if actions:
        selected=[]
        for i,a in enumerate(actions):
            if st.checkbox(a["label"],key=f"clean_{source.name}_{i}"): selected.append(a)
        if selected:
            st.info(f"{len(selected)} operation(s) selected. Review the list above before applying.")
            confirm=st.checkbox("I approve these cleaning operations and understand they create a new output file.",key="approve_clean")
            if st.button("Apply approved cleaning",type="primary",disabled=not confirm):
                out=OUTPUTS/(source.stem+"_cleansense_cleaned.csv")
                try:
                    with st.spinner("Cleaning dataset. Large CSVs are processed chunk by chunk..."):
                        result=clean_csv_chunked(source,out,[{"type":a["type"],"column":a["column"]} for a in selected],chunksize=chunk_size)
                    st.session_state.clean_path=str(out); st.session_state.clean_result=result
                    st.success("Cleaning completed. Original file was preserved.")
                except Exception as e: st.error(f"Cleaning failed: {e}")
    else: st.success("No basic missing-value, text-whitespace or duplicate actions were detected in the profile.")
    if st.session_state.clean_result:
        r=st.session_state.clean_result
        a,b,c=st.columns(3)
        with a: kpi("Rows before",f"{r['rows_before']:,}")
        with b: kpi("Rows after",f"{r['rows_after']:,}")
        with c: kpi("Rows removed",f"{r['removed']:,}")
        st.dataframe(pd.DataFrame(r["log"]),use_container_width=True,hide_index=True)
with tab_ml:
    section("ML readiness assessment","This is a bounded-sample baseline, not a guarantee of model performance. Preprocessing is fitted inside each CV fold.")
    sample=profile["sample"].copy(); target=st.selectbox("Select target column",list(sample.columns),key="ml_target")
    task=st.selectbox("Problem type",["Auto detect","Classification","Regression"],key="ml_task")
    if st.button("Run baseline evaluation",type="primary"):
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder,StandardScaler
            from sklearn.model_selection import cross_validate,StratifiedKFold,KFold
            from sklearn.linear_model import LogisticRegression,Ridge
            from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,HistGradientBoostingRegressor
            from sklearn.metrics import make_scorer, f1_score
            X=sample.drop(columns=[target]); y=sample[target]; valid=y.notna(); X=X.loc[valid].copy(); y=y.loc[valid]
            if len(X)<20: raise ValueError("At least 20 non-missing target rows are required for baseline evaluation.")
            is_class=(task=="Classification" or (task=="Auto detect" and (y.dtype=="object" or y.nunique()<=20)))
            if is_class and y.nunique()<2: raise ValueError("Classification needs at least two target classes.")
            nums=list(X.select_dtypes(include="number").columns); cats=[c for c in X.columns if c not in nums]
            prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore",max_categories=100))]),cats)],remainder="drop")
            if is_class:
                from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
                min_class=int(y.value_counts().min()); folds=min(5,min_class)
                if folds<2: raise ValueError("At least two records per class are needed for stratified CV.")
                cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=42); models=[("Logistic Regression",LogisticRegression(max_iter=700,class_weight="balanced")),("Random Forest",RandomForestClassifier(n_estimators=120,max_depth=12,n_jobs=-1,class_weight="balanced_subsample",random_state=42))]
                scoring={"accuracy":"accuracy","precision":"precision_weighted","recall":"recall_weighted","f1":"f1_weighted"}
            else:
                cv=KFold(n_splits=5,shuffle=True,random_state=42); models=[("Ridge Regression",Ridge()),("Random Forest",RandomForestRegressor(n_estimators=100,max_depth=12,n_jobs=-1,random_state=42))]; scoring={"r2":"r2","mae":"neg_mean_absolute_error","rmse":"neg_root_mean_squared_error"}
            rows=[]
            for name,model in models:
                pipe=Pipeline([("prep",prep),("model",model)])
                scores=cross_validate(pipe,X,y,cv=cv,scoring=scoring,n_jobs=1,error_score="raise")
                row={"Model":name}
                for metric in scoring:
                    val=float(scores["test_"+metric].mean()); row[metric.upper()]=round(-val if metric in ("mae","rmse") else val,4)
                rows.append(row)
            res=pd.DataFrame(rows).sort_values("F1",ascending=False) if is_class else pd.DataFrame(rows).sort_values("R2",ascending=False)
            st.success(f"Evaluation completed on {len(X):,} sampled rows using {folds if is_class else 5}-fold cross-validation.")
            st.dataframe(res,use_container_width=True,hide_index=True)
        except Exception as e: st.error(f"Baseline evaluation could not complete: {e}")
with tab_export:
    section("Export center","Download the original or cleaned data, plus a compact quality summary.")
    summary=build_summary(profile,insights)
    st.download_button("Download dataset summary",summary,file_name="cleansense_summary.txt",mime="text/plain",use_container_width=True)
    if st.session_state.clean_path and Path(st.session_state.clean_path).exists():
        p=Path(st.session_state.clean_path)
        st.download_button("Download cleaned CSV",p.read_bytes(),file_name=p.name,mime="text/csv",type="primary",use_container_width=True)
    else: st.info("Apply approved cleaning operations to enable the cleaned-data download.")
    if st.button("Generate profile JSON"):
        export={k:v for k,v in profile.items() if k!="sample"}
        st.download_button("Download profile JSON",pd.Series(export).to_json(indent=2),file_name="cleansense_profile.json",mime="application/json")

st.divider()
st.caption("CleanSense AI · Local-first data quality studio · Original uploads are preserved · Profile sample limitations are disclosed")
