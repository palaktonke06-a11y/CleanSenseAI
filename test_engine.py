import pandas as pd
from cleansense.engine import clean_dataframe, create_insights, build_summary

def test_clean_dataframe_missing_and_duplicates():
    df=pd.DataFrame({'n':[1,None,1],'s':[' a ','b',' a ']})
    out,log=clean_dataframe(df,[{'type':'fill_missing_median','column':'n'},{'type':'standardize_text','column':'s'},{'type':'drop_duplicates'}])
    assert out['n'].isna().sum()==0
    assert out['s'].tolist()==['a','b']
    assert len(log)==3

def test_summary_and_insights():
    profile={'total_rows':10,'columns_count':2,'health_score':80,'missing_values':1,'missing_pct':5,'duplicate_rows':0,'duplicate_pct':0,'numeric_columns':1,'categorical_columns':1,'columns':[]}
    insights=create_insights(profile)
    assert insights and '10 rows' in build_summary(profile,insights)
