import pandas as pd

import mongodb


client = mongodb.client
db_names = mongodb.db_names

col_reordered =  [
    '_id',
    'Review_Id',
    'Location',
    'DatePost',
    'Department',
    'Employee_status',
    'Company_name',
    'Title',
    'Pros',
    'Cons',
    'To_Management',
    'Ratings',
    'Culture',
    'WorkLifeBalance',
    'Benefits',
    'Management',
    'Opportunity',
    'Potential',
    'Recommend'
]

def get_collections(db_no):
    db = client.get_database(db_names[db_no])
    coll_names = {}
    for i, coll in enumerate(db.list_collection_names()):
        coll_names[i] = coll
    return coll_names

def get_df(coll, collection_no):
    cursor = coll.find()
    df = pd.DataFrame(list(cursor))
    return df[col_reordered]

def get_comp(df, company_name):
    df_ = df[df['Company_name'] == company_name]
    df_['DatePost'] = pd.to_datetime(df_['DatePost'], errors='coerce')
    df_['year'] = df_['DatePost'].apply(lambda x: x.year)
    return df_
